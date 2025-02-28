import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import torch.nn.functional as F
import math

def cross_entropy_from_softmax(soft_pred, target, eps=1e-8):
    """
    Compute cross entropy loss when predictions are already softmaxed.
    
    Args:
        soft_pred: Tensor of softmax probabilities [*, num_classes]
        target: Tensor of target indices [*]
        eps: Small value for numerical stability
    """
    # Convert targets to one-hot
    target_one_hot = F.one_hot(target, num_classes=soft_pred.size(-1)).float()
    
    # Compute cross entropy manually: -sum(target * log(pred))
    loss = -(target_one_hot * torch.log(soft_pred + eps)).sum(dim=-1)
    return loss.mean()

def cosine_annealing(epoch, n_epochs, start_val, end_val):
    """
    Cosine annealing schedule from start_val to end_val.
    Returns end_val for epoch >= n_epochs.
    """
    if epoch >= n_epochs:
        return end_val
    
    cos_out = math.cos(math.pi * epoch / n_epochs) + 1  # Range [0, 2]
    return end_val + 0.5 * (start_val - end_val) * cos_out

def test_encoder_overfitting(dvae, batch_size=8, n_epochs=1500, n_anneal=1000,
                             tau_start=1.0, tau_end=0.1,
                             hardness_start=0.0, hardness_end=1.0, reinMax=False):
    """Test if the encoder can overfit to a single batch by learning to map inputs to target codes"""
    # Create random input indices and target one-hot codes
    x = torch.randint(0, dvae.config.n_vocab-2, (batch_size, dvae.n_pos))
    x = x.to(next(dvae.parameters()).device)
    
    # Create random target indices
    target_indices = torch.randint(0, dvae.config.codebook_size, (batch_size, dvae.config.n_codes))
    target_indices = target_indices.to(x.device)
    
    # Create optimizer for encoder parameters only
    encoder_params = list(dvae.encoder_base.parameters()) + \
                     list(dvae.encoder_bottleneck.parameters()) + \
                     list(dvae.encoder_head.parameters())
    optimizer = optim.Adam(encoder_params, lr=1e-3)
    
    print("Testing encoder overfitting...")
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Cosine annealing for both tau and hardness
        tau = cosine_annealing(epoch, n_epochs-500, tau_start, tau_end)
        hardness = cosine_annealing(epoch, n_anneal, hardness_start, hardness_end)
        
        # Forward pass through encoder
        _, soft_code = dvae.encode(x, tau=tau, hardness=hardness, reinMax=reinMax)
        
        # Compute cross entropy loss on softmaxed values
        loss = cross_entropy_from_softmax(
            soft_code.reshape(-1, dvae.config.codebook_size),
            target_indices.reshape(-1)
        )
        
        # Compute entropy and perplexity
        epsilon = 1e-8
        entropy_vals = -(soft_code * torch.log(soft_code + epsilon)).sum(dim=-1)  # Shape: [Batch, n_codes]
        avg_entropy = entropy_vals.mean()
        avg_perplexity = torch.exp(entropy_vals).mean()
        
        # For the first epoch, check that avg_entropy is within 10% of ln(codebook_size)
        if epoch == 0:
            expected_entropy = math.log(dvae.config.codebook_size)
            lower_bound = 0.9 * expected_entropy
            upper_bound = 1.1 * expected_entropy
            if not (lower_bound <= avg_entropy.item() <= upper_bound):
                print(f"WARNING: Initial avg entropy {avg_entropy.item():.4f} is outside expected range [{lower_bound:.4f}, {upper_bound:.4f}]")
            else:
                print(f"Initial avg entropy {avg_entropy.item():.4f} is within the expected range.")

        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
            print(f'Tau: {tau:.4f}, Hardness: {hardness:.4f}')
            print(f'Avg Entropy: {avg_entropy.item():.4f}, Avg Perplexity: {avg_perplexity.item():.4f}')
            # Print accuracy
            pred = soft_code.argmax(dim=-1)
            accuracy = (pred == target_indices).float().mean()
            print(f'Accuracy: {accuracy.item():.4f}')

def test_decoder_overfitting(dvae, batch_size=8, n_epochs=1000):
    """Test if the decoder can overfit to a single batch by learning to map codes to target outputs"""
    # Create random one-hot codes as input
    code_indices = torch.randint(0, dvae.config.codebook_size, (batch_size, dvae.config.n_codes))
    codes = F.one_hot(code_indices, num_classes=dvae.config.codebook_size).float()
    codes = codes.to(next(dvae.parameters()).device)
    
    # Create random target outputs
    target = torch.randint(0, dvae.config.n_vocab-2, (batch_size, dvae.n_pos))
    target = target.to(codes.device)
    
    # Create optimizer for decoder parameters (including codebook)
    decoder_params = list(dvae.decoder_base.parameters()) + \
                    list(dvae.decoder_bottleneck.parameters()) + \
                    list(dvae.decoder_head.parameters()) + \
                    [dvae.codebook]
    optimizer = optim.Adam(decoder_params, lr=1e-3)
    
    print("\nTesting decoder overfitting...")
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward pass through decoder
        decoded_logits = dvae.decode(codes)
        
        # Compute cross entropy loss
        loss = dvae.reconstruction_loss(decoded_logits, target)
        loss = loss / dvae.n_pos

        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
            # Print accuracy
            pred = decoded_logits.argmax(dim=-1)
            accuracy = (pred == target).float().mean()
            print(f'Accuracy: {accuracy.item():.4f}')

def test_dvae_overfitting(dvae, batch_size=4, n_epochs=1500, n_anneal=1000,
                         tau_start=1.0, tau_end=0.1,
                         hardness_start=0.0, hardness_end=1.0, reinMax=False):
    """Test if the full DVAE can overfit to a single batch"""
    # Create random input indices
    x = torch.randint(0, dvae.config.n_vocab-2, (batch_size, dvae.n_pos))
    x = x.to(next(dvae.parameters()).device)
    
    # Create optimizer for all parameters
    optimizer = optim.AdamW(dvae.parameters(), lr=1e-4, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
    
    print("\nTesting full DVAE overfitting...")
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Cosine annealing for both tau and hardness
        tau = cosine_annealing(epoch, n_epochs-500, tau_start, tau_end)
        hardness = cosine_annealing(epoch, n_anneal, hardness_start, hardness_end)
        
        # Forward pass
        code, soft_code = dvae.encode(x, tau=tau, hardness=-1, reinMax=reinMax)
        decoded_logits = dvae.decode(code)
        
        # Compute reconstruction loss
        loss = dvae.reconstruction_loss(decoded_logits, x)
        loss = loss / dvae.n_pos  # Normalize by sequence length
        
        # Compute entropy and perplexity
        epsilon = 1e-8
        entropy_vals = -(soft_code * torch.log(soft_code + epsilon)).sum(dim=-1)  # Shape: [Batch, n_codes]
        avg_entropy = entropy_vals.mean()
        avg_perplexity = torch.exp(entropy_vals).mean()
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(dvae.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
            print(f'Tau: {tau:.4f}, Hardness: {hardness:.4f}')
            print(f'Avg Entropy: {avg_entropy.item():.4f}, Avg Perplexity: {avg_perplexity.item():.4f}')
            # Print accuracy
            pred = decoded_logits.argmax(dim=-1)
            accuracy = (pred == x).float().mean()
            print(f'Accuracy: {accuracy.item():.4f}')
            
            # Print discrete codes for each sample
            discrete_codes = soft_code.argmax(dim=-1)  # [B, n_codes]
            print("\nCodes for each sample:")
            for i in range(batch_size):
                print(f"Sample {i}: {discrete_codes[i].tolist()}")

def test_training_module_overfitting(dvae_module, x, n_epochs=1500):
    """Test if the full DVAETrainingModule can overfit to a single batch"""
    # Create random input indices
    # x = torch.randint(0, dvae_module.model_config.n_vocab-2, 
    #                  (batch_size, dvae_module.model_config.max_grid_height * dvae_module.model_config.max_grid_width))
    x = x.to(next(dvae_module.parameters()).device)
    
    # Create a mock trainer with the global_step we want
    class MockTrainer:
        def __init__(self):
            self.global_step = 4
    
    dvae_module._trainer = MockTrainer()
    MIN_LR_FACTOR = 0.01

     # Learning rate schedule with warmup and cosine decay
    def lr_lambda(step):
        if step < n_epochs:
            # Linear warmup
            return float(step) / float(max(1, n_epochs))
        else:
            # Cosine decay from 1.0 to MIN_LR_FACTOR
            progress = float(step - n_epochs) / float(max(1, 1000000 - n_epochs))
            return MIN_LR_FACTOR + 0.5 * (1.0 - MIN_LR_FACTOR) * (1.0 + np.cos(np.pi * progress))

        # Create optimizer
    optimizer = optim.Adam(dvae_module.parameters(), lr=1e-3)
    # scheduler = LambdaLR(optimizer, lr_lambda)
    

    
    print("\nTesting DVAETrainingModule overfitting...")
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward pass using the module's forward method
        output_dict, updated_q_z_marg = dvae_module(x, train=True)
        dvae_module.trainer.global_step = dvae_module.trainer.global_step + 1

        hardness = output_dict['hardness']
        tau = output_dict['tau']
        reinMax = dvae_module.experiment_config.reinMax and hardness >= 0
        avg_entropy = output_dict['avg_entropy']
        avg_perplexity = output_dict['avg_perplexity']
        accuracy = output_dict['accuracy(TOKENS)']
        
        loss = output_dict['loss']
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(dvae_module.parameters(), max_norm=1.0)
        
        optimizer.step()
        # scheduler.step()
        
        # Update q_z_marg if provided
        if updated_q_z_marg is not None:
            dvae_module.q_z_marg.copy_(updated_q_z_marg.detach())
        
        if epoch % 20 == 0:
            print(f'\n\nEpoch [{epoch}/{n_epochs}], Loss: {loss.item():.4f}')
            print(f'Tau: {tau:.4f}, Hardness: {hardness:.4f}')
            print(f'Avg Entropy: {avg_entropy:.4f}, Avg Perplexity: {avg_perplexity:.4f}')
            print(f'Accuracy: {accuracy:.4f}')
            # Get discrete codes from the last forward pass
            # _, soft_code = dvae_module.model.encode(x, tau=tau, hardness=hardness, reinMax=reinMax)
            # discrete_codes = soft_code.argmax(dim=-1)  # [B, n_codes]
            # print("\nCodes for each sample:")
            # for i in range(batch_size):
            #     print(f"Sample {i}: {discrete_codes[i].tolist()}")

if __name__ == "__main__":
    # Example usage
    from pips.dvae import GridDVAE, GridDVAEConfig
    from train_dvae import DVAETrainingModule, ExperimentConfig
    
    from train_dvae import create_dataloaders


    # Create configs
    model_config = GridDVAEConfig(
        n_dim=64,
        n_head=4,
        n_grid_layer=2,
        n_latent_layers=2,
        n_codes=16,
        n_vocab=16,
        max_grid_height=32,
        max_grid_width=32,
        rope_base=10_000,
        dropout=0.0,
        codebook_size=512,
    )
    dvae = GridDVAE(model_config)
    # dvae = dvae.to(device)

    sample = False
        
    # Test encoder
    batch_size = 8
    tau_start = 1.0
    tau_end = 0.0625
    hardness_start =  0.0 if sample else -1
    hardness_end = 1.0 if sample else -1
    mask_percentage = 0.0
    n_epochs = 1_000
    n_anneal = 1000
    reinMax = False
    # test_encoder_overfitting(batch_size=8, dvae=dvae, tau_start=tau_start, tau_end=tau_end, 
    #                        hardness_start=hardness_start, hardness_end=hardness_end, 
    #                        n_epochs=n_epochs, reinMax=reinMax)
    
    # # Test decoder
    # # test_decoder_overfitting(dvae)
    # test_encoder_overfitting(batch_size=8, dvae=dvae, tau_start=tau_start, tau_end=tau_end, 
    #                        hardness_start=hardness_start, hardness_end=hardness_end, 
    #                        n_epochs=n_epochs, reinMax=reinMax)
    
    # # Test decoder
    # # test_decoder_overfitting(dvae)
    # test_encoder_overfitting(batch_size=8, dvae=dvae, tau_start=tau_start, tau_end=tau_end, 
    # Test full DVAE
    # test_dvae_overfitting(dvae, batch_size=batch_size, tau_start=tau_start, tau_end=tau_end,
    #                      hardness_start=hardness_start, hardness_end=hardness_end,
    #                      n_epochs=n_epochs, n_anneal=n_anneal, reinMax=reinMax)

    

    
    # # Create training module
    experiment_config = ExperimentConfig(
        batch_size=batch_size,
        model_config=model_config,
        # seed=42,
        max_mask_pct=mask_percentage,
        hardness_start=hardness_start,
        hardness=hardness_end,
        tau_start=tau_start,
        tau=tau_end,
        warmup_steps_hardness=n_anneal,
        warmup_steps_tau=n_anneal,
        warmup_steps_lr=n_epochs,
    )

    train_loader, val_loader = create_dataloaders(experiment_config, permute_train=False)
    for idx, batch in enumerate(train_loader):
        x, _, _  = batch

        if idx == 2:
            break

    print("batch:", x)
    dvae_module = DVAETrainingModule(experiment_config)
    dvae_module.configure_model()
    
        # Test overfitting
    test_training_module_overfitting(
        dvae_module,
        x,
        n_epochs=n_epochs,
    )