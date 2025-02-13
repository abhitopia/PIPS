import torch
from pips.dvae import GridDVAEConfig, GridDVAE
import time
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import Trainer
from torch.optim import AdamW

torch.set_float32_matmul_precision('high')


class MinimalDVAEModule(pl.LightningModule):
    def __init__(self, config: GridDVAEConfig):
        super().__init__()
        self.config = config
        self.model = GridDVAE(config)
        self.padding_idx = config.n_vocab - 1
        self.automatic_optimization = False  # We'll handle optimization manually for testing
        self.register_buffer('q_z_marg', None, persistent=True)
        
    def forward(self, x, train=True):
        # Fixed parameters for testing
        reinMax = True
        tau = 0.9
        hard = True
        
        # Sample mask percentage
        mask_pct = 0.0
        if train:
            max_mask_pct = 0.5  # Fixed value for testing
            mask_pct = torch.empty(1, device=x.device).uniform_(0.0, max_mask_pct)[0]
        
        # Forward pass with q_z_marg
        logits, reconstruction_loss, kld_losses, updated_q_z_marg = self.model.forward(
            x, 
            q_z_marg=self.q_z_marg,
            mask_percentage=mask_pct, 
            hard=hard, 
            reinMax=reinMax,
            tau=tau
        )

        # Update the global q_z_marg estimate
        self.q_z_marg = updated_q_z_marg

        # Calculate accuracies
        predictions = logits.argmax(dim=-1)
        correct_tokens = (predictions == x).float()
        token_accuracy = correct_tokens.mean()

        # Calculate token accuracy excluding padding tokens
        non_padding_mask = (x != self.padding_idx)
        acc_no_pad = (correct_tokens * non_padding_mask).sum() / non_padding_mask.sum()

        # Calculate sample accuracy
        sample_correct = correct_tokens.all(dim=1).float()
        sample_accuracy = sample_correct.mean()

        # Normalize reconstruction loss by number of tokens
        reconstruction_loss = reconstruction_loss / x.size(1)

        # Fixed beta values for testing
        beta_values = {
            'beta(MI)': 1.0,
            'beta(DWKL)': 1.0,
            'beta(TC)': 1.0,
            'beta(KL)': 1.0
        }

        # Compute loss components
        loss_components = {
            'loss(CE)': reconstruction_loss,
            'loss(MI)': kld_losses['mi_loss'] * beta_values['beta(MI)'],
            'loss(DWKL)': kld_losses['dwkl_loss'] * beta_values['beta(DWKL)'],
            'loss(TC)': kld_losses['tc_loss'] * beta_values['beta(TC)'],
            'loss(KL)': kld_losses['kl_loss'] * beta_values['beta(KL)']
        }
        
        total_loss = sum(loss_components.values())

        return {
            'loss': total_loss,
            **{k: v.detach() for k, v in loss_components.items()},
            'mask_pct': mask_pct,
            'token_accuracy': token_accuracy.detach(),
            'acc_no_pad': acc_no_pad.detach(),
            'sample_accuracy': sample_accuracy.detach(),
            'logits': logits
        }
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-4)
        return optimizer


def test_dvae_compile():
    # Create minimal config
    config = GridDVAEConfig(
        n_dim=256,
        n_head=8,
        n_layers=1,  # Minimum layers
        n_codes=16,
        codebook_size=512,
        n_vocab=16,
        max_grid_height=32,
        max_grid_width=32,
        dropout=0.0
    )

    # Create model 
    model = MinimalDVAEModule(config)
    
    # Move to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = model.configure_optimizers()
    
    print("Compiling model...")
    model = torch.compile(
        model,
        mode="reduce-overhead",
        fullgraph=True,
    )
    
    # Create a minimal batch
    batch_size = 2
    seq_length = config.max_grid_height * config.max_grid_width  # Should be 1024
    x = torch.randint(0, config.n_vocab, (batch_size, seq_length), device=device, requires_grad=False)
    
    print("Running first forward pass...")
    try:
        # First forward pass (includes compilation)
        outputs = model(x)
        print("First forward pass successful!")
        print("\nOutputs:")
        for k, v in outputs.items():
            if k != 'logits':
                print(f"- {k}: {v.item() if torch.is_tensor(v) and v.numel() == 1 else v}")
        
        # Test backward pass
        print("\nTesting backward pass...")
        loss = outputs['loss']
        loss.backward()
        print("Backward pass successful!")
        
        # Check if gradients exist
        has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
        print(f"All parameters have gradients: {has_grad}")
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Now run multiple iterations with forward and backward
        num_iters = 100
        print(f"\nRunning timing test for {num_iters} iterations...")
        
        def run_benchmark(include_backward=True, include_optimize=True):
            # Warmup
            for _ in range(3):
                # Create new random input with requires_grad=False
                x = torch.randint(0, config.n_vocab, (batch_size, seq_length), 
                                 device=device, requires_grad=False)
                outputs = model(x)
                if include_backward:
                    loss = outputs['loss']
                    loss.backward()
                    if include_optimize:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Timing loop
            start_time = time.perf_counter()
            
            for i in range(num_iters):
                # Create new random input with requires_grad=False
                x = torch.randint(0, config.n_vocab, (batch_size, seq_length), 
                                 device=device, requires_grad=False)
                
                # Forward pass
                outputs = model(x)
                
                if include_backward:
                    loss = outputs['loss']
                    # Backward pass
                    loss.backward()
                    
                    if include_optimize:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        # Optimizer step and zero grad
                        optimizer.step()
                        optimizer.zero_grad()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            
            # Calculate statistics
            total_time = end_time - start_time
            avg_time = total_time / num_iters
            
            components = "forward"
            if include_backward:
                components += " + backward"
            if include_optimize:
                components += " + optimize"
            
            print(f"\nTiming Results ({components}):")
            print(f"Total time: {total_time:.4f} seconds")
            print(f"Average time per iteration: {avg_time*1000:.2f} ms")
            print(f"Iterations per second: {num_iters/total_time:.2f}")
        
        # Run different benchmark configurations
        print("\nBenchmark 1: Forward only")
        run_benchmark(include_backward=False, include_optimize=False)
        
        print("\nBenchmark 2: Forward + Backward")
        run_benchmark(include_backward=True, include_optimize=False)
        
        print("\nBenchmark 3: Forward + Backward + Optimize")
        run_benchmark(include_backward=True, include_optimize=True)
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    test_dvae_compile() 