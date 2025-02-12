import torch
from pips.dvae import GridDVAEConfig, GridDVAE

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
    model = GridDVAE(config)
    
    # Move to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print("Compiling model...")
    model = torch.compile(model)
    
    # Create a minimal batch
    batch_size = 2
    seq_length = config.max_grid_height * config.max_grid_width  # Should be 1024
    x = torch.randint(0, config.n_vocab, (batch_size, seq_length), device=device)
    
    print("Running forward pass...")
    try:
        # Run forward pass with minimal parameters
        decoded_logits, reconstruction_loss, kld_losses = model(
            x,
            tau=0.9,
            hard=True,
            mask_percentage=0.0
        )
        print("Forward pass successful!")
        print(f"Output shapes:")
        print(f"- decoded_logits: {decoded_logits.shape}")
        print(f"- reconstruction_loss: {reconstruction_loss.item():.4f}")
        print(f"- kld_losses: {kld_losses}")
        
    except Exception as e:
        print(f"Error during forward pass: {str(e)}")
        raise

if __name__ == "__main__":
    test_dvae_compile() 