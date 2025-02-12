import torch
from pips.dvae import GridDVAEConfig, GridDVAE
import time
import pytorch_lightning as pl

torch.set_float32_matmul_precision('high')


class MinimalDVAEModule(pl.LightningModule):
    def __init__(self, config: GridDVAEConfig):
        super().__init__()
        self.config = config
        self.model = GridDVAE(config)
        
    def forward(self, x):
        return self.model(
            x,
            tau=0.9,
            hard=True,
            mask_percentage=0.0
        )
    
    def training_step(self, batch, batch_idx):
        x = batch
        decoded_logits, reconstruction_loss, kld_losses = self(x)
        loss = reconstruction_loss + sum(kld_losses.values())
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


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
    
    print("Compiling model...")
    # Use different compilation mode to avoid SM warning
    model = torch.compile(
        model,
        mode="reduce-overhead",
        fullgraph=True,
        # options={
        #     "max_autotune": False,
        #     "trace.enabled": True,
        #     "trace.graph_diagram": True,
        # }
    )
    
    # Create a minimal batch
    batch_size = 2
    seq_length = config.max_grid_height * config.max_grid_width  # Should be 1024
    x = torch.randint(0, config.n_vocab, (batch_size, seq_length), device=device)
    
    print("Running first forward pass...")
    try:
        # First forward pass (includes compilation)
        decoded_logits, reconstruction_loss, kld_losses = model(x)
        print("First forward pass successful!")
        print(f"Output shapes:")
        print(f"- decoded_logits: {decoded_logits.shape}")
        print(f"- reconstruction_loss: {reconstruction_loss.item():.4f}")
        print(f"- kld_losses: {kld_losses}")
        
        # Now run multiple iterations and time them
        num_iters = 100
        print(f"\nRunning timing test for {num_iters} iterations...")
        
        # Warmup
        for _ in range(3):
            model(x)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Timing loop
        start_time = time.perf_counter()
        
        for _ in range(num_iters):
            decoded_logits, reconstruction_loss, kld_losses = model(x)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        
        # Calculate statistics
        total_time = end_time - start_time
        avg_time = total_time / num_iters
        
        print(f"\nTiming Results:")
        print(f"Total time: {total_time:.4f} seconds")
        print(f"Average time per iteration: {avg_time*1000:.2f} ms")
        print(f"Iterations per second: {num_iters/total_time:.2f}")
        
        # Test training step
        print("\nTesting training step...")
        loss = model.training_step(x, 0)
        print(f"Training step successful! Loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    test_dvae_compile() 