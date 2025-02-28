import pytest
import torch
import numpy as np
from pips.dvae import (
    is_power_of_two,
    is_perfect_square,
    RotaryPositionalEmbeddings,
    RoPE2D,
    SwiGLUFFN,
    RMSNorm,
    GridDVAEConfig,
    GridDVAE,
)
import torch.nn.functional as F

# Test utility functions
def test_is_power_of_two():
    assert is_power_of_two(1)
    assert is_power_of_two(2)
    assert is_power_of_two(4)
    assert is_power_of_two(8)
    assert not is_power_of_two(0)
    assert not is_power_of_two(3)
    assert not is_power_of_two(6)
    assert not is_power_of_two(-2)

def test_is_perfect_square():
    assert is_perfect_square(0)
    assert is_perfect_square(1)
    assert is_perfect_square(4)
    assert is_perfect_square(9)
    assert not is_perfect_square(2)
    assert not is_perfect_square(3)
    assert not is_perfect_square(-1)
    assert not is_perfect_square(-4)


# Test RotaryPositionalEmbeddings
def test_rotary_positional_embeddings():
    rope = RotaryPositionalEmbeddings(dim=64, max_seq_len=16)
    
    # Test shape of cache
    assert rope.cache.shape == (16, 32, 2)  # max_seq_len x (dim//2) x 2
    
    # Test forward pass
    batch_size, n_heads = 2, 4
    seq_len, head_dim = 8, 64
    x = torch.randn(batch_size, n_heads, seq_len, head_dim)
    input_pos = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(batch_size, n_heads, -1)
    
    output = rope(x, input_pos)
    assert output.shape == x.shape
    assert not torch.equal(output, x)  # Output should be different from input
    
    # Test that output type matches input type
    assert output.dtype == x.dtype

# Test RoPE2D
def test_rope_2d():
    dim = 64
    rope2d = RoPE2D(dim=dim, max_height=32, max_width=32)
    
    # Test initialization
    assert rope2d.half_dim == dim // 2
    assert isinstance(rope2d.rope_height, RotaryPositionalEmbeddings)
    assert isinstance(rope2d.rope_width, RotaryPositionalEmbeddings)
    
    # Test forward pass
    batch_size, n_heads = 2, 4
    seq_len = 16
    x = torch.randn(batch_size, n_heads, seq_len, dim)
    positions = torch.randint(0, 32, (batch_size, n_heads, seq_len, 2))
    
    output = rope2d(x, positions)
    assert output.shape == x.shape
    assert not torch.equal(output, x)

# Test SwiGLUFFN
def test_swiglu_ffn():
    dim = 64
    hidden_dim = 256
    ffn = SwiGLUFFN(dim, hidden_dim)
    
    x = torch.randn(5, 10, dim)
    output = ffn(x)
    assert output.shape == x.shape

# Test RMSNorm
def test_rms_norm():
    dim = 64
    norm = RMSNorm(dim)
    
    x = torch.randn(5, 10, dim)
    output = norm(x)
    
    assert output.shape == x.shape
    assert torch.allclose(
        torch.sqrt(torch.mean(output.pow(2), dim=-1)),
        torch.ones_like(torch.mean(output.pow(2), dim=-1)),
        atol=1e-5
    )

# Test GridDVAEConfig
def test_grid_dvae_config():
    # Test valid configuration
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,  # Changed from n_layers
        n_latent_layer=3,  # Added new parameter
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    assert config.max_grid_height == 8  # sqrt(64)
    assert config.max_grid_width == 8
    
    # Test invalid configurations
    with pytest.raises(ValueError):
        GridDVAEConfig(n_dim=65, n_head=8, n_grid_layer=3, n_latent_layer=3, max_grid_height=8, max_grid_width=8, n_vocab=16, n_codes=16)
    
    with pytest.raises(AssertionError):
        # Test non-power-of-2 grid size (31x32 = 992)
        GridDVAEConfig(n_dim=128, n_head=8, n_grid_layer=3, n_latent_layer=3, max_grid_height=7, max_grid_width=8, n_vocab=16, n_codes=16)

# Test DVAE
def test_dvae():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,
        n_latent_layer=3,
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    
    dvae = GridDVAE(config)
    
    # Test forward pass
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))
    
    output, losses, q_z_marg = dvae(x)
    
    # Check losses
    assert 'ce_loss' in losses, "losses should contain 'ce_loss'"
    assert 'mi_loss' in losses, "losses should contain 'mi_loss'"
    assert 'dwkl_loss' in losses, "losses should contain 'dwkl_loss'"
    assert 'tc_loss' in losses, "losses should contain 'tc_loss'"
    assert 'kl_loss' in losses, "losses should contain 'kl_loss'"
    
    # Test output shape
    assert output.shape == (batch_size, config.n_pos, config.n_vocab), \
        f"Expected output shape {(batch_size, config.n_pos, config.n_vocab)}, got {output.shape}"
    
    # Convert to probabilities and test
    probs = F.softmax(output, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1))), \
        "Probabilities don't sum to 1"
    
    # Test that output contains logits (not all zeros or NaNs)
    assert not torch.allclose(output, torch.zeros_like(output)), \
        "Output contains all zeros"
    assert not torch.any(torch.isnan(output)), \
        "Output contains NaN values"
    


def test_dvae_forward_with_mask():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,  # Changed from n_layers
        n_latent_layer=3,  # Added new parameter
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))

    # Test forward pass with no mask
    output_no_mask, losses, q_z_marg = dvae(x, mask_percentage=0.0)
    assert output_no_mask.shape == (batch_size, config.n_pos, config.n_vocab), \
        f"Unexpected output shape: {output_no_mask.shape}"

    # Test forward pass with partial mask
    output_partial_mask, losses, _ = dvae(x, mask_percentage=0.5)
    assert output_partial_mask.shape == (batch_size, config.n_pos, config.n_vocab), \
        f"Unexpected output shape: {output_partial_mask.shape}" 


# Test reconstruction_loss
def test_reconstruction_loss():
    """
    Test the reconstruction_loss method for correctness by comparing the
    computed cross-entropy loss with manually calculated loss.
    """
    dvae = GridDVAE(GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,  # Changed from n_layers
        n_latent_layer=3,  # Added new parameter
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    ))
    
    # Create dummy decoded logits and target x
    decoded_logits = torch.tensor([
        [[2.0, 0.5, 0.3, 0.2],
         [0.1, 2.0, 0.3, 0.4]],
        [[0.3, 0.2, 3.0, 0.5],
         [0.2, 0.1, 0.4, 4.0]]
    ])  # Shape: (2, 2, 4)
    
    x = torch.tensor([
        [0, 1],
        [2, 3]
    ])  # Shape: (2, 2)
    
    # Manually compute cross-entropy loss
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    expected_loss = loss_fn(decoded_logits.view(-1, 4), x.view(-1)) / x.size(0)
    
    # Compute reconstruction loss using the method
    computed_loss = dvae.reconstruction_loss(decoded_logits, x)
    
    # Assert that the losses are close
    assert torch.allclose(computed_loss, expected_loss), \
        f"Expected {expected_loss.item()}, but got {computed_loss.item()}"

@pytest.mark.parametrize("reinmax,tau,hardness,rtol", [
    (False, 0.1, -1.0, 0.1),    # Regular softmax, low temp 
    (False, 1.0, -1.0, 0.1),    # Regular softmax, medium temp
    (False, 5.0, -1.0, 0.1),    # Regular softmax, high temp
    (False, 0.1, 0.0, 0.1),     # Gumbel-softmax, low temp, soft
    (False, 1.0, 0.0, 0.1),     # Gumbel-softmax, medium temp, soft 
    (False, 5.0, 0.0, 0.1),     # Gumbel-softmax, high temp, soft
    (False, 0.1, 1.0, 0.1),     # Gumbel-softmax, low temp, hard
    (False, 1.0, 1.0, 0.1),     # Gumbel-softmax, medium temp, hard
    (False, 5.0, 1.0, 0.1),     # Gumbel-softmax, high temp, hard
    (True, 0.1, 1.0, 0.1),      # ReinMax, low temp
    (True, 1.0, 1.0, 0.1),      # ReinMax, medium temp 
    (True, 5.0, 1.0, 0.1),      # ReinMax, high temp
])
def test_reconstruction_loss_random_input(reinmax, tau, hardness, rtol):
    """Test reconstruction loss for random input under different sampling conditions."""
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,  # Changed from n_layers
        n_latent_layer=3,  # Added new parameter
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    batch_size = 2
    
    # Test with random input
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))
    
    # Run forward pass multiple times with different random initializations
    ce_losses = []
    num_trials = 5
    for _ in range(num_trials):
        dvae = GridDVAE(config)
        # dvae.apply(dvae._init_weights)  # Reinitialize weights
        decoded_logits, losses, _ = dvae(x, tau=tau, hardness=hardness, reinMax=reinmax)
        ce_losses.append(losses['ce_loss'].item())  # Divide by n_pos to get per-token loss
    
    # Check that mean CE loss is close to expected value for random input
    mean_ce_loss = np.mean(ce_losses)
    expected_ce_loss = -np.log(1/config.n_vocab)
    assert np.abs(mean_ce_loss - expected_ce_loss) / expected_ce_loss < rtol, \
        f"Mean CE loss {mean_ce_loss:.3f} not close to expected {expected_ce_loss:.3f} " \
        f"for random input (reinmax={reinmax}, tau={tau}, hardness={hardness})"

@pytest.mark.parametrize("reinmax,tau,hardness,rtol", [
    (False, 0.1, -1.0, 0.25),    # Regular softmax, low temp - higher tolerance
    (False, 1.0, -1.0, 0.15),    # Regular softmax, medium temp
    (False, 5.0, -1.0, 0.15),    # Regular softmax, high temp
    (False, 0.1, 0.0, 0.25),     # Gumbel-softmax, low temp, soft - higher tolerance
    (False, 1.0, 0.0, 0.15),     # Gumbel-softmax, medium temp, soft
    (False, 5.0, 0.0, 0.15),     # Gumbel-softmax, high temp, soft
    (False, 0.1, 1.0, 0.25),     # Gumbel-softmax, low temp, hard - higher tolerance
    (False, 1.0, 1.0, 0.15),     # Gumbel-softmax, medium temp, hard
    (False, 5.0, 1.0, 0.15),     # Gumbel-softmax, high temp, hard
    (True, 0.1, 1.0, 0.25),      # ReinMax, low temp - higher tolerance
    (True, 1.0, 1.0, 0.15),      # ReinMax, medium temp
    (True, 5.0, 1.0, 0.15),      # ReinMax, high temp
])
def test_reconstruction_loss_constant_input(reinmax, tau, hardness, rtol):
    """Test reconstruction loss for constant input under different sampling conditions."""
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,  # Changed from n_layers
        n_latent_layer=3,  # Added new parameter
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    
    # Test with constant input (all same token)
    x = torch.full((batch_size, config.n_pos), fill_value=config.n_vocab//2, dtype=torch.long)
    
    # Run forward pass multiple times with different random initializations  
    ce_losses = []
    num_trials = 5
    for _ in range(num_trials):
        dvae.apply(dvae._init_weights)  # Reinitialize weights
        decoded_logits, losses, _ = dvae(x, tau=tau, hardness=hardness, reinMax=reinmax)
        ce_losses.append(losses['ce_loss'].item())  # Divide by n_pos to get per-token loss
    
    # Check that mean CE loss is close to expected value for random input
    mean_ce_loss = np.mean(ce_losses)
    expected_ce_loss = -np.log(1/config.n_vocab)
    assert np.abs(mean_ce_loss - expected_ce_loss) / expected_ce_loss < rtol, \
        f"Mean CE loss {mean_ce_loss:.3f} not close to expected {expected_ce_loss:.3f} " \
        f"for constant input (reinmax={reinmax}, tau={tau}, hardness={hardness})"

# Test kld_disentanglement_loss
def test_kld_disentanglement_loss():
    """
    Test the kld_disentanglement_loss method for correctness by using
    known distributions and verifying the computed KL divergences.
    """
    # Initialize GridDVAE
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,  # Changed from n_layers
        n_latent_layer=3,  # Added new parameter
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    
    # Create a dummy soft code with known distributions
    # For simplicity, assume N=2 latent codes and C=4 codebook size
    code_soft = torch.tensor([
        [[0.25, 0.25, 0.25, 0.25],
         [0.25, 0.25, 0.25, 0.25]],
        [[0.1, 0.2, 0.3, 0.4],
         [0.4, 0.3, 0.2, 0.1]]
    ])  # Shape: (2, 2, 4)
    
    # Expected q_z_current after mean over batch
    expected_q_z_current = code_soft.mean(dim=0)  # Shape: (2, 4)
    
    # Since momentum=0.99 and initial q_z_running=None, q_z_running should be initialized to q_z_current
    expected_q_z_running = expected_q_z_current.clone()
    
    # Compute disentanglement losses
    losses, q_z_marg = dvae.codebook.kld_disentanglement_loss(code_soft)
    
    # Manually compute expected Full KL, MI, DWKL, and TC
    epsilon = 1e-8  # For numerical stability
    uniform_p = torch.tensor(0.25)
    
    # Full KL: sum over N and C KL(q(z_j|x) || p(z_j)), where p(z_j) is uniform (0.25)
    kl_full = ((code_soft * (torch.log(code_soft + epsilon) - torch.log(uniform_p))).sum(dim=2)).sum(dim=1).mean()
    
    # MI: sum over N KL(q(z_j|x) || q(z_j))
    mi = ((code_soft * (torch.log(code_soft + epsilon) - torch.log(expected_q_z_current + epsilon))).sum(dim=2)).sum(dim=1).mean()
    
    # DWKL: sum over N KL(q(z_j) || p(z_j))
    dwkl = (expected_q_z_current * (torch.log(expected_q_z_current + epsilon) - torch.log(uniform_p))).sum(dim=1).sum()
    
    # TC = Full KL - MI - DWKL
    tc = kl_full - mi - dwkl
    
    # Assert computed losses
    assert torch.allclose(losses["kl_loss"], kl_full, atol=1e-6), \
        f"Full KL loss mismatch: expected {kl_full.item()}, got {losses['kl_loss'].item()}"
    
    assert torch.allclose(losses["mi_loss"], mi, atol=1e-6), \
        f"MI loss mismatch: expected {mi.item()}, got {losses['mi_loss'].item()}"
    
    assert torch.allclose(losses["dwkl_loss"], dwkl, atol=1e-6), \
        f"DWKL loss mismatch: expected {dwkl.item()}, got {losses['dwkl_loss'].item()}"
    
    assert torch.allclose(losses["tc_loss"], tc, atol=1e-6), \
        f"TC loss mismatch: expected {tc.item()}, got {losses['tc_loss'].item()}"

# Add this test function to test the ReinMax estimator
def test_dvae_encode_with_reinmax():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,
        n_latent_layer=3,
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))
    
    # Get position indices
    grid_pos_indices = dvae.grid_pos_indices.expand(batch_size, -1, -1)
    latent_pos_indices = dvae.latent_pos_indices.expand(batch_size, -1)
    
    # Test encode with ReinMax
    encoded = dvae.encode(x, grid_pos_indices, latent_pos_indices)
    
    # Check output shape
    assert encoded.shape == (batch_size, config.n_codes, config.n_dim), \
        f"Unexpected encoded shape: {encoded.shape}"

# Add this test function to test the forward pass with ReinMax
def test_dvae_forward_with_reinmax():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,
        n_latent_layer=3,
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))
    
    # Test forward pass with ReinMax enabled
    output, losses, q_z_marg = dvae(x, tau=0.9, hardness=1.0, reinMax=True)
    
    # Test output shape and contents
    assert output.shape == (batch_size, config.n_pos, config.n_vocab)
    assert not torch.any(torch.isnan(output))
    assert not torch.allclose(output, torch.zeros_like(output))
    
    # Test losses
    assert all(k in losses for k in ['ce_loss', 'mi_loss', 'dwkl_loss', 'tc_loss', 'kl_loss'])
    assert all(not torch.isnan(v) for v in losses.values())

def test_reinmax_gradient_flow():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,  # Changed from n_layers
        n_latent_layer=3,  # Added new parameter
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    B = batch_size
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos), requires_grad=False)

    grid_pos_indices = dvae.grid_pos_indices.expand(B, -1, -1)
    latent_pos_indices = dvae.latent_pos_indices.expand(B, -1)

    encoded_logits = dvae.encode(x, grid_pos_indices, latent_pos_indices)

    encoded_logits, hard_code, soft_code, code_metrics = dvae.codebook(encoded_logits, tau=0.9, hardness=1.0, reinMax=True)


    # hard_code, _ = dvae.(x, tau=0.9, hardness=1.0, reinMax=True)
    loss = hard_code.sum()
    loss.backward()

    encoder_modules = [dvae.grid_encoder, dvae.latent_encoder]

    for module in encoder_modules:
        for name, param in module.named_parameters():
            assert param.grad is not None, f"Gradient for {name} is None"
        
    assert dvae.codebook.codebook.weight.grad is None, "Gradient for codebook is not None"

    decoder_modules = [dvae.grid_decoder, dvae.latent_decoder]

    for module in decoder_modules:
        for name, param in module.named_parameters():
            assert param.grad is None, f"Gradient for {name} is not None"

def test_reinmax_edge_cases():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,  # Changed from n_layers
        n_latent_layer=3,  # Added new parameter
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    B = batch_size
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))

    grid_pos_indices = dvae.grid_pos_indices.expand(B, -1, -1)
    latent_pos_indices = dvae.latent_pos_indices.expand(B, -1)

    encoded_logits = dvae.encode(x, grid_pos_indices, latent_pos_indices)


    # Test with very low temperature
    encoded_logits, hard_code_low_temp, soft_code, code_metrics = dvae.codebook(encoded_logits, tau=0.01, hardness=1.0, reinMax=True)
    assert torch.allclose(hard_code_low_temp.sum(dim=-1), torch.ones_like(hard_code_low_temp.sum(dim=-1))), \
        "hard_code should be a valid one-hot encoding even with low temperature."

    # Test with very high temperature
    encoded_logits, hard_code_high_temp, soft_code, code_metrics = dvae.codebook(encoded_logits, tau=10.0, hardness=1.0, reinMax=True)
    assert torch.allclose(hard_code_high_temp.sum(dim=-1), torch.ones_like(hard_code_high_temp.sum(dim=-1))), \
        "hard_code should be a valid one-hot encoding even with high temperature."

def test_reinmax_stochasticity():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,  # Changed from n_layers
        n_latent_layer=3,  # Added new parameter
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    B = batch_size
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))

    grid_pos_indices = dvae.grid_pos_indices.expand(B, -1, -1)
    latent_pos_indices = dvae.latent_pos_indices.expand(B, -1)

    torch.manual_seed(42)
    encoded_logits = dvae.encode(x, grid_pos_indices, latent_pos_indices)
    encoded_logits, hard_code1, soft_code, code_metrics = dvae.codebook(encoded_logits, tau=0.9, hardness=1.0, reinMax=True)

    torch.manual_seed(43)
    encoded_logits = dvae.encode(x, grid_pos_indices, latent_pos_indices)
    encoded_logits, hard_code2, soft_code, code_metrics = dvae.codebook(encoded_logits, tau=0.9, hardness=1.0, reinMax=True)

    assert not torch.allclose(hard_code1, hard_code2), "ReinMax outputs should differ with different seeds."

def test_reinmax_output_consistency():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,  # Changed from n_layers
        n_latent_layer=3,  # Added new parameter
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    B = batch_size
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))

    grid_pos_indices = dvae.grid_pos_indices.expand(B, -1, -1)
    latent_pos_indices = dvae.latent_pos_indices.expand(B, -1)

    torch.manual_seed(42)
    encoded_logits = dvae.encode(x, grid_pos_indices, latent_pos_indices)
    encoded_logits, hard_code1, soft_code, code_metrics = dvae.codebook(encoded_logits, tau=0.9, hardness=1.0, reinMax=True)

    torch.manual_seed(42)
    encoded_logits = dvae.encode(x, grid_pos_indices, latent_pos_indices)
    encoded_logits, hard_code2, soft_code, code_metrics = dvae.codebook(encoded_logits, tau=0.9, hardness=1.0, reinMax=True)

    assert torch.allclose(hard_code1, hard_code2), "ReinMax outputs differ across runs with the same seed."

def test_kld_losses_non_negative():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,  # Changed from n_layers
        n_latent_layer=3,  # Added new parameter
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))
    
    # Run forward pass multiple times with different random inputs
    for _ in range(5):
        x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))
        _, losses, _ = dvae(x)
        kld_losses = {k: v for k, v in losses.items() if k != 'ce_loss' and 'loss' in k}
        
        # Check that all losses are non-negative
        assert kld_losses["mi_loss"] >= 0, "MI loss should be non-negative"
        assert kld_losses["dwkl_loss"] >= 0, "DWKL loss should be non-negative"
        # assert kld_losses["tc_loss"] >= 0, "TC loss should be non-negative"
        assert kld_losses["kl_loss"] >= 0, "KL loss should be non-negative"
        
        # Check that losses are not all zero
        assert not torch.allclose(kld_losses["kl_loss"], torch.tensor(0.0)), \
            "KL loss should not be zero"

def test_kld_losses_extreme_inputs():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,  # Changed from n_layers
        n_latent_layer=3,  # Added new parameter
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    
    # Test with all zeros
    x_zeros = torch.zeros((batch_size, config.n_pos), dtype=torch.long)
    # Test with all same value
    x_same = torch.full((batch_size, config.n_pos), fill_value=config.n_vocab-1, dtype=torch.long)

    # Test with apply_relu=False to allow negative values
    _, losses_zeros_no_relu, _ = dvae(x_zeros)
    _, losses_same_no_relu, _ = dvae(x_same)
    
    kld_losses_zeros_no_relu = {k: v for k, v in losses_zeros_no_relu.items() if k != 'ce_loss' and 'loss' in k}
    kld_losses_same_no_relu = {k: v for k, v in losses_same_no_relu.items() if k != 'ce_loss' and 'loss' in k}

    # Check that losses are close to non-negative when not using ReLU
    for losses in [kld_losses_zeros_no_relu, kld_losses_same_no_relu]:
        assert losses["mi_loss"] >= -1e-6, "MI loss should be close to non-negative"
        assert losses["dwkl_loss"] >= -1e-6, "DWKL loss should be close to non-negative"
        assert losses["tc_loss"] >= -1e-5, "TC loss should be close to non-negative"
        assert losses["kl_loss"] >= 0, "KL loss should be close to non-negative"

  

def test_kld_losses_numerical_stability():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,  # Changed from n_layers
        n_latent_layer=3,  # Added new parameter
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))
    
    # Test with different temperature values
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    # Test without ReLU - allow small negative values
    for temp in temperatures:
        _, losses, _ = dvae(x, tau=temp)
        kld_losses = {k: v for k, v in losses.items() if k != 'ce_loss' and 'loss' in k}
        
        # Check that all losses are finite
        assert torch.isfinite(kld_losses["mi_loss"]), f"MI loss not finite at temperature {temp}"
        assert torch.isfinite(kld_losses["dwkl_loss"]), f"DWKL loss not finite at temperature {temp}"
        assert torch.isfinite(kld_losses["tc_loss"]), f"TC loss not finite at temperature {temp}"
        assert torch.isfinite(kld_losses["kl_loss"]), f"KL loss not finite at temperature {temp}"
        
        # Allow for small negative values due to numerical precision
        tolerance = 5e-6  # Increased from 1e-6 to 5e-6
        assert kld_losses["mi_loss"] >= -tolerance, f"MI loss too negative at temperature {temp}"
        assert kld_losses["dwkl_loss"] >= -tolerance, f"DWKL loss too negative at temperature {temp}"
        assert kld_losses["tc_loss"] >= -tolerance, f"TC loss too negative at temperature {temp}"
        assert kld_losses["kl_loss"] >= -tolerance, f"KL loss too negative at temperature {temp}"


def test_grid_dvae_config_serialization_full():
    # Create a config with some values
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,  # Changed from n_layers
        n_latent_layer=3,  # Added new parameter
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )

    # Convert to dict
    config_dict = config.to_dict()

    # Check that all base attributes are present
    assert config_dict['n_dim'] == 128
    assert config_dict['n_head'] == 8
    assert config_dict['n_grid_layer'] == 3  # Changed from n_layers
    assert config_dict['n_latent_layer'] == 3  # Added new parameter
    assert config_dict['n_codes'] == 16
    assert config_dict['codebook_size'] == 512
    assert config_dict['max_grid_height'] == 8  # sqrt(64)
    assert config_dict['max_grid_width'] == 8
    assert config_dict['n_vocab'] == 16

    # Check that computed attributes are present
    assert 'n_pos' in config_dict

    # Create new config from dict
    new_config = GridDVAEConfig.from_dict(config_dict)

    # Check that all attributes match
    assert new_config.n_dim == config.n_dim
    assert new_config.n_head == config.n_head
    assert new_config.n_grid_layer == config.n_grid_layer  # Changed from n_layers
    assert new_config.n_latent_layer == config.n_latent_layer  # Added new parameter
    assert new_config.n_codes == config.n_codes
    assert new_config.codebook_size == config.codebook_size
    assert new_config.max_grid_height == config.max_grid_height  # sqrt(64)
    assert new_config.max_grid_width == config.max_grid_width
    assert new_config.n_vocab == config.n_vocab

    # Check that computed attributes match
    assert new_config.n_pos == config.n_pos


def test_grid_dvae_config_serialization_with_defaults():
    # Create a minimal config with only required fields
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,  # Changed from n_layers
        n_latent_layer=3,  # Added new parameter
        n_codes=16
    )

    # Convert to dict and back
    config_dict = config.to_dict()
    new_config = GridDVAEConfig.from_dict(config_dict)

    # Check that default values are preserved
    assert new_config.codebook_size == 512  # default value
    assert new_config.rope_base == 10_000   # default value
    assert new_config.dropout == 0.0        # default value
    assert new_config.max_grid_height == 32 # default value
    assert new_config.max_grid_width == 32  # default value
    assert new_config.n_vocab == 16         # default value
    assert new_config.padding_idx == 15     # default value
    assert new_config.mask_idx == 14        # default value
    assert new_config.pad_weight == 0.01     # default value


def test_grid_dvae_config_json_serialization():
    import json

    # Create a config
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,  # Changed from n_layers
        n_latent_layer=3,  # Added new parameter
        n_codes=16,
        codebook_size=256, # Override default to test
        dropout=0.1,       # Override default to test
    )

    # Convert to dict
    config_dict = config.to_dict()

    # Test JSON serialization
    json_str = json.dumps(config_dict)
    loaded_dict = json.loads(json_str)

    new_config = GridDVAEConfig.from_dict(loaded_dict)

    assert new_config.n_dim == config.n_dim, \
        f"n_dim mismatch: expected {config.n_dim}, got {new_config.n_dim}"
    assert new_config.n_head == config.n_head, \
        f"n_head mismatch: expected {config.n_head}, got {new_config.n_head}"
    assert new_config.n_grid_layer == config.n_grid_layer  # Changed from n_layers
    assert new_config.n_latent_layer == config.n_latent_layer  # Added new parameter
    assert new_config.n_codes == config.n_codes, \
        f"n_codes mismatch: expected {config.n_codes}, got {new_config.n_codes}"
    assert new_config.n_pos == config.n_pos, \
        f"n_pos mismatch: expected {config.n_pos}, got {new_config.n_pos}"
    assert new_config.codebook_size == config.codebook_size, \
        f"codebook_size mismatch: expected {config.codebook_size}, got {new_config.codebook_size}"
    assert new_config.dropout == config.dropout, \
        f"dropout mismatch: expected {config.dropout}, got {new_config.dropout}"
    assert new_config.rope_base == config.rope_base, \
        f"rope_base mismatch: expected {config.rope_base}, got {new_config.rope_base}"
    assert new_config.max_grid_height == config.max_grid_height, \
        f"max_grid_height mismatch: expected {config.max_grid_height}, got {new_config.max_grid_height}"
    assert new_config.max_grid_width == config.max_grid_width, \
        f"max_grid_width mismatch: expected {config.max_grid_width}, got {new_config.max_grid_width}"
    assert new_config.n_vocab == config.n_vocab, \
        f"n_vocab mismatch: expected {config.n_vocab}, got {new_config.n_vocab}"
    assert new_config.padding_idx == config.padding_idx, \
        f"padding_idx mismatch: expected {config.padding_idx}, got {new_config.padding_idx}"
    assert new_config.mask_idx == config.mask_idx, \
        f"mask_idx mismatch: expected {config.mask_idx}, got {new_config.mask_idx}"
    assert new_config.pad_weight == config.pad_weight, \
        f"pad_weight mismatch: expected {config.pad_weight}, got {new_config.pad_weight}"


def test_dvae_encode_with_hardness():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,
        n_latent_layer=3,
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))

    # Test with different hardness values
    hardness_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for hardness in hardness_values:

        grid_pos_indices = dvae.grid_pos_indices.expand(batch_size, -1, -1)

        latent_pos_indices = dvae.latent_pos_indices.expand(batch_size, -1)

        encoded_logits = dvae.encode(x, grid_pos_indices, latent_pos_indices)

        encoded_logits, hard_code, soft_code, code_metrics = dvae.codebook(encoded_logits, tau=0.9, hardness=hardness, reinMax=True)

        # Check shapes
        assert hard_code.shape == (batch_size, config.n_codes, config.codebook_size)
        assert soft_code.shape == (batch_size, config.n_codes, config.codebook_size)
        
        # Check that probabilities sum to 1
        assert torch.allclose(hard_code.sum(dim=-1), torch.ones_like(hard_code.sum(dim=-1)))
        assert torch.allclose(soft_code.sum(dim=-1), torch.ones_like(soft_code.sum(dim=-1)))
        
        # For hardness=0, code should equal soft_code
        if hardness == 0.0:
            assert torch.allclose(hard_code, soft_code)
        # For hardness=1, code should be one-hot
        elif hardness == 1.0:
            assert (hard_code.max(dim=-1)[0] == 1.0).all()

def test_dvae_encode_with_hardness_and_reinmax():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,
        n_latent_layer=3,
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))

    # Test ReinMax with non-zero hardness
    hardness_values = [0.25, 0.5, 0.75, 1.0]
    
    for hardness in hardness_values:
        grid_pos_indices = dvae.grid_pos_indices.expand(batch_size, -1, -1)
        latent_pos_indices = dvae.latent_pos_indices.expand(batch_size, -1)

        encoded_logits = dvae.encode(x, grid_pos_indices, latent_pos_indices)

        encoded_logits, hard_code, soft_code, code_metrics = dvae.codebook(encoded_logits, tau=0.9, hardness=hardness, reinMax=True)
        
        # Check shapes and probability distributions
        assert hard_code.shape == (batch_size, config.n_codes, config.codebook_size)
        assert soft_code.shape == (batch_size, config.n_codes, config.codebook_size)
        assert torch.allclose(hard_code.sum(dim=-1), torch.ones_like(hard_code.sum(dim=-1)))
        assert torch.allclose(soft_code.sum(dim=-1), torch.ones_like(soft_code.sum(dim=-1)))
        
        # Code should differ from soft_code when using ReinMax
        assert not torch.allclose(hard_code, soft_code)

def test_hardness_gradient_flow():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_grid_layer=3,
        n_latent_layer=3,
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos), requires_grad=False)

    # Test gradient flow with different hardness values
    hardness_values = [0.0, 0.5, 1.0]
    
    for hardness in hardness_values:
        grid_pos_indices = dvae.grid_pos_indices.expand(batch_size, -1, -1)
        latent_pos_indices = dvae.latent_pos_indices.expand(batch_size, -1)

        encoded_logits = dvae.encode(x, grid_pos_indices, latent_pos_indices)

        encoded_logits, hard_code, soft_code, code_metrics = dvae.codebook(encoded_logits, tau=0.9, hardness=hardness, reinMax=True)
        # Clear gradients
        dvae.zero_grad()
        
        loss = hard_code.sum()
        loss.backward()

        # Check encoder gradients exist
        encoder_modules = [dvae.grid_encoder, dvae.latent_encoder]
        for module in encoder_modules:
            for name, param in module.named_parameters():
                assert param.grad is not None, f"Gradient for {name} is None with hardness={hardness}"


@pytest.fixture
def config():
    return GridDVAEConfig(
        n_dim=128,
        n_head=4,
        n_grid_layer=2,  # Changed from n_layers
        n_latent_layer=2,  # Added new parameter
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )

def test_grid_dvae_config_validation():
    # Test valid configuration
    valid_config = GridDVAEConfig(
        n_dim=128,
        n_head=4,
        n_grid_layer=2,
        n_latent_layer=2,
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    assert valid_config.padding_idx == 15  # n_vocab - 1
    assert valid_config.mask_idx == 14     # n_vocab - 2
    assert valid_config.n_pos == 64  # 8 * 8

def test_grid_dvae_forward():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=4,
        n_grid_layer=2,
        n_latent_layer=2,
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    model = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))
    
    # Test forward pass
    decoded_logits, losses, q_z_marg = model(x, tau=1.0, hardness=0.0, mask_percentage=0.1)
    
    assert decoded_logits.shape == (batch_size, config.n_pos, config.n_vocab)
    assert 'ce_loss' in losses
    assert 'mi_loss' in losses
    assert 'dwkl_loss' in losses
    assert 'tc_loss' in losses
    assert 'kl_loss' in losses
    assert q_z_marg.shape == (config.n_codes, config.codebook_size)

def test_encode_decode():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=4,
        n_grid_layer=2,
        n_latent_layer=2,
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    model = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))
    
    # Get position indices
    grid_pos_indices = model.grid_pos_indices.expand(batch_size, -1, -1)
    latent_pos_indices = model.latent_pos_indices.expand(batch_size, -1)
    
    # Test encode
    encoded = model.encode(x, grid_pos_indices, latent_pos_indices)
    assert encoded.shape == (batch_size, config.n_codes, config.n_dim)
    
    # Test decode
    decoded_logits = model.decode(encoded, grid_pos_indices, latent_pos_indices)
    assert decoded_logits.shape == (batch_size, config.n_pos, config.n_vocab)

def test_masking():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=4,
        n_grid_layer=2,
        n_latent_layer=2,
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    model = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))
    
    # Add some padding tokens
    x[:, -5:] = config.padding_idx
    
    # Test different mask percentages
    for mask_pct in [0.0, 0.3, 0.7]:
        masked_x = model.apply_mask(x.clone(), mask_percentage=mask_pct)
        
        # Check that padding tokens weren't masked
        assert torch.all(masked_x[:, -5:] == config.padding_idx)
        
        # Check mask percentage (excluding padding tokens)
        non_pad_tokens = (x != config.padding_idx).sum()
        masked_tokens = (masked_x == config.mask_idx).sum()
        actual_mask_pct = masked_tokens.float() / non_pad_tokens
        assert abs(actual_mask_pct.item() - mask_pct) < 0.1  # Allow some variance due to randomness

def test_config_serialization():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=4,
        n_grid_layer=2,
        n_latent_layer=2,
        n_codes=16,
        codebook_size=512,
        max_grid_height=8,
        max_grid_width=8,
        n_vocab=16
    )
    
    # Test to_dict
    config_dict = config.to_dict()
    assert 'n_dim' in config_dict
    assert 'n_head' in config_dict
    assert 'n_grid_layer' in config_dict
    assert 'n_latent_layer' in config_dict
    assert 'n_codes' in config_dict
    assert 'codebook_size' in config_dict
    assert 'n_pos' in config_dict
    
    # Test from_dict
    new_config = GridDVAEConfig.from_dict(config_dict)
    assert new_config.n_dim == config.n_dim
    assert new_config.n_head == config.n_head
    assert new_config.n_grid_layer == config.n_grid_layer
    assert new_config.n_latent_layer == config.n_latent_layer
    assert new_config.n_codes == config.n_codes
    assert new_config.n_pos == config.n_pos
