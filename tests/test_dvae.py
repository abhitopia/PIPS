import pytest
import torch
import numpy as np
from pips.dvae import (
    is_power_of_two,
    is_perfect_square,
    gumbel_softmax,
    RotaryPositionalEmbeddings,
    RoPE2D,
    SwiGLU,
    SwiGLUFFN,
    RMSNorm,
    GridDVAEConfig,
    GridDVAE,
    # AttentionPool,
    # StackedPooling,
    StackedTransformerProjection,
    Transformer,
    ResidualProjection,
    TransformerProjection
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

def test_create_grid_position_tensor():
    # Test 2x2 grid
    pos_2x2 = GridDVAE.create_grid_position_tensor(2, 2, requires_grad=False)
    expected_2x2 = torch.tensor([
        [0, 0],  # top-left
        [0, 1],  # top-right
        [1, 0],  # bottom-left
        [1, 1]   # bottom-right
    ], dtype=torch.long)
    assert torch.equal(pos_2x2, expected_2x2)
    
    # Test 3x2 grid
    pos_3x2 = GridDVAE.create_grid_position_tensor(3, 2, requires_grad=False)
    expected_3x2 = torch.tensor([
        [0, 0], [0, 1],  # first row
        [1, 0], [1, 1],  # second row
        [2, 0], [2, 1]   # third row
    ], dtype=torch.long)
    assert torch.equal(pos_3x2, expected_3x2)
    
    # Test requires_grad parameter
    pos_no_grad = GridDVAE.create_grid_position_tensor(2, 2, requires_grad=False)
    assert not pos_no_grad.requires_grad
    assert pos_no_grad.dtype == torch.long
    
    pos_with_grad = GridDVAE.create_grid_position_tensor(2, 2, requires_grad=True)
    assert pos_with_grad.requires_grad
    assert pos_with_grad.dtype == torch.float  # Must be float to support gradients

def test_gumbel_softmax():
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    # Test temperature scaling
    out_high_temp = gumbel_softmax(logits, tau=10.0, hard=False)
    out_low_temp = gumbel_softmax(logits, tau=0.1, hard=False)
    
    # Higher temperature should lead to more uniform distribution
    assert torch.std(out_high_temp) < torch.std(out_low_temp)
    
    # Test hard=True
    out_hard = gumbel_softmax(logits, tau=1.0, hard=True)
    
    # Check that output sums to 1 along dim=-1
    assert torch.allclose(out_hard.sum(dim=-1), torch.ones_like(out_hard.sum(dim=-1)))
    
    # Check that hard=True produces one-hot vectors
    assert torch.all(torch.logical_or(torch.isclose(out_hard, torch.tensor(0.0)), 
                                    torch.isclose(out_hard, torch.tensor(1.0))))
    
    # Test invalid temperature
    with pytest.raises(ValueError):
        gumbel_softmax(logits, tau=0.0)

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

# Test SwiGLU
def test_swiglu():
    swiglu = SwiGLU()
    x = torch.randn(5, 10)
    output = swiglu(torch.cat([x, x], dim=-1))
    assert output.shape == x.shape

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
        n_layers=6,
        n_codes=8,  # Directly specify the number of codes
        codebook_size=512,
        max_grid_height=32,  # Replace n_pos with grid dimensions
        max_grid_width=32,
        n_vocab=16
    )
    assert config.compression_factor == 128  # 1024/8
    assert config.max_grid_height == 32  # sqrt(1024)
    assert config.max_grid_width == 32
    
    # Test invalid configurations
    with pytest.raises(ValueError):
        GridDVAEConfig(n_dim=65, n_head=8, n_layers=6, max_grid_height=32, max_grid_width=32, n_vocab=16, n_codes=8)
    
    with pytest.raises(AssertionError):
        # Test non-power-of-2 grid size (31x32 = 992)
        GridDVAEConfig(n_dim=128, n_head=8, n_layers=6, max_grid_height=31, max_grid_width=32, n_vocab=16, n_codes=8)

# Test DVAE
def test_dvae():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    
    dvae = GridDVAE(config)
    
    # Verify bottleneck widths are correct
    expected_widths = [1024, 512, 256, 128, 64, 32, 16, 8]  # n_pos -> n_codes
    assert config.bottleneck_widths == expected_widths, \
        f"Bottleneck widths incorrect.\nExpected: {expected_widths}\nGot: {config.bottleneck_widths}\nDifference in lengths: {len(expected_widths)} vs {len(config.bottleneck_widths)}"
    

    # Verify encoder bottleneck input sequence length
    assert dvae.encoder_bottleneck.input_seq_len == config.n_pos, \
        "Encoder bottleneck input sequence length doesn't match n_pos"
    
    # Verify encoder bottleneck sequence lengths
    assert dvae.encoder_bottleneck.output_seq_lens == expected_widths[1:], \
        "Encoder bottleneck sequence lengths don't match expected widths"
    

    # Verify decoder bottleneck input sequence length
    assert dvae.decoder_bottleneck.input_seq_len == config.n_codes, \
        "Decoder bottleneck input sequence length doesn't match n_codes"
    
    # Verify decoder bottleneck sequence lengths
    # For decoder, we start from n_codes (8) and go up to n_pos (1024)
    assert dvae.decoder_bottleneck.output_seq_lens == expected_widths[::-1][1:], \
        "Decoder bottleneck sequence lengths don't match expected widths"
    
    # Test forward pass
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))
    
    output, losses, q_z_marg = dvae(x)
    # Access reconstruction loss and kld losses from the losses dictionary
    reconstruction_loss = losses['ce_loss']
    kld_losses = {k: v for k, v in losses.items() if k != 'ce_loss' and 'loss' in k}
    
    # Test output shape
    assert output.shape == (batch_size, config.n_pos, config.n_vocab), \
        f"Expected output shape {(batch_size, config.n_pos, config.n_vocab)}, got {output.shape}"
    assert isinstance(reconstruction_loss, torch.Tensor), "reconstruction_loss should be a tensor."
    assert reconstruction_loss.dim() == 0, "reconstruction_loss should be a scalar tensor."
    assert isinstance(kld_losses, dict), "kld_losses should be a dictionary."
    assert 'mi_loss' in kld_losses, "kld_losses should contain 'mi_loss'."
    assert 'dwkl_loss' in kld_losses, "kld_losses should contain 'dwkl_loss'."
    assert 'tc_loss' in kld_losses, "kld_losses should contain 'tc_loss'."
    assert 'kl_loss' in kld_losses, "kld_losses should contain 'kl_loss'."
    
    # Convert to probabilities and test
    probs = F.softmax(output, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1))), \
        "Probabilities don't sum to 1"
    
    # Test that output contains logits (not all zeros or NaNs)
    assert not torch.allclose(output, torch.zeros_like(output)), \
        "Output contains all zeros"
    assert not torch.any(torch.isnan(output)), \
        "Output contains NaN values"


def test_create_random_mask_no_mask():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    seq_length = 10
    
    # Test with mask_percentage = 0
    mask = dvae.create_random_mask(batch_size, seq_length, mask_percentage=0.0)
    # Instead of checking for None, check that all values are True (no masking)
    assert torch.all(mask == True)

def test_create_random_mask_full_mask():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    seq_length = 10
    
    # Test with mask_percentage = 1
    with pytest.raises(AssertionError):
        dvae.create_random_mask(batch_size, seq_length, mask_percentage=1.0)

def test_create_random_mask_partial_mask():
    dvae = GridDVAE(GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    ))
    B, S = 4, 1024
    mask_percentage = 0.5
    mask = dvae.create_random_mask(B, S, mask_percentage=mask_percentage)
    assert mask is not None, "Expected a mask when mask_percentage is between 0 and 1."
    assert mask.shape == (B, 1, S), f"Unexpected mask shape: {mask.shape}"
    # Check that approximately half of the mask is False
    assert torch.isclose(mask.float().mean(), torch.tensor(1 - mask_percentage), atol=0.1), \
        "Mask does not have the expected proportion of masked tokens."

def test_dvae_forward_with_mask():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
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



@pytest.mark.parametrize("mask_percentage,same_mask_for_all", [
    (0.5, True),   # Test with 50% masking, same mask for all
    (0.5, False),  # Test with 50% masking, different masks
    (0.3, True),   # Test with 30% masking, same mask for all
    (0.7, False),  # Test with 70% masking, different masks
])
def test_dvae_masking_effect(mask_percentage, same_mask_for_all):
    """
    Test that masked positions do not affect the encoding at unmasked positions.
    The outputs should be identical at unmasked positions regardless of the input at masked positions.
    """
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    B = 2
    S = config.n_pos
    
    # Create two identical inputs
    torch.manual_seed(42)  # For reproducibility
    x1 = torch.randint(0, config.n_vocab, (B, S))
    x2 = x1.clone()

    # Create position indices
    grid_height = int(S**0.5)
    grid_width = grid_height
    positions = GridDVAE.create_grid_position_tensor(grid_height, grid_width, requires_grad=False)
    positions = positions.unsqueeze(0).expand(B, -1, -1)  # Expand to [B, S, 2]

    # Create mask
    mask = dvae.create_random_mask(B, S, mask_percentage, same_mask_for_all=same_mask_for_all)
    assert mask is not None, "Mask should not be None"
    
    # Handle both [B, 1, S] and [1, 1, S] mask shapes
    mask_expanded = mask.squeeze(1)  # Remove middle dimension to get [B/1, S]
    if same_mask_for_all:  # If using same mask for all, expand to match batch size
        mask_expanded = mask_expanded.expand(B, -1)
    
    # Modify x2 at masked positions
    x2[~mask_expanded] = torch.randint(0, config.n_vocab, (torch.sum(~mask_expanded).item(),))

    # Test each step of the encoding process
    with torch.no_grad():
        # Test embeddings
        emb1 = dvae.embd(x1)
        emb2 = dvae.embd(x2)
        assert torch.allclose(emb1[mask_expanded], emb2[mask_expanded], atol=1e-5), \
            "Embeddings differ at unmasked positions"

        # Test encoder base
        enc1, _ = dvae.encoder_base(emb1, mask, positions=positions)
        enc2, _ = dvae.encoder_base(emb2, mask, positions=positions)
        assert torch.allclose(enc1[mask_expanded], enc2[mask_expanded], atol=1e-5), \
            "Encoder base outputs differ at unmasked positions"

        # Test encoder bottleneck
        bottleneck1 = dvae.encoder_bottleneck(enc1, mask.squeeze(1))
        bottleneck2 = dvae.encoder_bottleneck(enc2, mask.squeeze(1))
        assert torch.allclose(bottleneck1, bottleneck2, atol=1e-5), \
            "Encoder bottleneck outputs differ"

        # Test encoder head
        head1 = dvae.encoder_head(bottleneck1)
        head2 = dvae.encoder_head(bottleneck2)
        assert torch.allclose(head1, head2, atol=1e-5), \
            "Encoder head outputs differ"

        # Test gumbel-softmax
        torch.manual_seed(42)
        gumbel1 = F.gumbel_softmax(head1, tau=0.9, hard=False)
        torch.manual_seed(42)
        gumbel2 = F.gumbel_softmax(head2, tau=0.9, hard=False)
        assert torch.allclose(gumbel1, gumbel2, atol=1e-6), \
            "Gumbel-softmax outputs differ"

        # Test final encoding
        torch.manual_seed(42)
        code1, _ = dvae.encode(x1, attn_mask=mask, tau=0.9, hardness=0.0)
        torch.manual_seed(42)
        code2, _ = dvae.encode(x2, attn_mask=mask, tau=0.9, hardness=0.0)
        assert torch.allclose(code1, code2, atol=1e-6), \
            "Final codes differ when only masked positions are changed"

# Test reconstruction_loss
def test_reconstruction_loss():
    """
    Test the reconstruction_loss method for correctness by comparing the
    computed cross-entropy loss with manually calculated loss.
    """
    dvae = GridDVAE(GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=4  # Reduced vocab size for simplicity
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
        n_layers=6,
        n_codes=4,  # Reduced for simplicity
        codebook_size=512,
        max_grid_height=4,    # 4x4 grid = 16 positions
        max_grid_width=4,
        n_vocab=4
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
    losses, q_z_marg = dvae.kld_disentanglement_loss(code_soft)
    
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
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))

    # Test encode with ReinMax enabled
    hard_code, soft_code = dvae.encode(x, tau=0.9, hardness=1.0, reinMax=True)

    # Check that the output shapes are correct
    assert hard_code.shape == (batch_size, config.n_codes, config.codebook_size), \
        f"Unexpected hard_code shape: {hard_code.shape}"
    assert soft_code.shape == (batch_size, config.n_codes, config.codebook_size), \
        f"Unexpected soft_code shape: {soft_code.shape}"

    # Check that hard_code is a valid one-hot encoding
    assert torch.allclose(hard_code.sum(dim=-1), torch.ones_like(hard_code.sum(dim=-1))), \
        "hard_code should be a valid one-hot encoding."

    # Check that ReinMax logic is applied (hard_code should differ from soft_code)
    assert not torch.allclose(hard_code, soft_code), \
        "hard_code should differ from soft_code when ReinMax is applied."

# Add this test function to test the forward pass with ReinMax
def test_dvae_forward_with_reinmax():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))

    # Test forward pass with ReinMax enabled
    output, losses, q_z_marg = dvae(x, tau=0.9, hardness=1.0, reinMax=True)
    reconstruction_loss = losses['ce_loss']
    kld_losses = {k: v for k, v in losses.items() if k != 'ce_loss' and 'loss' in k}

    # Test output shape
    assert output.shape == (batch_size, config.n_pos, config.n_vocab), \
        f"Unexpected output shape: {output.shape}"
    assert isinstance(reconstruction_loss, torch.Tensor), "reconstruction_loss should be a tensor."
    assert reconstruction_loss.dim() == 0, "reconstruction_loss should be a scalar tensor."
    assert isinstance(kld_losses, dict), "kld_losses should be a dictionary."
    assert 'mi_loss' in kld_losses, "kld_losses should contain 'mi_loss'."
    assert 'dwkl_loss' in kld_losses, "kld_losses should contain 'dwkl_loss'."
    assert 'tc_loss' in kld_losses, "kld_losses should contain 'tc_loss'."
    assert 'kl_loss' in kld_losses, "kld_losses should contain 'kl_loss'."

    # Convert to probabilities and test
    probs = F.softmax(output, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)))

    # Test that output contains logits (not all zeros or NaNs)
    assert not torch.allclose(output, torch.zeros_like(output)), \
        "Output should not be all zeros when ReinMax is applied."
    assert not torch.any(torch.isnan(output)), "Output should not contain NaNs when ReinMax is applied."

# Add this test to your existing test suite 

def test_reinmax_gradient_flow():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos), requires_grad=False)

    hard_code, _ = dvae.encode(x, tau=0.9, hardness=1.0, reinMax=True)
    loss = hard_code.sum()
    loss.backward()


    encoder_modules = [dvae.encoder_bottleneck, dvae.encoder_base, dvae.encoder_head, dvae.embd]

    for module in encoder_modules:
        for name, param in module.named_parameters():
            assert param.grad is not None, f"Gradient for {name} is None"
        
    assert dvae.codebook.grad is None, "Gradient for codebook is not None"

    decoder_modules = [dvae.decoder_bottleneck, dvae.decoder_base, dvae.decoder_head]

    for module in decoder_modules:
        for name, param in module.named_parameters():
            assert param.grad is None, f"Gradient for {name} is not None"

def test_reinmax_edge_cases():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))

    # Test with very low temperature
    hard_code_low_temp, _ = dvae.encode(x, tau=0.01, hardness=1.0, reinMax=True)
    assert torch.allclose(hard_code_low_temp.sum(dim=-1), torch.ones_like(hard_code_low_temp.sum(dim=-1))), \
        "hard_code should be a valid one-hot encoding even with low temperature."

    # Test with very high temperature
    hard_code_high_temp, _ = dvae.encode(x, tau=10.0, hardness=1.0, reinMax=True)
    assert torch.allclose(hard_code_high_temp.sum(dim=-1), torch.ones_like(hard_code_high_temp.sum(dim=-1))), \
        "hard_code should be a valid one-hot encoding even with high temperature."

def test_reinmax_stochasticity():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))

    torch.manual_seed(42)
    hard_code1, _ = dvae.encode(x, tau=0.9, hardness=1.0, reinMax=True)

    torch.manual_seed(43)
    hard_code2, _ = dvae.encode(x, tau=0.9, hardness=1.0, reinMax=True)

    assert not torch.allclose(hard_code1, hard_code2), "ReinMax outputs should differ with different seeds."

def test_reinmax_output_consistency():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))

    torch.manual_seed(42)
    hard_code1, _ = dvae.encode(x, tau=0.9, hardness=1.0, reinMax=True)

    torch.manual_seed(42)
    hard_code2, _ = dvae.encode(x, tau=0.9, hardness=1.0, reinMax=True)

    assert torch.allclose(hard_code1, hard_code2), "ReinMax outputs differ across runs with the same seed."

def test_kld_losses_non_negative():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))
    
    # Run forward pass multiple times with different random inputs
    for _ in range(5):
        x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))
        _, losses, _ = dvae(x, apply_relu=True)
        kld_losses = {k: v for k, v in losses.items() if k != 'ce_loss' and 'loss' in k}
        
        # Check that all losses are non-negative
        assert kld_losses["mi_loss"] >= 0, "MI loss should be non-negative"
        assert kld_losses["dwkl_loss"] >= 0, "DWKL loss should be non-negative"
        assert kld_losses["tc_loss"] >= 0, "TC loss should be non-negative"
        assert kld_losses["kl_loss"] >= 0, "KL loss should be non-negative"
        
        # Check that losses are not all zero
        assert not torch.allclose(kld_losses["kl_loss"], torch.tensor(0.0)), \
            "KL loss should not be zero"

def test_kld_losses_extreme_inputs():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    
    # Test with all zeros
    x_zeros = torch.zeros((batch_size, config.n_pos), dtype=torch.long)
    # Test with all same value
    x_same = torch.full((batch_size, config.n_pos), fill_value=config.n_vocab-1, dtype=torch.long)

    # Test with apply_relu=False to allow negative values
    _, losses_zeros_no_relu, _ = dvae(x_zeros, apply_relu=False)
    _, losses_same_no_relu, _ = dvae(x_same, apply_relu=False)
    
    kld_losses_zeros_no_relu = {k: v for k, v in losses_zeros_no_relu.items() if k != 'ce_loss' and 'loss' in k}
    kld_losses_same_no_relu = {k: v for k, v in losses_same_no_relu.items() if k != 'ce_loss' and 'loss' in k}

    # Check that losses are close to non-negative when not using ReLU
    for losses in [kld_losses_zeros_no_relu, kld_losses_same_no_relu]:
        assert losses["mi_loss"] >= -1e-6, "MI loss should be close to non-negative"
        assert losses["dwkl_loss"] >= -1e-6, "DWKL loss should be close to non-negative"
        assert losses["tc_loss"] >= -1e-5, "TC loss should be close to non-negative"
        assert losses["kl_loss"] >= 0, "KL loss should be close to non-negative"

    # Test with apply_relu=True to ensure non-negative values
    _, losses_zeros_relu, _ = dvae(x_zeros, apply_relu=True)
    _, losses_same_relu, _ = dvae(x_same, apply_relu=True)
    
    kld_losses_zeros_relu = {k: v for k, v in losses_zeros_relu.items() if k != 'ce_loss' and 'loss' in k}
    kld_losses_same_relu = {k: v for k, v in losses_same_relu.items() if k != 'ce_loss' and 'loss' in k}

    # Check that all losses are strictly non-negative when using ReLU
    for losses in [kld_losses_zeros_relu, kld_losses_same_relu]:
        assert losses["mi_loss"] >= 0, "MI loss should be non-negative with ReLU"
        assert losses["dwkl_loss"] >= 0, "DWKL loss should be non-negative with ReLU"
        assert losses["tc_loss"] >= 0, "TC loss should be non-negative with ReLU"
        assert losses["kl_loss"] >= 0, "KL loss should be non-negative with ReLU"

def test_kld_losses_numerical_stability():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))
    
    # Test with different temperature values
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    # Test without ReLU - allow small negative values
    for temp in temperatures:
        _, losses, _ = dvae(x, tau=temp, apply_relu=False)
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

    # Test with ReLU - ensure strictly non-negative
    for temp in temperatures:
        _, losses, _ = dvae(x, tau=temp, apply_relu=True)
        kld_losses = {k: v for k, v in losses.items() if k != 'ce_loss' and 'loss' in k}
        
        # Check that all losses are finite and strictly non-negative
        assert torch.isfinite(kld_losses["mi_loss"]), f"MI loss not finite at temperature {temp}"
        assert torch.isfinite(kld_losses["dwkl_loss"]), f"DWKL loss not finite at temperature {temp}"
        assert torch.isfinite(kld_losses["tc_loss"]), f"TC loss not finite at temperature {temp}"
        assert torch.isfinite(kld_losses["kl_loss"]), f"KL loss not finite at temperature {temp}"
        
        assert kld_losses["mi_loss"] >= 0, f"MI loss negative at temperature {temp}"
        assert kld_losses["dwkl_loss"] >= 0, f"DWKL loss negative at temperature {temp}"
        assert kld_losses["tc_loss"] >= 0, f"TC loss negative at temperature {temp}"
        assert kld_losses["kl_loss"] >= 0, f"KL loss negative at temperature {temp}"

def test_grid_dvae_config_serialization():
    # Create a config with some values
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    
    # Convert to dict
    config_dict = config.to_dict()
    
    # Check that all base attributes are present
    assert config_dict['n_dim'] == 128
    assert config_dict['n_head'] == 8
    assert config_dict['n_layers'] == 6
    assert config_dict['n_codes'] == 8
    assert config_dict['codebook_size'] == 512
    assert config_dict['max_grid_height'] == 32
    assert config_dict['max_grid_width'] == 32
    assert config_dict['n_vocab'] == 16
    
    # Check that computed attributes are present
    assert 'n_pos' in config_dict
    assert 'compression_factor' in config_dict
    assert 'bottleneck_widths' in config_dict
    
    # Create new config from dict
    new_config = GridDVAEConfig.from_dict(config_dict)
    
    # Check that all attributes match
    assert new_config.n_dim == config.n_dim
    assert new_config.n_head == config.n_head
    assert new_config.n_layers == config.n_layers
    assert new_config.n_codes == config.n_codes
    assert new_config.codebook_size == config.codebook_size
    assert new_config.max_grid_height == config.max_grid_height
    assert new_config.max_grid_width == config.max_grid_width
    assert new_config.n_vocab == config.n_vocab
    
    # Check that computed attributes match
    assert new_config.n_pos == config.n_pos
    assert new_config.compression_factor == config.compression_factor
    assert new_config.bottleneck_widths == config.bottleneck_widths, \
        f"Bottleneck widths don't match.\nExpected: {config.bottleneck_widths}\nGot: {new_config.bottleneck_widths}"

def test_grid_dvae_config_serialization_with_defaults():
    # Create a minimal config with only required fields
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8
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

def test_grid_dvae_config_json_serialization():
    import json
    
    # Create a config
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8
    )
    
    # Convert to dict
    config_dict = config.to_dict()
    
    # Test JSON serialization
    json_str = json.dumps(config_dict)
    loaded_dict = json.loads(json_str)
    
    # Create new config from loaded dict
    new_config = GridDVAEConfig.from_dict(loaded_dict)
    
    # Check that all attributes match
    assert new_config.n_dim == config.n_dim, \
        f"n_dim mismatch: expected {config.n_dim}, got {new_config.n_dim}"
    assert new_config.n_head == config.n_head, \
        f"n_head mismatch: expected {config.n_head}, got {new_config.n_head}"
    assert new_config.n_layers == config.n_layers, \
        f"n_layers mismatch: expected {config.n_layers}, got {new_config.n_layers}"
    assert new_config.n_codes == config.n_codes, \
        f"n_codes mismatch: expected {config.n_codes}, got {new_config.n_codes}"
    assert new_config.n_pos == config.n_pos, \
        f"n_pos mismatch: expected {config.n_pos}, got {new_config.n_pos}"
    assert new_config.compression_factor == config.compression_factor, \
        f"compression_factor mismatch: expected {config.compression_factor}, got {new_config.compression_factor}"
    assert new_config.bottleneck_widths == config.bottleneck_widths, \
        f"bottleneck_widths mismatch.\nExpected: {config.bottleneck_widths}\nGot: {new_config.bottleneck_widths}"

@pytest.mark.parametrize("batch_specific_mask,S,K", [
    (False, 16, 8),   # single mask, compression
    (False, 8, 16),   # single mask, expansion
    (False, 16, 16),  # single mask, same size
    (True, 16, 8),    # batch-specific masks, compression
    (True, 8, 16),    # batch-specific masks, expansion
    (True, 16, 16),   # batch-specific masks, same size
])
def test_residual_projection_masking_effect(batch_specific_mask, S, K):
    """Test that masked inputs produce identical outputs when only masked values differ."""
    B, d = 4, 64
    
    proj = ResidualProjection(S=S, K=K, d=d, use_mask_norm=True)
    
    # Create two identical inputs
    x1 = torch.randn(B, S, d)
    x2 = x1.clone()
    
    # Create mask [B, S] or [1, S] depending on batch_specific_mask
    batch_size = B if batch_specific_mask else 1
    mask = torch.rand(batch_size, S) > 0.5
    
    # Modify x2 at masked positions
    mask_expanded = mask if batch_specific_mask else mask.expand(B, -1)
    x2[~mask_expanded] = torch.randn(torch.sum(~mask_expanded).item(), d)
    
    # Apply projection with mask
    output1 = proj(x1, mask=mask)
    output2 = proj(x2, mask=mask)
    
    # Outputs should be identical since differences were only in masked positions
    assert torch.allclose(output1, output2, atol=1e-5)

def test_residual_projection_use_mask_norm():
    # Use a simple configuration where we can manually compute the expected output.
    # Let S = 4 (input sequence length), K = 2 (output sequence length), and d = 1 (one feature channel).
    S, K, d = 4, 2, 1
    B = 1  # batch size
    rp = ResidualProjection(S, K, d, use_mask_norm=True)
    
    # Set the projection weights to 1 and bias to 0 for a deterministic behavior.
    with torch.no_grad():
        rp.proj.weight.fill_(1.0)
        if rp.proj.bias is not None:
            rp.proj.bias.zero_()
    
    # Define an input x: shape (B, S, d). For example, [1, 2, 3, 4].
    x = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])  # shape: (1, 4, 1)
    
    # Case 1: All tokens are valid.
    mask_all = torch.tensor([[True, True, True, True]])
    # The projection sums the input over S: 1+2+3+4 = 10.
    # The normalization factor is computed by projecting the mask:
    # For a mask of [1,1,1,1] and weight ones, the factor is 1+1+1+1 = 4 for each output.
    # Thus, the expected output for each output channel is 10 / 4 = 2.5.
    expected_all = torch.tensor([[[2.5], [2.5]]])  # shape: (1, 2, 1)
    
    out_all = rp(x, mask=mask_all)
    np.testing.assert_allclose(out_all.detach().numpy(), expected_all.numpy(), atol=1e-5)
    
    # Case 2: Only the first two tokens are valid.
    mask_partial = torch.tensor([[True, True, False, False]])
    # Then x becomes [1, 2, 0, 0] so the sum is 1+2 = 3.
    # The normalization factor becomes 1+1+0+0 = 2.
    # So expected output is 3 / 2 = 1.5 for each output.
    expected_partial = torch.tensor([[[1.5], [1.5]]])
    
    out_partial = rp(x, mask=mask_partial)
    np.testing.assert_allclose(out_partial.detach().numpy(), expected_partial.numpy(), atol=1e-5)
    
    # Case 3: Alternating tokens are valid.
    mask_alternate = torch.tensor([[True, False, True, False]])
    # Then x becomes [1, 0, 3, 0] so the sum is 1+3 = 4.
    # The normalization factor becomes 1+0+1+0 = 2.
    # So expected output is 4 / 2 = 2.0 for each output.
    expected_alternate = torch.tensor([[[2.0], [2.0]]])
    
    out_alternate = rp(x, mask=mask_alternate)
    np.testing.assert_allclose(out_alternate.detach().numpy(), expected_alternate.numpy(), atol=1e-5)

@pytest.mark.parametrize("S,K", [
    (16, 8),   # compression
    (8, 16),   # expansion
    (16, 16),  # same size
])
@pytest.mark.parametrize("mask_shape", [
    (1, 16),     # valid: single mask for all batches
    (4, 16),     # valid: batch-specific mask
    (1, 8),      # invalid: sequence length doesn't match
    (2, 16),     # invalid: batch size doesn't match
])
def test_residual_projection_mask_shape_validation(S, K, mask_shape):
    """Test that mask shape validation works correctly."""
    B, d = 4, 64
    proj = ResidualProjection(S=S, K=K, d=d)
    x = torch.randn(B, S, d)
    mask = torch.ones(mask_shape, dtype=torch.bool)  # Create boolean mask
    
    # Check if the mask shape is valid
    is_valid_shape = mask_shape in [(1, S), (B, S)]
    
    if is_valid_shape:
        # Should not raise an error
        try:
            proj(x, mask=mask)
        except (AssertionError, RuntimeError) as e:
            pytest.fail(f"Unexpected error for valid mask shape {mask_shape}: {e}")
    else:
        # Should raise either an AssertionError or RuntimeError
        with pytest.raises((AssertionError, RuntimeError)) as excinfo:
            proj(x, mask=mask)
        error_msg = str(excinfo.value)
        assert ("Expected mask shape" in error_msg or 
                "must match the size" in error_msg), \
            f"Expected error message about mask shape, got: {error_msg}"

# Transformer Projection Tests 

@pytest.mark.parametrize("S,K", [
    (16, 8),   # compression
    (8, 16),   # expansion
    (16, 16),  # same size
])
@pytest.mark.parametrize("batch_specific_mask", [
    False,    # single mask for all batches [1, 1, S]
    True,     # batch-specific masks [B, 1, S]
])
def test_transformer_projection_masking_effect(S, K, batch_specific_mask):
    """Test that masked inputs produce identical outputs when only masked values differ."""
    B, d = 4, 64
    config = GridDVAEConfig(
        n_dim=d,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    
    proj = TransformerProjection(config=config, input_seq_len=S, output_seq_len=K)
    
    # Create two identical inputs
    x1 = torch.randn(B, S, d)
    x2 = x1.clone()
    
    # Create mask [B, S] or [1, S] depending on batch_specific_mask
    batch_size = B if batch_specific_mask else 1
    mask = torch.rand(batch_size, S) > 0.5
    
    # Modify x2 at masked positions with random values
    mask_expanded = mask if batch_specific_mask else mask.expand(B, -1)
    x2[~mask_expanded] = torch.randn(torch.sum(~mask_expanded).item(), d)
    
    # Apply projection with mask
    output1 = proj(x1, mask=mask)
    output2 = proj(x2, mask=mask)
    
    # Outputs should be identical since differences were only in masked positions
    assert torch.allclose(output1, output2, atol=1e-5), \
        f"Outputs differ when only masked positions are changed (S={S}, K={K}, batch_specific_mask={batch_specific_mask})"

@pytest.mark.parametrize("S,K", [
    (16, 8),   # compression
    (8, 16),   # expansion
    (16, 16),  # same size
])
@pytest.mark.parametrize("mask_shape", [
    (1, 16),     # valid: single mask for all batches
    (4, 16),     # valid: batch-specific mask
    (1, 8),      # invalid: sequence length doesn't match
    (2, 16),     # invalid: batch size doesn't match
])
def test_transformer_projection_mask_shape_validation(S, K, mask_shape):
    """Test that mask shape validation works correctly."""
    B, d = 4, 64
    config = GridDVAEConfig(
        n_dim=d,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    
    proj = TransformerProjection(config=config, input_seq_len=S, output_seq_len=K)
    x = torch.randn(B, S, d)
    mask = torch.ones(mask_shape, dtype=torch.bool)  # Create boolean mask
    
    # Check if the mask shape is valid
    is_valid_shape = mask_shape in [(1, S), (B, S)]
    
    if is_valid_shape:
        # Should not raise an error
        try:
            proj(x, mask=mask)
        except (AssertionError, RuntimeError) as e:
            pytest.fail(f"Unexpected error for valid mask shape {mask_shape}: {e}")
    else:
        # Should raise either an AssertionError or RuntimeError
        with pytest.raises((AssertionError, RuntimeError)) as excinfo:
            proj(x, mask=mask)
        error_msg = str(excinfo.value)
        assert ("Expected mask shape" in error_msg or 
                "must match the size" in error_msg), \
            f"Expected error message about mask shape, got: {error_msg}"

def test_transformer_projection_output_shape():
    """Test that output shapes are correct for various configurations."""
    B, d = 4, 64
    config = GridDVAEConfig(
        n_dim=d,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    
    # Test different input/output sequence lengths
    test_sizes = [(16, 8), (8, 16), (16, 16)]
    
    for S, K in test_sizes:
        proj = TransformerProjection(config=config, input_seq_len=S, output_seq_len=K)
        x = torch.randn(B, S, d)
        
        # Test without mask
        output = proj(x)
        assert output.shape == (B, K, d), \
            f"Unexpected output shape for S={S}, K={K}: {output.shape}"
        
        # Test with mask
        mask = torch.ones((1, S), dtype=torch.bool)  # Changed to 2D mask
        output_masked = proj(x, mask=mask)
        assert output_masked.shape == (B, K, d), \
            f"Unexpected output shape with mask for S={S}, K={K}: {output_masked.shape}"
        

# StackedTransformerProjection Tests


@pytest.mark.parametrize("input_seq_len,output_seq_lens", [
    (32, [16, 8, 4]),    # progressive compression
    (4, [8, 16, 32]),    # progressive expansion
    (16, [16, 16, 16]),  # same size
    (32, [16, 32, 8]),   # mixed compression/expansion
])
def test_stacked_transformer_projection_output_shape(input_seq_len, output_seq_lens):
    """Test that output shapes are correct for various configurations."""
    B, d = 4, 64
    config = GridDVAEConfig(
        n_dim=d,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    
    stacked_proj = StackedTransformerProjection(
        config=config,
        input_seq_len=input_seq_len,
        output_seq_lens=output_seq_lens
    )
    x = torch.randn(B, input_seq_len, d)
    
    # Test without mask
    output = stacked_proj(x)
    assert output.shape == (B, output_seq_lens[-1], d), \
        f"Unexpected output shape for input_len={input_seq_len}, final_len={output_seq_lens[-1]}: {output.shape}"
    
    # Test with mask
    mask = torch.ones((1, input_seq_len), dtype=torch.bool)
    output_masked = stacked_proj(x, mask=mask)
    assert output_masked.shape == (B, output_seq_lens[-1], d), \
        f"Unexpected output shape with mask for input_len={input_seq_len}, final_len={output_seq_lens[-1]}: {output_masked.shape}"

@pytest.mark.parametrize("input_seq_len,output_seq_lens", [
    (32, [16, 8, 4]),    # progressive compression
    (16, [16, 16, 16]),  # same size
])
@pytest.mark.parametrize("batch_specific_mask", [
    False,    # single mask for all batches [1, S]
    True,     # batch-specific masks [B, S]
])
def test_stacked_transformer_projection_masking_effect(input_seq_len, output_seq_lens, batch_specific_mask):
    """Test that masked inputs produce identical outputs when only masked values differ."""
    B, d = 4, 64
    config = GridDVAEConfig(
        n_dim=d,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    
    stacked_proj = StackedTransformerProjection(
        config=config,
        input_seq_len=input_seq_len,
        output_seq_lens=output_seq_lens
    )
    
    # Create two identical inputs
    x1 = torch.randn(B, input_seq_len, d)
    x2 = x1.clone()
    
    # Create mask [B, S] or [1, S] depending on batch_specific_mask
    batch_size = B if batch_specific_mask else 1
    mask = torch.rand(batch_size, input_seq_len) > 0.5
    
    # Modify x2 at masked positions with random values
    mask_expanded = mask if batch_specific_mask else mask.expand(B, -1)
    x2[~mask_expanded] = torch.randn(torch.sum(~mask_expanded).item(), d)
    
    # Apply projection with mask
    output1 = stacked_proj(x1, mask=mask)
    output2 = stacked_proj(x2, mask=mask)
    
    # Outputs should be identical since differences were only in masked positions
    assert torch.allclose(output1, output2, atol=1e-5), \
        f"Outputs differ when only masked positions are changed (input_len={input_seq_len}, " \
        f"output_lens={output_seq_lens}, batch_specific_mask={batch_specific_mask})"

@pytest.mark.parametrize("input_seq_len,output_seq_lens,mask_shape", [
    (16, [8, 4], (1, 16)),     # valid: single mask for all batches
    (16, [8, 4], (4, 16)),     # valid: batch-specific mask
    (16, [8, 4], (1, 8)),      # invalid: sequence length doesn't match
    (16, [8, 4], (2, 16)),     # invalid: batch size doesn't match
])
def test_stacked_transformer_projection_mask_shape_validation(input_seq_len, output_seq_lens, mask_shape):
    """Test that mask shape validation works correctly."""
    B, d = 4, 64
    config = GridDVAEConfig(
        n_dim=d,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    
    stacked_proj = StackedTransformerProjection(
        config=config,
        input_seq_len=input_seq_len,
        output_seq_lens=output_seq_lens
    )
    x = torch.randn(B, input_seq_len, d)
    mask = torch.ones(mask_shape, dtype=torch.bool)
    
    # Check if the mask shape is valid
    is_valid_shape = mask_shape in [(1, input_seq_len), (B, input_seq_len)]
    
    if is_valid_shape:
        # Should not raise an error
        try:
            stacked_proj(x, mask=mask)
        except (AssertionError, RuntimeError) as e:
            pytest.fail(f"Unexpected error for valid mask shape {mask_shape}: {e}")
    else:
        # Should raise either an AssertionError or RuntimeError
        with pytest.raises((AssertionError, RuntimeError)) as excinfo:
            stacked_proj(x, mask=mask)
        error_msg = str(excinfo.value)
        assert ("Expected mask shape" in error_msg or 
                "must match the size" in error_msg), \
            f"Expected error message about mask shape, got: {error_msg}"


## Transformer Masking Effect Tests

@pytest.mark.parametrize("batch_specific_mask", [
    False,    # single mask for all batches [1, 1, S]
    True,     # batch-specific masks [B, 1, S]
])
def test_transformer_masking_effect(batch_specific_mask):
    """Test that masked inputs produce identical outputs when only masked values differ."""
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    transformer = Transformer(config)
    B, S = 2, config.n_pos
    x1 = torch.randn(B, S, config.n_dim)
    x2 = x1.clone()

    # Create mask based on batch_specific parameter
    mask_percentage = 0.5
    if batch_specific_mask:
        mask = torch.rand(B, 1, S) > mask_percentage
        mask_expanded = mask.squeeze(1)  # [B, S]
    else:
        mask = torch.rand(1, 1, S) > mask_percentage
        mask_expanded = mask.expand(B, -1, -1).squeeze(1)  # [B, S]

    # Alter x2 at masked positions
    x2[~mask_expanded] = torch.randn(torch.sum(~mask_expanded).item(), config.n_dim)

    # Create position indices using DVAE.create_grid_position_tensor
    grid_height = int(S**0.5)
    grid_width = grid_height
    positions = GridDVAE.create_grid_position_tensor(grid_height, grid_width, requires_grad=False)
    positions = positions.unsqueeze(0).expand(B, -1, -1)  # Expand to [B, S, 2]

    # Test forward pass with the same mask
    output1, _ = transformer(x1, attn_mask=mask, positions=positions)
    output2, _ = transformer(x2, attn_mask=mask, positions=positions)

    # Get appropriate indices based on mask type
    unmasked_indices = mask_expanded  # [B, S]
    masked_indices = ~mask_expanded   # [B, S]

    # Check that the outputs are the same for unmasked positions
    assert torch.allclose(output1[unmasked_indices], output2[unmasked_indices], atol=1e-5), \
        f"Outputs differ at unmasked positions (batch_specific_mask={batch_specific_mask})"

    # Check that the outputs are different for masked positions
    assert not torch.allclose(output1[masked_indices], output2[masked_indices], atol=1e-5), \
        f"Outputs are the same at masked positions (batch_specific_mask={batch_specific_mask})"


@pytest.mark.parametrize("batch_specific_mask,mask_shape,expected_error", [
    (True, (2, 1, 1024), None),              # valid: batch-specific mask
    (False, (1, 1, 1024), None),             # valid: single mask for all batches
    (True, (2, 1, 8), RuntimeError),         # invalid: wrong sequence length
    (True, (2, 2, 1024), RuntimeError),      # invalid: wrong middle dimension
])
def test_transformer_mask_shape_validation(batch_specific_mask, mask_shape, expected_error):
    """Test that mask shape validation works correctly."""
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    transformer = Transformer(config)
    B, S = 2, config.n_pos  # S is 1024 (32*32)
    x = torch.randn(B, S, config.n_dim)
    
    # Create position indices
    grid_height = int(S**0.5)
    grid_width = grid_height
    positions = GridDVAE.create_grid_position_tensor(grid_height, grid_width, requires_grad=False)
    positions = positions.unsqueeze(0).expand(B, -1, -1)
    
    # Create mask with specified shape
    mask = torch.ones(mask_shape, dtype=torch.bool)
    
    if expected_error is None:
        # Should not raise an error
        try:
            transformer(x, attn_mask=mask, positions=positions)
        except Exception as e:
            pytest.fail(f"Unexpected error for valid mask shape {mask_shape}: {e}")
    else:
        # Should raise the expected error
        with pytest.raises(expected_error) as excinfo:
            transformer(x, attn_mask=mask, positions=positions)
        error_msg = str(excinfo.value)
        assert "size" in error_msg.lower(), \
            f"Expected error message about tensor size mismatch, got: {error_msg}"

def test_dvae_encode_with_hardness():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))

    # Test with different hardness values
    hardness_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for hardness in hardness_values:
        code, soft_code = dvae.encode(x, tau=0.9, hardness=hardness)
        
        # Check shapes
        assert code.shape == (batch_size, config.n_codes, config.codebook_size)
        assert soft_code.shape == (batch_size, config.n_codes, config.codebook_size)
        
        # Check that probabilities sum to 1
        assert torch.allclose(code.sum(dim=-1), torch.ones_like(code.sum(dim=-1)))
        assert torch.allclose(soft_code.sum(dim=-1), torch.ones_like(soft_code.sum(dim=-1)))
        
        # For hardness=0, code should equal soft_code
        if hardness == 0.0:
            assert torch.allclose(code, soft_code)
        # For hardness=1, code should be one-hot
        elif hardness == 1.0:
            assert (code.max(dim=-1)[0] == 1.0).all()

def test_dvae_encode_with_hardness_and_reinmax():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))

    # Test ReinMax with zero hardness (should raise assertion)
    with pytest.raises(AssertionError):
        dvae.encode(x, tau=0.9, hardness=0.0, reinMax=True)

    # Test ReinMax with non-zero hardness
    hardness_values = [0.25, 0.5, 0.75, 1.0]
    
    for hardness in hardness_values:
        code, soft_code = dvae.encode(x, tau=0.9, hardness=hardness, reinMax=True)
        
        # Check shapes and probability distributions
        assert code.shape == (batch_size, config.n_codes, config.codebook_size)
        assert soft_code.shape == (batch_size, config.n_codes, config.codebook_size)
        assert torch.allclose(code.sum(dim=-1), torch.ones_like(code.sum(dim=-1)))
        assert torch.allclose(soft_code.sum(dim=-1), torch.ones_like(soft_code.sum(dim=-1)))
        
        # Code should differ from soft_code when using ReinMax
        assert not torch.allclose(code, soft_code)

def test_hardness_gradient_flow():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,
        codebook_size=512,
        max_grid_height=32,
        max_grid_width=32,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos), requires_grad=False)

    # Test gradient flow with different hardness values
    hardness_values = [0.0, 0.5, 1.0]
    
    for hardness in hardness_values:
        # Clear gradients
        dvae.zero_grad()
        
        code, _ = dvae.encode(x, tau=0.9, hardness=hardness)
        loss = code.sum()
        loss.backward()

        # Check encoder gradients exist
        encoder_modules = [dvae.encoder_bottleneck, dvae.encoder_base, dvae.encoder_head, dvae.embd]
        for module in encoder_modules:
            for name, param in module.named_parameters():
                assert param.grad is not None, f"Gradient for {name} is None with hardness={hardness}"