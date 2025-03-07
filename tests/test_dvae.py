"""
Tests for the DVAE implementation in pips/dvae.py.

The tests check:
  - Utility functions (is_power_of_two, is_perfect_square)
  - Forward and backward passes (i.e. gradient flow) for all modules:
      • RotaryPositionalEmbeddings (including an edge-case with negative positions)
      • RoPE2D (with both broadcast and per-head positions)
      • SwiGLUFFN
      • RMSNorm
      • MultiHeadAttention (with and without a RoPE instance, with cached KV, and using an attention mask)
      • TransformerBlock, Transformer, LatentTransformer
      • GumbelCodebook (both training via forward() and inference mode)
      • The complete GridDVAE: forward pass, reconstruction loss, internal methods, and parameter gradient flow.
  - Edge cases (e.g. ensuring that negative positions bypass rotation, and testing masking functionality in GridDVAE)
"""

import math
import torch
import pytest
import torch.nn.functional as F

from pips.dvae import (
    is_power_of_two,
    is_perfect_square,
    RotaryPositionalEmbeddings,
    RoPE2D,
    SwiGLUFFN,
    RMSNorm,
    MultiHeadAttention,
    TransformerBlock,
    Transformer,
    LatentTransformer,
    GumbelCodebook,
    GridDVAEConfig,
    GridDVAE
)

# --------------------------
# Utility function tests
# --------------------------

def test_is_power_of_two():
    assert is_power_of_two(1) is True
    assert is_power_of_two(2) is True
    assert is_power_of_two(4) is True
    assert is_power_of_two(16) is True
    assert is_power_of_two(3) is False
    assert is_power_of_two(0) is False

def test_is_perfect_square():
    assert is_perfect_square(0) is True
    assert is_perfect_square(1) is True
    assert is_perfect_square(4) is True
    assert is_perfect_square(16) is True
    assert is_perfect_square(14) is False
    assert is_perfect_square(-4) is False

# --------------------------
# RotaryPositionalEmbeddings tests
# --------------------------

def test_rotary_positional_embeddings_basic():
    batch, heads, seq_len, dim = 2, 2, 5, 8  # dim must be even
    model = RotaryPositionalEmbeddings(dim=dim, max_seq_len=10)
    x = torch.randn(batch, heads, seq_len, dim, requires_grad=True)
    # Use a positive positions tensor of shape [B, 1, S] (will be broadcast)
    pos = torch.randint(low=0, high=10, size=(batch, 1, seq_len))
    out = model(x, pos)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None

def test_rotary_positional_embeddings_negative_pos():
    # Test that positions with negative values do not get rotated.
    batch, heads, seq_len, dim = 1, 1, 3, 8
    model = RotaryPositionalEmbeddings(dim=dim, max_seq_len=10)
    x = torch.randn(batch, heads, seq_len, dim)
    # Create positions with one negative value; negative positions bypass RoPE.
    pos = torch.tensor([[[0, -1, 2]]])
    out = model(x, pos)
    # Reshape x to compute the unrotated version
    x_shaped = x.reshape(batch, heads, seq_len, -1, 2).flatten(-2)
    # For token at index 1 (pos < 0), the output should equal the original value.
    assert torch.allclose(out[:, :, 1, :], x_shaped[:, :, 1, :], atol=1e-5)

def test_rotary_reset_parameters():
    """Test that calling reset_parameters rebuilds internal buffers without error."""
    batch, heads, seq_len, dim = 1, 1, 3, 8
    model = RotaryPositionalEmbeddings(dim=dim, max_seq_len=10)
    x = torch.randn(batch, heads, seq_len, dim)
    pos = torch.randint(low=0, high=10, size=(batch, 1, seq_len))
    out1 = model(x, pos)
    model.reset_parameters()
    out2 = model(x, pos)
    assert out1.shape == out2.shape

# --------------------------
# RoPE2D tests
# --------------------------

def test_rope2d_basic():
    batch, heads, seq_len, dim = 2, 2, 5, 8  # dim splits into two halves (each 4)
    model = RoPE2D(dim=dim, max_height=10, max_width=10)
    x = torch.randn(batch, heads, seq_len, dim, requires_grad=True)
    # Provide positions as [B, 1, S, 2]
    pos = torch.randint(low=0, high=10, size=(batch, 1, seq_len, 2))
    out = model(x, pos)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None

def test_rope2d_with_heads_positions():
    batch, heads, seq_len, dim = 2, 3, 4, 8
    model = RoPE2D(dim=dim, max_height=10, max_width=10)
    x = torch.randn(batch, heads, seq_len, dim, requires_grad=True)
    # Provide positions as [B, H, S, 2]
    pos = torch.randint(low=0, high=10, size=(batch, heads, seq_len, 2))
    out = model(x, pos)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None

# --------------------------
# SwiGLUFFN tests
# --------------------------

def test_swigluffn():
    batch, features = 4, 16
    model = SwiGLUFFN(dim=features, hidden_dim=32)
    x = torch.randn(batch, features, requires_grad=True)
    out = model(x)
    assert out.shape == (batch, features)
    out.sum().backward()
    assert x.grad is not None

# --------------------------
# RMSNorm tests
# --------------------------

def test_rmsnorm():
    batch, features = 4, 16
    model = RMSNorm(dim=features)
    x = torch.randn(batch, features, requires_grad=True)
    out = model(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None

# --------------------------
# MultiHeadAttention tests
# --------------------------

def test_multiheadattention_basic():
    batch, seq_len, d_model, n_head = 2, 5, 32, 4
    # Test without a RoPE module.
    mha = MultiHeadAttention(n_dim=d_model, n_head=n_head, dropout=0.0, rope=None)
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)
    out, kv_cache = mha(x, x, x, attn_mask=None)
    assert out.shape == (batch, seq_len, d_model)
    out.sum().backward()
    assert x.grad is not None

def test_multiheadattention_with_rope():
    batch, seq_len, d_model, n_head = 2, 5, 32, 4
    # With RoPE2D (each head's dim = d_model // n_head).
    rope = RoPE2D(dim=d_model // n_head, max_height=seq_len, max_width=seq_len)
    mha = MultiHeadAttention(n_dim=d_model, n_head=n_head, dropout=0.0, rope=rope)
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)
    # Provide positions as 2D tensor: [B, S, 2]
    positions = torch.stack([torch.arange(seq_len), torch.arange(seq_len)], dim=-1)\
                     .unsqueeze(0).expand(batch, seq_len, 2)
    out, kv_cache = mha(x, x, x, positions=positions)
    assert out.shape == (batch, seq_len, d_model)
    out.sum().backward()
    assert x.grad is not None

def test_multiheadattention_kv_cache():
    batch, seq_len, d_model, n_head = 2, 3, 32, 4
    rope = RoPE2D(dim=d_model // n_head, max_height=seq_len+2, max_width=seq_len+2)
    mha = MultiHeadAttention(n_dim=d_model, n_head=n_head, dropout=0.0, rope=rope)
    x = torch.randn(batch, seq_len, d_model)
    # Provide positions as 2D tensor: [B, S, 2]
    positions = torch.stack([torch.arange(seq_len), torch.arange(seq_len)], dim=-1)\
                     .unsqueeze(0).expand(batch, seq_len, 2)
    # First call: no cached key/values.
    out1, _ = mha(x, x, x, positions=positions)
    # Create dummy past key/values (simulate 2 past tokens).
    dummy_k = torch.randn(batch, n_head, 2, d_model // n_head)
    dummy_v = torch.randn(batch, n_head, 2, d_model // n_head)
    kv_cache = (dummy_k, dummy_v)
    # Second call: with cache provided.
    out2, new_kv_cache = mha(x, x, x, positions=positions, kv_cache=kv_cache, return_kv_cache=True)
    assert new_kv_cache is not None
    new_k, new_v = new_kv_cache
    # New key should have dummy tokens + current tokens.
    assert new_k.shape[2] == 2 + seq_len

def test_multiheadattention_with_attn_mask():
    batch, seq_len, d_model, n_head = 2, 5, 32, 4
    mha = MultiHeadAttention(n_dim=d_model, n_head=n_head, dropout=0.0, rope=None)
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)
    # Create a causal attention mask (upper triangular with -inf entries) of shape [B, T, T]
    attn_mask = torch.triu(torch.full((batch, seq_len, seq_len), float('-inf')), diagonal=1)
    out, _ = mha(x, x, x, attn_mask=attn_mask)
    assert out.shape == (batch, seq_len, d_model)
    out.sum().backward()
    assert x.grad is not None

# --------------------------
# TransformerBlock tests
# --------------------------

def test_transformerblock():
    batch, seq_len, d_model, n_head = 2, 5, 32, 4
    block = TransformerBlock(d_model=d_model, n_head=n_head, rope=None, dim_feedforward=64, dropout=0.0)
    queries = torch.randn(batch, seq_len, d_model, requires_grad=True)
    context = torch.randn(batch, seq_len, d_model)
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch, seq_len)
    out, _ = block(queries, context, positions=positions)
    assert out.shape == queries.shape
    out.sum().backward()
    assert queries.grad is not None

# --------------------------
# Transformer tests
# --------------------------

def test_transformer():
    batch, seq_len, d_model, n_head, n_layer = 2, 5, 32, 4, 2
    transformer = Transformer(d_model=d_model, n_head=n_head, n_layer=n_layer, rope=None)
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch, seq_len)
    out, kv_caches = transformer(x, positions=positions, attn_mask=None, return_kv_caches=True)
    assert out.shape == (batch, seq_len, d_model)
    # Check one kv_cache per layer.
    assert len(kv_caches) == n_layer
    out.sum().backward()
    assert x.grad is not None

# --------------------------
# LatentTransformer tests
# --------------------------

def test_latenttransformer():
    batch, n_latent, d_model, n_head, n_layer, seq_len = 2, 4, 32, 4, 2, 5
    latent_transformer = LatentTransformer(n_latent=n_latent, d_model=d_model, n_head=n_head, n_layer=n_layer, rope=None)
    # Context input: [B, seq_len, d_model]
    context = torch.randn(batch, seq_len, d_model)
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch, seq_len)
    out, kv_caches = latent_transformer(context, positions=positions, attn_mask=None, return_kv_caches=True)
    assert out.shape == (batch, n_latent, d_model)

# --------------------------
# GumbelCodebook tests
# --------------------------

def test_gumbelcodebook_forward_backward():
    batch, N, d_model, codebook_size = 2, 10, 32, 16
    model = GumbelCodebook(d_model=d_model, codebook_size=codebook_size, use_exp_relaxed=False)
    logits = torch.randn(batch, N, d_model, requires_grad=True)
    quantized, log_alpha, z = model(logits, tau=0.5)
    assert quantized.shape == (batch, N, d_model)
    assert log_alpha.shape == (batch, N, codebook_size)
    assert z.shape == (batch, N, codebook_size)
    quantized.sum().backward()
    assert logits.grad is not None

def test_gumbelcodebook_inference():
    batch, N, d_model, codebook_size = 2, 10, 32, 16
    model = GumbelCodebook(d_model=d_model, codebook_size=codebook_size, use_exp_relaxed=False)
    logits = torch.randn(batch, N, d_model)
    quantized, log_alpha, hard_one_hot = model.inference(logits)
    assert quantized.shape == (batch, N, d_model)
    assert log_alpha.shape == (batch, N, codebook_size)
    assert hard_one_hot.shape == (batch, N, codebook_size)
    one_hot_sum = hard_one_hot.sum(dim=-1)
    assert torch.allclose(one_hot_sum, torch.ones_like(one_hot_sum))

# --------------------------
# GridDVAE tests
# --------------------------

@pytest.fixture
def small_config():
    # Use a small configuration for testing.
    config_dict = {
        'n_dim': 32,
        'n_head': 4,
        'n_grid_layer': 1,
        'n_latent_layer': 1,
        'n_codes': 4,           # must be a power of 2
        'codebook_size': 16,
        'rope_base_height': 10007,
        'rope_base_width': 5003,
        'dropout': 0.0,
        'max_grid_height': 4,   # grid height: 4x4 gives n_pos=16 (a power of 2)
        'max_grid_width': 4,
        'n_vocab': 16,
        'padding_idx': None,
        'mask_idx': None,
        'pad_weight': 0.01,
        'use_exp_relaxed': False,
        'use_monte_carlo_kld': False
    }
    return GridDVAEConfig(**config_dict)

def test_griddvae_forward_backward(small_config):
    model = GridDVAE(small_config)
    batch = 2
    S = small_config.n_pos  # n_pos = max_grid_height * max_grid_width (e.g., 16)
    x = torch.randint(0, small_config.n_vocab, (batch, S))
    decoded_logits, log_alpha, losses, q_z_marg = model(x, tau=0.5, mask_percentage=0.5)
    assert decoded_logits.shape == (batch, S, small_config.n_vocab)
    assert "ce_loss" in losses
    loss = losses["ce_loss"]
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None

def test_apply_mask(small_config):
    model = GridDVAE(small_config)
    x = torch.tensor([
        [1, 2, 3, small_config.n_vocab - 1, 4],
        [small_config.n_vocab - 1, 1, 2, 3, 4]
    ])
    x_masked = model.apply_mask(x, mask_percentage=1.0)
    expected = torch.where(x == small_config.padding_idx,
                           x,
                           torch.full_like(x, small_config.mask_idx))
    assert torch.equal(x_masked, expected)

def test_reconstruction_loss():
    batch, seq_len, n_vocab = 2, 10, 16
    decoded_logits = torch.randn(batch, seq_len, n_vocab, requires_grad=True)
    x = torch.randint(0, n_vocab, (batch, seq_len))
    pad_value = n_vocab - 1  # usually the pad token
    config = GridDVAEConfig(
        n_dim=32,
        n_head=4,
        n_grid_layer=1,
        n_latent_layer=1,
        n_codes=4,
        codebook_size=16,
        max_grid_height=4,
        max_grid_width=4,
        n_vocab=n_vocab,
        padding_idx=pad_value,
        mask_idx=n_vocab - 2,
        pad_weight=0.01,
        use_exp_relaxed=False,
        use_monte_carlo_kld=False
    )
    model = GridDVAE(config)
    loss = model.reconstruction_loss(decoded_logits, x, pad_value=pad_value, pad_weight=0.01)
    loss.backward()
    assert decoded_logits.grad is not None

# -------------------------------------------
# Additional tests to verify masking correctness
# -------------------------------------------

def test_apply_mask_no_mask(small_config):
    """When mask_percentage is 0, the input should remain unchanged."""
    model = GridDVAE(small_config)
    input_tensor = torch.randint(0, small_config.n_vocab, (2, 10))
    output_tensor = model.apply_mask(input_tensor, mask_percentage=0.0)
    assert torch.equal(input_tensor, output_tensor)

def test_apply_mask_random_reproducibility(small_config):
    """Ensure that setting a seed produces deterministic masking."""
    torch.manual_seed(42)
    model = GridDVAE(small_config)
    input_tensor = torch.randint(0, small_config.n_vocab, (1, 10))
    # Save the current RNG state
    rng_state = torch.get_rng_state()
    output1 = model.apply_mask(input_tensor, mask_percentage=0.5)
    torch.set_rng_state(rng_state)
    output2 = model.apply_mask(input_tensor, mask_percentage=0.5)
    assert torch.equal(output1, output2)

def test_apply_mask_preserves_pad(small_config):
    """Pad tokens should always remain unchanged by masking."""
    model = GridDVAE(small_config)
    pad_val = small_config.padding_idx
    input_tensor = torch.tensor([[pad_val, 1, 2, 3, 4]])
    torch.manual_seed(0)
    output_tensor = model.apply_mask(input_tensor, mask_percentage=0.8)
    assert output_tensor[0, 0] == pad_val

# --------------------------
# Additional tests for GridDVAE internal methods
# --------------------------

def test_griddvae_encode(small_config):
    """Test that the encode method returns latent representations of shape [B, n_codes, n_dim]."""
    model = GridDVAE(small_config)
    batch = 2
    S = small_config.n_pos
    x = torch.randint(0, small_config.n_vocab, (batch, S))
    grid_pos = model.grid_pos_indices.expand(batch, -1, -1)
    latent_pos = model.latent_pos_indices.expand(batch, -1)
    latent_rep = model.encode(x, grid_pos, latent_pos)
    expected_shape = (batch, small_config.n_codes, small_config.n_dim)
    assert latent_rep.shape == expected_shape, f"Expected {expected_shape}, got {latent_rep.shape}"

def test_griddvae_decode(small_config):
    """Test that the decode method returns logits of shape [B, n_pos, n_vocab]."""
    model = GridDVAE(small_config)
    batch = 2
    latent = torch.randn(batch, small_config.n_codes, small_config.n_dim)
    grid_pos = model.grid_pos_indices.expand(batch, -1, -1)
    latent_pos = model.latent_pos_indices.expand(batch, -1)
    decoded_logits = model.decode(latent, grid_pos, latent_pos)
    expected_shape = (batch, small_config.n_pos, small_config.n_vocab)
    assert decoded_logits.shape == expected_shape, f"Expected {expected_shape}, got {decoded_logits.shape}"

def test_griddvae_qz_marg_propagation(small_config):
    """Test that the provided q_z_marg is passed through unchanged in the forward method."""
    # Override to force the Monte Carlo KLD branch so that q_z_marg is unmodified.
    small_config.use_monte_carlo_kld = True
    model = GridDVAE(small_config)
    batch = 2
    S = small_config.n_pos  # n_pos = max_grid_height * max_grid_width (e.g. 16)
    x = torch.randint(0, small_config.n_vocab, (batch, S))
    # Create dummy q_z_marg with the expected shape [n_codes, codebook_size]
    dummy_q_z_marg = torch.randn(small_config.n_codes, small_config.codebook_size)
    _, _, _, q_z_marg_out = model(x, q_z_marg=dummy_q_z_marg, tau=0.5, mask_percentage=0.0)
    assert torch.equal(dummy_q_z_marg, q_z_marg_out)
