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
    AttentionPool,
    StackedPooling,
    Transformer
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
        n_pos=1024,
        n_vocab=16
    )
    assert config.compression_factor == 128  # 1024/8
    assert config.max_grid_height == 32  # sqrt(1024)
    assert config.max_grid_width == 32
    
    # Test invalid configurations
    with pytest.raises(ValueError):
        GridDVAEConfig(n_dim=65, n_head=8, n_layers=6, n_pos=1024, n_vocab=16, n_codes=8)
    
    with pytest.raises(AssertionError):
        GridDVAEConfig(n_dim=128, n_head=8, n_layers=6, n_pos=1000, n_vocab=16, n_codes=8)
    
    with pytest.raises(AssertionError):
        GridDVAEConfig(n_dim=128, n_head=8, n_layers=6, n_pos=2048, n_vocab=16, n_codes=8)

# Test DVAE
def test_dvae():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,  # Directly specify the number of codes
        codebook_size=512,
        n_pos=1024,
        n_vocab=16
    )
    
    dvae = GridDVAE(config)
    
    # Test forward pass
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))
    
    output = dvae(x)
    
    # Test output shape
    assert output.shape == (batch_size, config.n_pos, config.n_vocab)
    
    # Convert to probabilities and test
    probs = F.softmax(output, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)))
    
    # Test that output contains logits (not all zeros or NaNs)
    assert not torch.allclose(output, torch.zeros_like(output))
    assert not torch.any(torch.isnan(output)) 

def test_attention_pool_no_mask():
    dim = 64
    num_queries = 10
    B, S = 4, 20
    x = torch.randn(B, S, dim)
    
    pool = AttentionPool(dim=dim, num_queries=num_queries)
    output = pool(x)
    
    assert output.shape == (B, num_queries, dim), f"Unexpected output shape: {output.shape}"

def test_attention_pool_mask_shape_1_1_S():
    dim = 64
    num_queries = 10
    B, S = 4, 20
    x = torch.randn(B, S, dim)
    attn_mask = torch.ones(1, 1, S)
    
    pool = AttentionPool(dim=dim, num_queries=num_queries)
    output = pool(x, attn_mask=attn_mask)
    
    assert output.shape == (B, num_queries, dim), f"Unexpected output shape: {output.shape}"

def test_attention_pool_mask_shape_B_1_S():
    dim = 64
    num_queries = 10
    B, S = 4, 20
    x = torch.randn(B, S, dim)
    attn_mask = torch.ones(B, 1, S)
    
    pool = AttentionPool(dim=dim, num_queries=num_queries)
    output = pool(x, attn_mask=attn_mask)
    
    assert output.shape == (B, num_queries, dim), f"Unexpected output shape: {output.shape}"

def test_attention_pool_mask_shape_B_K_S():
    dim = 64
    num_queries = 10
    B, S = 4, 20
    x = torch.randn(B, S, dim)
    attn_mask = torch.ones(B, num_queries, S)
    
    pool = AttentionPool(dim=dim, num_queries=num_queries)
    output = pool(x, attn_mask=attn_mask)
    
    assert output.shape == (B, num_queries, dim), f"Unexpected output shape: {output.shape}" 

def test_stacked_pooling_no_mask():
    dim = 64
    pool_sizes = [10, 5, 2]
    B, S = 4, 20
    x = torch.randn(B, S, dim)
    
    stacked_pool = StackedPooling(dim=dim, pool_sizes=pool_sizes)
    output = stacked_pool(x)
    
    assert output.shape == (B, pool_sizes[-1], dim), f"Unexpected output shape: {output.shape}"

def test_stacked_pooling_with_mask():
    dim = 64
    pool_sizes = [10, 5, 2]
    B, S = 4, 20
    x = torch.randn(B, S, dim)
    attn_mask = torch.ones(1, 1, S)
    
    stacked_pool = StackedPooling(dim=dim, pool_sizes=pool_sizes)
    output = stacked_pool(x, attn_mask=attn_mask)
    
    assert output.shape == (B, pool_sizes[-1], dim), f"Unexpected output shape: {output.shape}"

def test_stacked_pooling_with_partial_mask():
    dim = 64
    pool_sizes = [10, 5, 2]
    B, S = 4, 20
    x = torch.randn(B, S, dim)
    attn_mask = torch.cat([torch.zeros(1, 1, S//2), torch.ones(1, 1, S//2)], dim=-1)
    
    stacked_pool = StackedPooling(dim=dim, pool_sizes=pool_sizes)
    output = stacked_pool(x, attn_mask=attn_mask)
    
    assert output.shape == (B, pool_sizes[-1], dim), f"Unexpected output shape: {output.shape}" 

def test_create_random_mask_no_mask():
    dvae = GridDVAE(GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,  # Directly specify the number of codes
        codebook_size=512,
        n_pos=1024,
        n_vocab=16
    ))
    B, S = 4, 1024
    mask = dvae.create_random_mask(B, S, mask_percentage=0.0)
    assert mask is None, "Expected no mask when mask_percentage is 0."

def test_create_random_mask_full_mask():
    dvae = GridDVAE(GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,  # Directly specify the number of codes
        codebook_size=512,
        n_pos=1024,
        n_vocab=16
    ))
    B, S = 4, 1024
    with pytest.raises(ValueError, match="mask_percentage of 1 would mask all tokens, which is not allowed."):
        dvae.create_random_mask(B, S, mask_percentage=1.0)

def test_create_random_mask_partial_mask():
    dvae = GridDVAE(GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,  # Directly specify the number of codes
        codebook_size=512,
        n_pos=1024,
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
        n_codes=8,  # Directly specify the number of codes
        codebook_size=512,
        n_pos=1024,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))

    # Test forward pass with no mask
    output_no_mask = dvae(x, mask_percentage=0.0)
    assert output_no_mask.shape == (batch_size, config.n_pos, config.n_vocab), \
        f"Unexpected output shape: {output_no_mask.shape}"

    # Test forward pass with partial mask
    output_partial_mask = dvae(x, mask_percentage=0.5)
    assert output_partial_mask.shape == (batch_size, config.n_pos, config.n_vocab), \
        f"Unexpected output shape: {output_partial_mask.shape}" 


def test_attention_pool_masking_effect():
    dim = 64
    num_queries = 10
    B, S = 4, 20
    x1 = torch.randn(B, S, dim)
    x2 = x1.clone()

    # Create a random mask
    mask_percentage = 0.5
    mask = torch.rand(1, 1, S) > mask_percentage

    # Alter x2 at masked positions
    mask = mask.squeeze(1).expand(B, -1)  # Ensure mask is [B, S]
    x2[~mask, :] = torch.randn(torch.sum(~mask).item(), dim)

    # Initialize AttentionPool
    pool = AttentionPool(dim=dim, num_queries=num_queries)

    # Test forward pass with the same mask
    output1 = pool(x1, attn_mask=mask.unsqueeze(1))  # Expand mask to [1, 1, S]
    output2 = pool(x2, attn_mask=mask.unsqueeze(1))

    # Check that the outputs are the same
    assert torch.allclose(output1, output2, atol=1e-5), "Outputs differ when only masked positions are changed."

def test_attention_pool_masking_effect_batch_mask():
    dim = 64
    num_queries = 10
    B, S = 4, 20
    x1 = torch.randn(B, S, dim)
    x2 = x1.clone()

    # Create a random mask for each batch
    mask_percentage = 0.5
    mask = torch.rand(B, 1, S) > mask_percentage

    # Alter x2 at masked positions
    x2[~mask.squeeze(1)] = torch.randn(torch.sum(~mask).item(), dim)

    # Initialize AttentionPool
    pool = AttentionPool(dim=dim, num_queries=num_queries)

    # Test forward pass with the same mask
    output1 = pool(x1, attn_mask=mask)
    output2 = pool(x2, attn_mask=mask)

    # Check that the outputs are the same
    assert torch.allclose(output1, output2, atol=1e-5), "Outputs differ when only masked positions are changed."

def test_stacked_pooling_masking_effect_single_mask():
    dim = 64
    pool_sizes = [10, 5, 2]
    B, S = 4, 20
    x1 = torch.randn(B, S, dim)
    x2 = x1.clone()

    # Create a random mask
    mask_percentage = 0.5
    mask = torch.rand(1, 1, S) > mask_percentage

    # Alter x2 at masked positions
    mask = mask.squeeze(1).expand(B, -1)  # Ensure mask is [B, S]
    x2[~mask] = torch.randn(torch.sum(~mask).item(), dim)

    # Initialize StackedPooling
    stacked_pool = StackedPooling(dim=dim, pool_sizes=pool_sizes)

    # Test forward pass with the same mask
    output1 = stacked_pool(x1, attn_mask=mask.unsqueeze(1))  # Expand mask to [1, 1, S]
    output2 = stacked_pool(x2, attn_mask=mask.unsqueeze(1))

    # Check that the outputs are the same
    assert torch.allclose(output1, output2, atol=1e-5), "Outputs differ when only masked positions are changed."

def test_stacked_pooling_masking_effect_batch_mask():
    dim = 64
    pool_sizes = [10, 5, 2]
    B, S = 4, 20
    x1 = torch.randn(B, S, dim)
    x2 = x1.clone()

    # Create a random mask for each batch
    mask_percentage = 0.5
    mask = torch.rand(B, 1, S) > mask_percentage

    # Alter x2 at masked positions
    x2[~mask.squeeze(1)] = torch.randn(torch.sum(~mask).item(), dim)

    # Initialize StackedPooling
    stacked_pool = StackedPooling(dim=dim, pool_sizes=pool_sizes)

    # Test forward pass with the same mask
    output1 = stacked_pool(x1, attn_mask=mask)
    output2 = stacked_pool(x2, attn_mask=mask)

    # Check that the outputs are the same
    assert torch.allclose(output1, output2, atol=1e-5), "Outputs differ when only masked positions are changed."


def test_transformer_masking_effect():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,  # Directly specify the number of codes
        codebook_size=512,
        n_pos=1024,
        n_vocab=16
    )
    transformer = Transformer(config)
    B, S = 2, config.n_pos
    x1 = torch.randn(B, S, config.n_dim)
    x2 = x1.clone()

    # Create a random mask
    mask_percentage = 0.5
    mask = torch.rand(B, 1, S) > mask_percentage

    # Alter x2 at masked positions
    x2[~mask.squeeze(1)] = torch.randn(torch.sum(~mask).item(), config.n_dim)

    # Create position indices using DVAE.create_grid_position_tensor
    grid_height = int(S**0.5)
    grid_width = grid_height
    positions = GridDVAE.create_grid_position_tensor(grid_height, grid_width, requires_grad=False)
    positions = positions.unsqueeze(0).expand(B, -1, -1)  # Expand to [B, S, 2]

    # Test forward pass with the same mask
    output1, _ = transformer(x1, attn_mask=mask, positions=positions)
    output2, _ = transformer(x2, attn_mask=mask, positions=positions)

    # Check that the outputs are the same for unmasked positions
    assert torch.allclose(output1[mask.squeeze(1)], output2[mask.squeeze(1)], atol=1e-5), \
        "Outputs differ at unmasked positions."

    # Check that the outputs are different for masked positions
    assert not torch.allclose(output1[~mask.squeeze(1)], output2[~mask.squeeze(1)], atol=1e-5), \
        "Outputs are the same at masked positions."

def test_transformer_masking_effect_single_mask():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,  # Directly specify the number of codes
        codebook_size=512,
        n_pos=1024,
        n_vocab=16
    )
    transformer = Transformer(config)
    B, S = 2, config.n_pos
    x1 = torch.randn(B, S, config.n_dim)
    x2 = x1.clone()

    # Create a single mask for all batches
    mask_percentage = 0.5
    mask = torch.rand(1, 1, S) > mask_percentage

    # Alter x2 at masked positions
    mask_expanded = mask.expand(B, -1, -1).squeeze(1)  # Expand mask to [B, S]
    x2[~mask_expanded] = torch.randn(torch.sum(~mask_expanded).item(), config.n_dim)

    # Create position indices using DVAE.create_grid_position_tensor
    grid_height = int(S**0.5)
    grid_width = grid_height
    positions = GridDVAE.create_grid_position_tensor(grid_height, grid_width, requires_grad=False)
    positions = positions.unsqueeze(0).expand(B, -1, -1)  # Expand to [B, S, 2]

    # Test forward pass with the same mask
    output1, _ = transformer(x1, attn_mask=mask, positions=positions)
    output2, _ = transformer(x2, attn_mask=mask, positions=positions)

    # Check that the outputs are the same for unmasked positions
    unmasked_indices = mask.squeeze(0).expand(B, -1)  # Expand mask to [B, S]
    assert torch.allclose(output1[unmasked_indices], output2[unmasked_indices], atol=1e-5), \
        "Outputs differ at unmasked positions."

    # Check that the outputs are different for masked positions
    masked_indices = ~mask.squeeze(0).expand(B, -1)  # Expand mask to [B, S]
    assert not torch.allclose(output1[masked_indices], output2[masked_indices], atol=1e-5), \
        "Outputs are the same at masked positions."


def test_dvae_masking_effect():
    config = GridDVAEConfig(
        n_dim=128,
        n_head=8,
        n_layers=6,
        n_codes=8,  # Directly specify the number of codes
        codebook_size=512,
        n_pos=1024,
        n_vocab=16
    )
    dvae = GridDVAE(config)
    B = 2
    S = config.n_pos
    x1 = torch.randint(0, config.n_vocab, (B, S))
    x2 = x1.clone()

    # Test with same_mask_for_all=True and hard=False
    mask_percentage = 0.5
    mask = dvae.create_random_mask(B, config.n_pos, mask_percentage, same_mask_for_all=True)

    if mask is not None:
        mask_tmp = mask.squeeze(1).expand(B, -1)
        x2[~mask_tmp] = torch.randint(0, config.n_vocab, (torch.sum(~mask_tmp).item(),))

    torch.manual_seed(42)
    code1, _ = dvae.encode(x1, attn_mask=mask, tau=0.9, hard=False)

    torch.manual_seed(42)
    code2, _ = dvae.encode(x2, attn_mask=mask, tau=0.9, hard=False)

    # Increase the tolerance to account for floating-point precision issues
    assert torch.allclose(code1, code2, atol=1e-4), "Codes differ when only masked positions are changed."

    # Test with same_mask_for_all=False and hard=False
    mask = dvae.create_random_mask(B, config.n_pos, mask_percentage, same_mask_for_all=False)

    if mask is not None:
        mask_tmp = mask.squeeze(1).expand(B, -1)
        x2[~mask_tmp] = torch.randint(0, config.n_vocab, (torch.sum(~mask_tmp).item(),))

    torch.manual_seed(42)
    code1, _ = dvae.encode(x1, attn_mask=mask, tau=0.9, hard=False)

    torch.manual_seed(42)
    code2, _ = dvae.encode(x2, attn_mask=mask, tau=0.9, hard=False)

    # Increase the tolerance to account for floating-point precision issues
    assert torch.allclose(code1, code2, atol=1e-4), "Codes differ when only masked positions are changed."

    # Test with same_mask_for_all=True and hard=True
    mask = dvae.create_random_mask(B, config.n_pos, mask_percentage, same_mask_for_all=True)

    if mask is not None:
        mask_tmp = mask.squeeze(1).expand(B, -1)
        x2[~mask_tmp] = torch.randint(0, config.n_vocab, (torch.sum(~mask_tmp).item(),))

    torch.manual_seed(42)
    code1, _ = dvae.encode(x1, attn_mask=mask, tau=0.9, hard=True)

    torch.manual_seed(42)
    code2, _ = dvae.encode(x2, attn_mask=mask, tau=0.9, hard=True)

    # Increase the tolerance to account for floating-point precision issues
    assert torch.allclose(code1, code2, atol=1e-4), "Codes differ when only masked positions are changed."

    # Test with same_mask_for_all=False and hard=True
    mask = dvae.create_random_mask(B, config.n_pos, mask_percentage, same_mask_for_all=False)

    if mask is not None:
        mask_tmp = mask.squeeze(1).expand(B, -1)
        x2[~mask_tmp] = torch.randint(0, config.n_vocab, (torch.sum(~mask_tmp).item(),))

    torch.manual_seed(42)
    code1, _ = dvae.encode(x1, attn_mask=mask, tau=0.9, hard=True)

    torch.manual_seed(42)
    code2, _ = dvae.encode(x2, attn_mask=mask, tau=0.9, hard=True)

    # Increase the tolerance to account for floating-point precision issues
    assert torch.allclose(code1, code2, atol=1e-4), "Codes differ when only masked positions are changed."

# Add this test to your existing test suite 