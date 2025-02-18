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
    Transformer,
    ResidualProjection
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
    
    # Test forward pass
    batch_size = 2
    x = torch.randint(0, config.n_vocab, (batch_size, config.n_pos))
    
    output, losses, q_z_marg = dvae(x)
    # Access reconstruction loss and kld losses from the losses dictionary
    reconstruction_loss = losses['ce_loss']
    kld_losses = {k: v for k, v in losses.items() if k != 'ce_loss' and 'loss' in k}
    
    # Test output shape
    assert output.shape == (batch_size, config.n_pos, config.n_vocab)
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
    x1 = torch.randint(0, config.n_vocab, (B, S))
    x2 = x1.clone()

    # Create position indices using DVAE.create_grid_position_tensor
    grid_height = int(S**0.5)
    grid_width = grid_height
    positions = GridDVAE.create_grid_position_tensor(grid_height, grid_width, requires_grad=False)
    positions = positions.unsqueeze(0).expand(B, -1, -1)  # Expand to [B, S, 2]

    def check_encoding_steps(x1, x2, mask, msg_prefix=""):
        mask_tmp = mask.squeeze(1).expand(B, -1)
        x2[~mask_tmp] = torch.randint(0, config.n_vocab, (torch.sum(~mask_tmp).item(),))

        # Debug intermediate representations
        torch.manual_seed(42)
        with torch.no_grad():
            # Get embeddings
            emb1 = dvae.embd(x1)
            emb2 = dvae.embd(x2)
            
            # Check embeddings at unmasked positions
            assert torch.allclose(emb1[mask_tmp], emb2[mask_tmp], atol=1e-5), \
                f"{msg_prefix}Embeddings differ at unmasked positions"

            # Get encoder base output
            enc1, _ = dvae.encoder_base(emb1, mask, positions=positions)
            enc2, _ = dvae.encoder_base(emb2, mask, positions=positions)
            
            # Check encoder base output at unmasked positions
            assert torch.allclose(enc1[mask_tmp], enc2[mask_tmp], atol=1e-5), \
                f"{msg_prefix}Encoder base outputs differ at unmasked positions"

            # Get encoder bottleneck output
            bottleneck1 = dvae.encoder_bottleneck(enc1, mask)
            bottleneck2 = dvae.encoder_bottleneck(enc2, mask)
            
            # Check bottleneck outputs
            assert torch.allclose(bottleneck1, bottleneck2, atol=1e-5), \
                f"{msg_prefix}Encoder bottleneck outputs differ"

            # Get encoder head output (before gumbel-softmax)
            head1 = dvae.encoder_head(bottleneck1)
            head2 = dvae.encoder_head(bottleneck2)
            
            # Check encoder head output
            assert torch.allclose(head1, head2, atol=1e-5), \
                f"{msg_prefix}Encoder head outputs differ"

            # Apply gumbel-softmax directly with same random seed
            torch.manual_seed(42)
            gumbel1 = F.gumbel_softmax(head1, tau=0.9, hard=False)
            
            torch.manual_seed(42)
            gumbel2 = F.gumbel_softmax(head2, tau=0.9, hard=False)
            
            # Check gumbel-softmax outputs
            assert torch.allclose(gumbel1, gumbel2, atol=1e-4), \
                f"{msg_prefix}Gumbel-softmax outputs differ"

            # Get final codes with fixed random seed
            torch.manual_seed(42)
            code1, _ = dvae.encode(x1, attn_mask=mask, tau=0.9, hard=False)
            
            torch.manual_seed(42)
            code2, _ = dvae.encode(x2, attn_mask=mask, tau=0.9, hard=False)
            
            # Check final codes
            assert torch.allclose(code1, code2, atol=1e-4), \
                f"{msg_prefix}Codes differ when only masked positions are changed"

    # Test with same_mask_for_all=True and hard=False
    mask_percentage = 0.5
    mask = dvae.create_random_mask(B, config.n_pos, mask_percentage, same_mask_for_all=True)
    if mask is not None:
        check_encoding_steps(x1, x2.clone(), mask, "Same mask for all: ")

    # Test with same_mask_for_all=False and hard=False
    mask = dvae.create_random_mask(B, config.n_pos, mask_percentage, same_mask_for_all=False)
    if mask is not None:
        check_encoding_steps(x1, x2.clone(), mask, "Different masks: ")

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
    hard_code, soft_code = dvae.encode(x, tau=0.9, hard=True, reinMax=True)

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
    output, losses, q_z_marg = dvae(x, tau=0.9, hard=True, reinMax=True)
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

    hard_code, _ = dvae.encode(x, tau=0.9, hard=True, reinMax=True)
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
    hard_code_low_temp, _ = dvae.encode(x, tau=0.01, hard=True, reinMax=True)
    assert torch.allclose(hard_code_low_temp.sum(dim=-1), torch.ones_like(hard_code_low_temp.sum(dim=-1))), \
        "hard_code should be a valid one-hot encoding even with low temperature."

    # Test with very high temperature
    hard_code_high_temp, _ = dvae.encode(x, tau=10.0, hard=True, reinMax=True)
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
    hard_code1, _ = dvae.encode(x, tau=0.9, hard=True, reinMax=True)

    torch.manual_seed(43)
    hard_code2, _ = dvae.encode(x, tau=0.9, hard=True, reinMax=True)

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
    hard_code1, _ = dvae.encode(x, tau=0.9, hard=True, reinMax=True)

    torch.manual_seed(42)
    hard_code2, _ = dvae.encode(x, tau=0.9, hard=True, reinMax=True)

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
        assert losses["tc_loss"] >= -1e-6, "TC loss should be close to non-negative"
        assert losses["kl_loss"] >= -1e-6, "KL loss should be close to non-negative"

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
        assert kld_losses["mi_loss"] >= -1e-6, f"MI loss too negative at temperature {temp}"
        assert kld_losses["dwkl_loss"] >= -1e-6, f"DWKL loss too negative at temperature {temp}"
        assert kld_losses["tc_loss"] >= -1e-6, f"TC loss too negative at temperature {temp}"
        assert kld_losses["kl_loss"] >= -1e-6, f"KL loss too negative at temperature {temp}"

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
    assert 'pool_sizes' in config_dict
    
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
    assert new_config.pool_sizes == config.pool_sizes

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
    assert new_config.n_dim == config.n_dim
    assert new_config.n_head == config.n_head
    assert new_config.n_layers == config.n_layers
    assert new_config.n_codes == config.n_codes
    assert new_config.n_pos == config.n_pos
    assert new_config.compression_factor == config.compression_factor
    assert new_config.pool_sizes == config.pool_sizes

@pytest.mark.parametrize("S,K", [
    (16, 8),   # compression
    (8, 16),   # expansion
    (16, 16),  # same size
])
@pytest.mark.parametrize("batch_specific_mask", [
    False,    # single mask for all batches [1, 1, S]
    True,     # batch-specific masks [B, 1, S]
])
@pytest.mark.parametrize("token_norm", [
    False,    # no token normalization
    True,     # with token normalization
])
def test_residual_projection_masking_effect(S, K, batch_specific_mask, token_norm):
    """Test that masked inputs produce identical outputs when only masked values differ."""
    B, d = 4, 64
    proj = ResidualProjection(S=S, K=K, d=d, token_norm=token_norm)
    
    # Create two identical inputs
    x1 = torch.randn(B, S, d)
    x2 = x1.clone()
    
    # Create mask [B, 1, S] or [1, 1, S] depending on batch_specific_mask
    batch_size = B if batch_specific_mask else 1
    mask = torch.rand(batch_size, 1, S) > 0.5
    
    # Modify x2 at masked positions with random values
    mask_expanded = mask.expand(B, 1, S)
    x2[~mask_expanded.squeeze(1)] = torch.randn(torch.sum(~mask_expanded).item(), d)
    
    # Apply projection with mask
    output1 = proj(x1, mask=mask)
    output2 = proj(x2, mask=mask)
    
    # Outputs should be identical since differences were only in masked positions
    assert torch.allclose(output1, output2, atol=1e-5), \
        f"Outputs differ when only masked positions are changed (S={S}, K={K}, batch_specific_mask={batch_specific_mask}, token_norm={token_norm})"

@pytest.mark.parametrize("S,K", [
    (16, 8),   # compression
    (8, 16),   # expansion
    (16, 16),  # same size
])
def test_residual_projection_normalization(S, K):
    """Test both token and feature normalization effects."""
    B, d = 4, 64
    
    # Test 1: Token normalization effect
    # Create projections with and without token normalization
    proj_with_norm = ResidualProjection(S=S, K=K, d=d, token_norm=True)
    proj_without_norm = ResidualProjection(S=S, K=K, d=d, token_norm=False)
    
    x = torch.randn(B, S, d)
    
    # Ensure outputs have correct shapes
    output_with_norm = proj_with_norm(x)
    output_without_norm = proj_without_norm(x)
    
    assert output_with_norm.shape == (B, K, d), \
        f"Unexpected output shape with token_norm: {output_with_norm.shape}"
    assert output_without_norm.shape == (B, K, d), \
        f"Unexpected output shape without token_norm: {output_without_norm.shape}"
    
    # Outputs should be different when token normalization is applied
    assert not torch.allclose(output_with_norm, output_without_norm, atol=1e-5), \
        f"Outputs are identical with and without token normalization (S={S}, K={K})"
    
    # Test 2: Feature normalization
    # Check that each feature vector has unit RMS for both variants
    for output, norm_type in [(output_with_norm, "with token_norm"), 
                             (output_without_norm, "without token_norm")]:
        feature_rms = torch.sqrt(torch.mean(output ** 2, dim=-1))
        assert torch.allclose(feature_rms, torch.ones_like(feature_rms), atol=1e-5), \
            f"Feature vectors do not have unit RMS after normalization ({norm_type}, S={S}, K={K})"

@pytest.mark.parametrize("S,K", [
    (16, 8),   # compression
    (8, 16),   # expansion
    (16, 16),  # same size
])
@pytest.mark.parametrize("mask_shape", [
    (1, 1, 16),    # valid: single mask for all batches
    (4, 1, 8),     # valid: batch-specific mask
    (1, 2, 16),    # invalid: wrong middle dimension
    (4, 2, 8),     # invalid: wrong middle dimension
    (2, 1, 8),     # invalid: batch size doesn't match
])
def test_residual_projection_mask_shape_validation(S, K, mask_shape):
    """Test that mask shape validation works correctly."""
    B, d = 4, 64
    proj = ResidualProjection(S=S, K=K, d=d)
    x = torch.randn(B, S, d)
    mask = torch.ones(mask_shape, dtype=torch.bool)  # Create boolean mask
    
    # Check if the mask shape is valid
    is_valid_shape = mask_shape in [(1, 1, S), (B, 1, S)]
    
    if is_valid_shape:
        # Should not raise an error
        try:
            proj(x, mask=mask)
        except AssertionError as e:
            pytest.fail(f"Unexpected assertion error for valid mask shape {mask_shape}: {e}")
    else:
        # Should raise an AssertionError
        with pytest.raises(AssertionError) as excinfo:
            proj(x, mask=mask)
        assert "Expected attn_mask shape" in str(excinfo.value), \
            f"Expected assertion error for invalid mask shape {mask_shape}"
