from dataclasses import asdict, dataclass
import math
import torch
from torch.autograd import Function
from torch.nn import functional as F
from torch import Tensor, nn
from typing import List, Optional, Tuple, Dict, Any
from torch.amp import autocast
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans




@dataclass
class Config:
    pass

def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


class AbsolutePositionalEmbedding(nn.Module):
    """
    Implements Absolute Positional Embeddings as described in "Attention is All You Need".
    
    Args:
        d_model (int): Embedding dimension.
        max_seq_len (int): Maximum sequence length.
    """
    def __init__(self, d_model, max_seq_len=1024):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # Use sequential positions if not provided
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.emb(positions)
        return x + pos_emb


class RotaryPositionalEmbeddings(nn.Module):
    """
    Implements Rotary Positional Embeddings (RoPE) as described in https://arxiv.org/abs/2104.09864.

    Args:
        dim (int): Embedding dimension per head.
        max_seq_len (int): Maximum sequence length.
        base (int): Base for geometric progression in angle computation.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 1024,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=self._device()).float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def _device(self):
        # Helper method to get device of theta if it exists
        return self.theta.device if hasattr(self, 'theta') else 'cpu'

    @autocast('cuda', enabled=False)
    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    @autocast('cuda', enabled=False)
    def forward(self, x: Tensor, input_pos: Tensor) -> Tensor:
        """
        Applies RoPE to the input tensor.

        Args:
            x (Tensor): Input tensor of shape [B, H, S, D].
            input_pos (Tensor): Position indices of shape [B, 1, S] or [B, H, S].

        Returns:
            Tensor: Tensor with RoPE applied, shape [B, H, S, D].
        """
        batch_size, n_heads, seq_len, head_dim = x.shape

        # Check if input_pos has shape [B, 1, S] and broadcast to [B, H, S]
        if input_pos.shape == (batch_size, 1, seq_len):
            input_pos = input_pos.expand(batch_size, n_heads, seq_len)  # Broadcast to [B, H, S]

        assert input_pos.shape == (batch_size, n_heads, seq_len), \
            f"Expected input_pos shape {(batch_size, n_heads, seq_len)}, got {input_pos.shape}"

        mask = input_pos >= 0  # Shape: [B, H, S]
        input_pos_clipped = input_pos.clamp(min=0)  # Shape: [B, H, S]

        # Index the cache with input_pos_clipped
        rope_cache = self.cache[input_pos_clipped]  # Shape: [B, H, S, D//2, 2]

        # Reshape x for RoPE application
        xshaped = x.float().reshape(batch_size, n_heads, seq_len, -1, 2)  # [B, H, S, D//2, 2]

        # Apply RoPE rotations
        x_rope_applied = torch.stack([
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ], dim=-1)  # [B, H, S, D//2, 2]

        # Use mask to decide where to apply RoPE
        mask = mask.unsqueeze(-1).unsqueeze(-1)  # [B, H, S, 1, 1]
        x_out = torch.where(mask, x_rope_applied, xshaped)  # [B, H, S, D//2, 2]

        # Reshape back to original dimensions
        x_out = x_out.flatten(-2)  # [B, H, S, D]

        return x_out.type_as(x)


class RoPE2D(nn.Module):
    """
    Implements 2D Rotary Positional Embeddings by applying separate RoPE modules to each positional dimension
    and concatenating the results.

    Args:
        dim (int): Total embedding dimension (must be even).
        max_height (int): Maximum expected value for the first positional dimension.
        max_width (int): Maximum expected value for the second positional dimension.
        base_height (int): Base for geometric progression in angle computation for the height dimension.
        base_width (int): Base for geometric progression in angle computation for the width dimension.
    """

    def __init__(
        self,
        dim: int,
        max_height: int = 1024,
        max_width: int = 1024,
        base_height: int = 10_000,
        base_width: int = 10_000,
    ) -> None:
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension 'dim' must be even to split for 2D RoPE."

        self.dim = dim
        self.half_dim = dim // 2

        # Initialize two RotaryPositionalEmbeddings for each positional dimension with separate bases
        self.rope_height = RotaryPositionalEmbeddings(
            dim=self.half_dim,
            max_seq_len=max_height,
            base=base_height
        )
        self.rope_width = RotaryPositionalEmbeddings(
            dim=self.half_dim,
            max_seq_len=max_width,
            base=base_width
        )

    @autocast('cuda', enabled=False)
    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """
        Applies 2D RoPE to the input tensor.

        Args:
            x (Tensor): Input tensor of shape [B, H, S, D].
            positions (Tensor): Position indices of shape [B, 1, S, 2] or [B, H, S, 2].

        Returns:
            Tensor: Tensor with 2D RoPE applied, shape [B, H, S, D].
        """
        B, H, S, D = x.shape
        assert D == self.dim, f"Expected embedding dimension {self.dim}, got {D}."

        # Check if positions has shape [B, 1, S, 2] and broadcast to [B, H, S, 2]
        if positions.shape == (B, 1, S, 2):
            positions = positions.expand(B, H, S, 2)  # Broadcast to [B, H, S, 2]

        assert positions.shape == (B, H, S, 2), f"Expected positions shape [B, H, S, 2], got {positions.shape}."

        # Split the embeddings into two halves
        x_height = x[..., :self.half_dim]  # [B, H, S, D/2]
        x_width = x[..., self.half_dim:]   # [B, H, S, D/2]

        # Split the positions into two separate tensors
        pos_height = positions[..., 0]  # [B, H, S]
        pos_width = positions[..., 1]   # [B, H, S]

        # Apply RoPE to each half
        x_height_rope = self.rope_height(x_height, pos_height)  # [B, H, S, D/2]
        x_width_rope = self.rope_width(x_width, pos_width)     # [B, H, S, D/2]

        # Concatenate the two halves back together
        x_rope2d = torch.cat([x_height_rope, x_width_rope], dim=-1)  # [B, H, S, D]

        return x_rope2d


class SwiGLUFFN(nn.Module):
    """SwiGLUFFN

    Taken from: https://github.com/kyegomez/zeta/tree/master
    Args:
        nn (_type_): _description_

    Examples:
    >>> import torch
    >>> x = torch.randn(5, 10)
    >>> swiglu = SwiGLUFFN(10, 20)
    >>> swiglu(x).shape
    torch.Size([5, 10])
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # Mark output projection for special scaling
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w2.RESCALE_INIT = True


    def forward(self, x):
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return x


class RMSNorm(nn.Module):
    """
    Ref Source: https://pytorch.org/torchtune/stable/_modules/torchtune/modules/rms_norm.html#RMSNorm
    Implements Root Mean Square Normalization introduced in
    https://arxiv.org/abs/1910.07467.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py

    Args:
        dim (int): embedding size
        eps (float): small value to avoid division by zero. Default: 1e-6
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply RMSNorm with higher precision.

        Args:
            x (Tensor): input tensor to normalize

        Returns:
            Tensor: The output tensor after applying RMSNorm.
        """
        with autocast(device_type='cuda', enabled=False):
            # Computation is in fp32
            x_fp32 = x.float()
            x_normed = (
                x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
            )
            # Apply scale parameter
            x_normed = x_normed * self.scale
            # Convert back to original dtype
            return x_normed.type_as(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_dim, n_head, dropout, rope=None, resample_positions=False):
        super().__init__()
        self.rope = rope
        self.n_dim = n_dim
        self.n_head = n_head
        self.dropout = dropout
        self.resample_positions = resample_positions
        assert n_dim % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.q_proj = nn.Linear(n_dim, n_dim, bias=False)
        self.k_proj = nn.Linear(n_dim, n_dim, bias=False)
        self.v_proj = nn.Linear(n_dim, n_dim, bias=False)
        # output projection
        self.c_proj = nn.Linear(n_dim, n_dim, bias=False)
        self.c_proj.RESCALE_INIT = True

    def get_resampled_positions(self, positions: Tensor, target_length: int) -> Tensor:
        """
        Given a positions tensor of shape [B, L] and a target length, 
        returns a tensor of shape [B, target_length] that samples positions evenly.
        """
        B, L = positions.shape
        # Create indices evenly spaced between 0 and L-1.
        new_indices = torch.linspace(0, L - 1, steps=target_length, device=positions.device)
        # Round them to the nearest integer and convert to long.
        new_indices = new_indices.round().long()
        # Clamp to ensure no index is out-of-bounds.
        new_indices = new_indices.clamp(0, L - 1)
        # Use advanced indexing to sample.
        return positions[:, new_indices]

    def forward(self,
            q: Tensor, 
            k: Tensor,
            v: Tensor,
            positions: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
            kv_cache: Optional[Tuple[Tensor, Tensor]] = None, 
            return_kv_cache: bool = False) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Computes multi-head self-attention on the input tensor.

        Args:
            x (Tensor): The input tensor to compute self-attention on.
            attn_mask (Optional[Tensor]): The attention mask to apply to the attention weights. Expected shape is [B, 1, S] or [1, 1, S], where B is the batch size and S is the sequence length.
            positions (Optional[Tensor]): The positions to use for RoPE encoding.
            kv_cache (Optional[Tuple[Tensor, Tensor]]): The cached key and value tensors.
            return_kv_cache (bool): Whether to return the updated key and value cache.

        Returns:
            Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]: The output tensor after self-attention and optionally the updated key and value cache.
        """
        qB, qT, qD = q.size()
        kB, kT, kD = k.size()
        vB, vT, vD = v.size()

        assert qB == kB == vB, "Batch size mismatch"
        assert kT == vT, "Sequence length mismatch"
        assert qD == kD == vD, "Dimension mismatch"

        B = qB
        D = qD

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Reshape for multi-head attention, but do not transpose yet!
        q = q.view(B, qT, self.n_head, D // self.n_head).transpose(1, 2) # (B, n_head, T, head_dim)
        k = k.view(B, kT, self.n_head, D // self.n_head).transpose(1, 2) 
        v = v.view(B, vT, self.n_head, D // self.n_head).transpose(1, 2)  

        # Apply Rope2D to q and k
        if self.rope is not None and positions is not None:
            if self.resample_positions:
                # Resample positions to match the current key and query lengths.
                key_positions = self.get_resampled_positions(positions, kT)  # shape: [B, 1, kT]
                query_positions = self.get_resampled_positions(positions, qT) # shape: [B, 1, qT]
            else:
                key_positions = positions[:, :kT]
                query_positions = positions[:, :qT]
            
            k = self.rope(k, key_positions.unsqueeze(1))
            q = self.rope(q, query_positions.unsqueeze(1))

        # If kv_cache is present, concatenate past keys and values
        if kv_cache is not None and torch.jit.isinstance(kv_cache, Tuple[Tensor, Tensor]):
            past_k, past_v = kv_cache  # K: (B, n_head, T_past, head_dim)
            k = torch.cat([past_k, k], dim=2)  # Concatenate along sequence length dimension
            v = torch.cat([past_v, v], dim=2)

        # Update new_kv_cache
        new_kv_cache: Optional[Tuple[Tensor, Tensor]] = (k, v) if return_kv_cache else None

        # Compute attention
        dropout_p = self.dropout if self.training else 0.0

        # Ensure attn_mask is broadcastable to [B, n_head, T, T]
        if attn_mask is not None:
            assert attn_mask.dim() == 3, "Attention Mask must be 3D"
            assert attn_mask.size(0) == B, "Attention Mask Batch size mismatch"
            assert attn_mask.size(-1) == kT, "Attention Mask Sequence length mismatch"
            assert attn_mask.size(-2) == qT, "Attention Mask Sequence length mismatch"
            attn_mask = attn_mask.unsqueeze(1)  # Expand to [B, 1, S, S]

        # attn_output: (B, n_head, T, head_dim)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)

        # Reshape back to (B, T, C)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, qT, D)

        # Output projection
        y = self.c_proj(attn_output)

        # Zero out NaN values, so they don't affect future computations
        # I have also verified that the it doesn't matter what the nan values are set to
        # y = torch.nan_to_num(y, nan=0.0)

        return y, new_kv_cache


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, rope=None, dim_feedforward=None, dropout=0.0, resample_positions: bool = False):
        super().__init__()

        self.mha = MultiHeadAttention(n_dim=d_model, n_head=n_head, dropout=dropout, rope=rope, resample_positions=resample_positions)
        self.dropout = nn.Dropout(dropout)
        self.norm_context = RMSNorm(dim=d_model)
        self.norm_queries = RMSNorm(dim=d_model)
        self.ff = SwiGLUFFN(dim=d_model, hidden_dim=dim_feedforward if dim_feedforward is not None else 4*d_model)
        self.norm_ff = RMSNorm(dim=d_model)

    def forward(self, queries, context, positions: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None, kv_cache: Optional[Tuple[Tensor, Tensor]] = None, 
            return_kv_cache: bool = False):
        """
        queries: (B, Tq, D)
        context: (B, Tc, D)
        Returns: (B, Tq, D)
        """
        normed_context = self.norm_context(context)
        normed_queries = self.norm_queries(queries)

        # Multi-head attention
        attn_out, new_kv_cache = self.mha(q=normed_queries, k=normed_context, v=normed_context, positions=positions, attn_mask=attn_mask, kv_cache=kv_cache, return_kv_cache=return_kv_cache)  # shape (B, Tq, D)
        queries = queries + self.dropout(attn_out)

        # Feed-forward
        queries = queries + self.dropout(self.ff(self.norm_ff(queries)))  # shape (B, Tq, D)
        return queries, new_kv_cache

    
class Transformer(nn.Module):
    def __init__(self, d_model, n_head, n_layer, dim_feedforward=None, rope=None, out_norm=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_head=n_head, dim_feedforward=dim_feedforward, rope=rope, resample_positions=False)
            for _ in range(n_layer)
        ])

        self.rms_out = RMSNorm(d_model) if out_norm else nn.Identity()

    def forward(self, x, positions=None, attn_mask=None, 
                kv_cache: Optional[List[Tuple[Tensor, Tensor]]] = None,
                return_kv_caches: bool = False):
        """
        x: (B, S, d_model) = input embeddings for S=1024 tokens
        Return: (B, n_latent, d_model)
        """

        loop_kv_caches: List[Tuple[Tensor, Tensor]] = []

        for i, block in enumerate(self.blocks):
            x, new_kv_cache = block(queries=x,
                        context=x, 
                        positions=positions, 
                        attn_mask=attn_mask,
                        kv_cache=kv_cache[i] if kv_cache is not None else None,
                        return_kv_cache=return_kv_caches)
            
            # Ensure new_kv_cache is not None before appending
            if return_kv_caches and new_kv_cache is not None:
                loop_kv_caches.append(new_kv_cache)

        x = self.rms_out(x)
        return x, loop_kv_caches


class LatentTransformer(nn.Module):
    def __init__(self,
                 n_latent,
                 d_model,
                 n_head,
                 n_layer,
                 dim_feedforward=None,
                 rope=None,
                 out_norm=True):
        super().__init__()
        # Learnable latent tokens: shape (1, n_latent, d_model)
        self.latent_tokens = nn.Parameter(torch.randn(1, n_latent, d_model))
        torch.nn.init.normal_(self.latent_tokens, mean=0.0, std=0.02)
        
        # Stack of cross-attn blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_head=n_head, dim_feedforward=dim_feedforward, 
                             rope=rope, resample_positions=True)
            for _ in range(n_layer)
        ])

        self.rms_norm = RMSNorm(d_model) if out_norm else nn.Identity()

    def forward(self, x, positions=None, attn_mask=None, 
                kv_cache: Optional[List[Tuple[Tensor, Tensor]]] = None,
                return_kv_caches: bool = False):
        """
        x: (B, S, d_model) = input embeddings for S=1024 tokens
        Return: (B, n_latent, d_model)
        """
        B = x.size(0)

        # Broadcast the learnable latents to match batch size
        latents = self.latent_tokens.expand(B, -1, -1)  # (B, n_latent, d_model)

        loop_kv_caches: List[Tuple[Tensor, Tensor]] = []

        for i, block in enumerate(self.blocks):
            latents, new_kv_cache = block(queries=latents,
                                    context=x, 
                                    positions=positions, 
                                    attn_mask=attn_mask, 
                                    kv_cache=kv_cache[i] if kv_cache is not None else None, 
                                    return_kv_cache=return_kv_caches)
            
            if return_kv_caches and new_kv_cache is not None:
                loop_kv_caches.append(new_kv_cache)

        latents = self.rms_norm(latents)
        return latents, loop_kv_caches


@dataclass
class LatentAutoEncoderConfig(Config):
    r"""
    Configuration class for LatentAutoEncoder model.
    
    Attributes:
        n_dim (int): Model dimension
        n_head (int): Number of attention heads
        n_grid_layer (int): Number of base transformer layers applied to the grid representations
        n_latent_layer (int): Number of transformer layers used for latent space processing
        n_latent (int): Number of latent tokens in the compressed representation
        dropout (float): Dropout probability
        grid_height (int): Grid height for input/output
        grid_width (int): Grid width for input/output
        n_vocab (int): Vocabulary size
        padding_idx (int | None): Index for padding token (default: n_vocab - 1)
        mask_idx (int | None): Index for masking token (default: n_vocab - 2)
        use_rope (bool): Whether to use Rotary Positional Embeddings
        rope_base (int): Base for RoPE's geometric progression
    """
    n_dim: int = 256
    n_head: int = 4
    n_grid_layer: int = 1
    n_latent_layer: int = 2
    n_latent: int = 128
    dropout: float = 0.0
    grid_height: int = 32
    grid_width: int = 32
    n_vocab: int = 16
    padding_idx: int | None = None
    mask_idx: int | None = None
    use_rope: bool = True
    rope_base: int = 10000  # ~10k, prime

    def __post_init__(self):
        if self.n_dim % self.n_head != 0:
            raise ValueError("n_dim must be divisible by n_head")
        
        C = self.n_dim // self.n_head
        assert C % 2 == 0, "n_dim // n_head must be divisible by 2"

        head_dim = C // 2  # Actual Head Dimension
        assert head_dim % 2 == 0, "Head dimension must be even"

        # Calculate n_pos from grid dimensions
        self.n_pos = self.grid_height * self.grid_width
        
        # Set default padding_idx and eos_idx if not provided
        if self.padding_idx is None:
            self.padding_idx = self.n_vocab - 1
        if self.mask_idx is None:
            self.mask_idx = self.n_vocab - 2

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)

class LatentAutoEncoder(nn.Module):
    def __init__(self, config: LatentAutoEncoderConfig):
        super().__init__()
        self.config = config
        self.n_pos = config.n_pos
        self.embd = nn.Embedding(config.n_vocab, config.n_dim)
        self.pad_value = config.padding_idx
        self.mask_value = config.mask_idx
        nn.init.normal_(self.embd.weight, mean=0.0, std=0.02)
        self.grid_height = config.grid_height
        self.grid_width = config.grid_width
        
        # Initialize RoPE if enabled
        rope_2d, rope_1d = None, None
        if config.use_rope:
            rope_2d = RoPE2D(
                    dim=config.n_dim // config.n_head,
                    max_height=self.grid_height,
                    max_width=self.grid_width,
                    base_height=config.rope_base,
                    base_width=config.rope_base)
            
            rope_1d = RotaryPositionalEmbeddings(
                dim=config.n_dim // config.n_head,
                max_seq_len=config.n_pos,
                base=config.rope_base
            )
        
        self.grid_ape = AbsolutePositionalEmbedding(
            d_model=config.n_dim,
            max_seq_len=self.grid_height * self.grid_width
        )
        
        self.grid_encoder = Transformer(
            d_model=config.n_dim,
            n_head=config.n_head,
            n_layer=config.n_grid_layer,
            out_norm=False,
            rope=rope_2d
        )

        self.latent_encoder = LatentTransformer(
            n_latent=config.n_latent,
            d_model=config.n_dim,
            n_head=config.n_head,
            n_layer=config.n_latent_layer,
            out_norm=False,
            rope=rope_1d
        )

        self.latent_decoder = LatentTransformer(
            n_latent=config.n_pos,
            d_model=config.n_dim,
            n_head=config.n_head,
            n_layer=config.n_latent_layer,
            out_norm=False,
            rope=rope_1d
        )

        self.grid_decoder = Transformer(
            d_model=config.n_dim,
            n_head=config.n_head,
            n_layer=config.n_grid_layer,
            out_norm=True,
            rope=rope_2d
        )

        self.decoder_head = nn.Linear(config.n_dim, config.n_vocab, bias=False)

        rows = torch.arange(config.grid_height, dtype=torch.long)
        cols = torch.arange(config.grid_width, dtype=torch.long)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
        grid_pos_indices = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).unsqueeze(0)
        latent_pos_indices = torch.arange(config.n_pos).unsqueeze(0)

        self.register_buffer("latent_pos_indices", latent_pos_indices, persistent=False)
        self.register_buffer('grid_pos_indices', grid_pos_indices, persistent=False)

        # Apply weight initialization on registered modules.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights like in Llama, but preserve VQEmbedding initialization."""
        if isinstance(module, nn.Linear):
            # For most linear layers, use standard normal initialization
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Skip VQEmbedding modules and the codebook specifically
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.scale)

    def encode(self, x: Tensor) -> Tensor:
        # Assume x is [B, H, W]
        B = x.size(0)
        x = x.view(B, self.grid_height * self.grid_width)
        x_embd = self.embd(x)  # [B, S, n_dim]
        grid_pos_indices = self.grid_pos_indices.expand(B, -1, -1)
        latent_pos_indices = self.latent_pos_indices.expand(B, -1)
        x_embd = self.grid_ape(x_embd)
        grid_encoded, _ = self.grid_encoder(x_embd, positions=grid_pos_indices)
        latent_encoded, _ = self.latent_encoder(grid_encoded, positions=latent_pos_indices)
        
        return latent_encoded


    def decode(self, x: Tensor) -> Tensor:
        B = x.size(0)
        grid_pos_indices = self.grid_pos_indices.expand(B, -1, -1)
        latent_pos_indices = self.latent_pos_indices.expand(B, -1)  
        # Pass through latent decoder
        latent_decoded, _ = self.latent_decoder(x, positions=latent_pos_indices)
        
        # No need for another APE before grid decoder
        grid_decoded, _ = self.grid_decoder(latent_decoded, positions=grid_pos_indices)
        
        grid_decoded_logits = self.decoder_head(grid_decoded)

        grid_decoded_logits = grid_decoded_logits.view(B, self.grid_height, self.grid_width, -1)
        return grid_decoded_logits
    
    def forward(self, x: Tensor) -> Tensor:
        z_e_x = self.encode(x) # [B, n_latent, n_dim]
        decoded_logits = self.decode(z_e_x)
        return decoded_logits


# For running model tests
if __name__ == "__main__":
    import torch
    # Create configuration
    config = LatentAutoEncoderConfig(
        n_dim=256,
        n_head=4,
        n_grid_layer=1,
        n_latent_layer=2,
        n_latent=128,
        dropout=0.0,
        grid_height=32,
        grid_width=32,
        n_vocab=16,
        use_rope=True
    )
    
    print(f"Initializing model with config: {config}")
    
    # Initialize model
    model = LatentAutoEncoder(config)
    
    # Generate dummy input data
    batch_size = 4
    # Generate random indices within vocab range (0 to n_vocab-1)
    dummy_input = torch.randint(
        low=0,
        high=config.n_vocab,  # Avoid using padding_idx and mask_idx
        size=(batch_size, config.grid_height, config.grid_width),
        dtype=torch.long
    )
    
    print(f"Dummy input shape: {dummy_input.shape}")
    
    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        output_logits = model(dummy_input)
    
    print(f"Output logits shape: {output_logits.shape}")
    
    # Compute predicted tokens
    predicted_tokens = output_logits.argmax(dim=-1)
    print(f"Predicted tokens shape: {predicted_tokens.shape}")
    
    # Check accuracy (just for demonstration)
    accuracy = (predicted_tokens == dummy_input).float().mean().item()
    print(f"Random initialization accuracy: {accuracy:.4f} (expected to be low)")
    
    # Demonstrate encoding and decoding separately
    with torch.no_grad():
        # Encode input to latent space
        encoded = model.encode(dummy_input)
        print(f"Encoded latent representation shape: {encoded.shape}")
        
        # Decode latent back to tokens
        decoded_logits = model.decode(encoded)
        print(f"Decoded logits shape: {decoded_logits.shape}")
        
    print("Model test completed successfully!")


