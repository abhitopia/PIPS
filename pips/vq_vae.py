from dataclasses import dataclass
import math
import torch
from torch.autograd import Function
from torch.nn import functional as F
from torch import Tensor, nn
from typing import List, Optional, Tuple
from torch.amp import autocast
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans




@dataclass
class Config:
    pass

def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


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


class MultiHeadAttentionWithEntropy(nn.Module):
    def __init__(self, n_dim, n_head, dropout, rope=None):
        super().__init__()
        self.rope = rope
        self.n_dim = n_dim
        self.n_head = n_head
        self.dropout = dropout
        assert n_dim % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.q_proj = nn.Linear(n_dim, n_dim, bias=False)
        self.k_proj = nn.Linear(n_dim, n_dim, bias=False)
        self.v_proj = nn.Linear(n_dim, n_dim, bias=False)
        # output projection
        self.c_proj = nn.Linear(n_dim, n_dim, bias=False)
        self.c_proj.RESCALE_INIT = True
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor, positions: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None, kv_cache: Optional[Tuple[Tensor, Tensor]] = None, return_kv_cache: bool = False) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Computes multi-head self-attention on the input tensor using vanilla PyTorch operations.
        This is functionally equivalent to the regular forward method but without using F.scaled_dot_product_attention.

        Args:
            q (Tensor): Query tensor of shape [B, Tq, D].
            k (Tensor): Key tensor of shape [B, Tk, D].
            v (Tensor): Value tensor of shape [B, Tv, D].
            positions (Optional[Tensor]): The positions to use for RoPE encoding.
            attn_mask (Optional[Tensor]): The attention mask to apply. Expected shape [B, Tq, Tk].
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
        head_dim = D // self.n_head

        # Apply projection layers
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Reshape for multi-head attention, but do not transpose yet!
        q = q.view(B, qT, self.n_head, head_dim).transpose(1, 2)  # (B, n_head, Tq, head_dim)
        k = k.view(B, kT, self.n_head, head_dim).transpose(1, 2)  # (B, n_head, Tk, head_dim)
        v = v.view(B, vT, self.n_head, head_dim).transpose(1, 2)  # (B, n_head, Tv, head_dim)

        # Apply RoPE2D to q and k
        if self.rope is not None and positions is not None:
            k = self.rope(k, positions[:, :kT].unsqueeze(1))
            q = self.rope(q, positions[:, :qT].unsqueeze(1))

        # If kv_cache is present, concatenate past keys and values
        if kv_cache is not None and torch.jit.isinstance(kv_cache, Tuple[Tensor, Tensor]):
            past_k, past_v = kv_cache  # K: (B, n_head, T_past, head_dim)
            k = torch.cat([past_k, k], dim=2)  # Concatenate along sequence length dimension
            v = torch.cat([past_v, v], dim=2)

        # Update new_kv_cache
        new_kv_cache: Optional[Tuple[Tensor, Tensor]] = (k, v) if return_kv_cache else None

        # --------- VANILLA ATTENTION IMPLEMENTATION ---------
        # Calculate attention scores
        # q: [B, n_head, Tq, head_dim]
        # k: [B, n_head, Tk, head_dim]
        # scores: [B, n_head, Tq, Tk]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            # Expand attn_mask to match scores dimensions: [B, 1, Tq, Tk] -> [B, n_head, Tq, Tk]
            expanded_mask = attn_mask.unsqueeze(1)
            scores = scores.masked_fill(expanded_mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Calculate entropy of attention distribution if requested
        with autocast(device_type='cuda', enabled=False):
            # Add a small epsilon to avoid log(0)
            eps = 1e-10
            # Cast to float32 for precise entropy calculation
            attn_weights_fp32 = attn_weights.float()
            # Entropy calculation: -sum(p * log(p)) for each attention distribution
            entropy = -torch.sum(
                attn_weights_fp32 * torch.log(attn_weights_fp32 + eps), 
                dim=-1
            )
            entropy = entropy.mean()
        
        # Apply dropout if training
        if self.training and self.dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # Apply attention weights to values
        # attn_weights: [B, n_head, Tq, Tk]
        # v: [B, n_head, Tv, head_dim]
        # attn_output: [B, n_head, Tq, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to [B, Tq, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, qT, D)
        
        # Output projection
        y = self.c_proj(attn_output)
        
        return y, new_kv_cache, entropy


class MultiHeadAttention(nn.Module):
    def __init__(self, n_dim, n_head, dropout, rope=None):
        super().__init__()
        self.rope = rope
        self.n_dim = n_dim
        self.n_head = n_head
        self.dropout = dropout
        assert n_dim % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.q_proj = nn.Linear(n_dim, n_dim, bias=False)
        self.k_proj = nn.Linear(n_dim, n_dim, bias=False)
        self.v_proj = nn.Linear(n_dim, n_dim, bias=False)
        # output projection
        self.c_proj = nn.Linear(n_dim, n_dim, bias=False)
        self.c_proj.RESCALE_INIT = True

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
            k = self.rope(k, positions[:, :kT].unsqueeze(1))
            q = self.rope(q, positions[:, :qT].unsqueeze(1))

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
    def __init__(self, d_model, n_head, rope=None, dim_feedforward=None, dropout=0.0, return_entropy: bool = False):
        super().__init__()
        if return_entropy:
            self.mha = MultiHeadAttentionWithEntropy(n_dim=d_model, n_head=n_head, dropout=dropout, rope=rope)
        else:
            self.mha = MultiHeadAttention(n_dim=d_model, n_head=n_head, dropout=dropout, rope=rope)
        self.dropout = nn.Dropout(dropout)

        self.return_entropy = return_entropy
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
        if self.return_entropy:
            attn_out, new_kv_cache, attn_entropy = self.mha(q=normed_queries, k=normed_context, v=normed_context, positions=positions, attn_mask=attn_mask, kv_cache=kv_cache, return_kv_cache=return_kv_cache)  # shape (B, Tq, D)
        else:
            attn_entropy = torch.tensor(0.0, device=queries.device)
            attn_out, new_kv_cache = self.mha(q=normed_queries, k=normed_context, v=normed_context, positions=positions, attn_mask=attn_mask, kv_cache=kv_cache, return_kv_cache=return_kv_cache)  # shape (B, Tq, D)
        queries = queries + self.dropout(attn_out)

        # Feed-forward
        queries = queries + self.dropout(self.ff(self.norm_ff(queries)))  # shape (B, Tq, D)
        return queries, new_kv_cache, attn_entropy

    
class Transformer(nn.Module):
    def __init__(self, d_model, n_head, n_layer, dim_feedforward=None, rope=None, out_norm=True, return_attn_entropy: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_head=n_head, dim_feedforward=dim_feedforward, rope=rope, return_entropy=return_attn_entropy)
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

        total_attn_entropy = torch.tensor(0.0, device=x.device)

        for i, block in enumerate(self.blocks):
            x, new_kv_cache, attn_entropy = block(queries=x,
                        context=x, 
                        positions=positions, 
                        attn_mask=attn_mask,
                        kv_cache=kv_cache[i] if kv_cache is not None else None,
                        return_kv_cache=return_kv_caches)
            
            total_attn_entropy += attn_entropy
            
            # Ensure new_kv_cache is not None before appending
            if return_kv_caches and new_kv_cache is not None:
                loop_kv_caches.append(new_kv_cache)

        x = self.rms_out(x)
        return x, loop_kv_caches, total_attn_entropy


class LatentTransformer(nn.Module):
    def __init__(self,
                 n_latent,
                 d_model,
                 n_head,
                 n_layer,
                 dim_feedforward=None,
                 rope=None,
                 out_norm=True,
                 return_attn_entropy: bool = False):
        super().__init__()
        # Learnable latent tokens: shape (1, n_latent, d_model)
        self.latent_tokens = nn.Parameter(torch.randn(1, n_latent, d_model))
        torch.nn.init.normal_(self.latent_tokens, mean=0.0, std=0.02)
        
        # Stack of cross-attn blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_head=n_head, dim_feedforward=dim_feedforward, rope=rope, return_entropy=return_attn_entropy)
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
        total_attn_entropy = torch.tensor(0.0, device=x.device)

        for i, block in enumerate(self.blocks):
            latents, new_kv_cache, attn_entropy = block(queries=latents,
                                    context=x, 
                                    positions=positions, 
                                    attn_mask=attn_mask, 
                                    kv_cache=kv_cache[i] if kv_cache is not None else None, 
                                    return_kv_cache=return_kv_caches)
            
            total_attn_entropy += attn_entropy
            if return_kv_caches and new_kv_cache is not None:
                loop_kv_caches.append(new_kv_cache)

        latents = self.rms_norm(latents)
        return latents, loop_kv_caches, total_attn_entropy



# Reference: https://github.com/Vrushank264/VQVAE-PyTorch/tree/main

class VectorQuantization(Function):
    """
    Custom autograd Function for vector quantization in a VQ-VAE.

    This function maps each input vector (of shape [B, N, C]) to the index of its closest
    embedding in the codebook (of shape [K, C]) using a squared L2 distance.
    
    Now, in addition to returning the indices, it also returns the corresponding minimum
    squared L2 distances (quantization errors) for each input vector.
    
    Expected input shape: [B, N, C]
      B = Batch size
      N = Number of tokens (or discrete codes)
      C = Embedding dimension
    """
    @staticmethod
    def forward(ctx, inputs, codebook):
        # Disable autocast to ensure full precision for distance calculations
        with autocast(device_type='cuda', enabled=False):
            # Cast to float32 explicitly for precise distance calculation
            inputs_fp32 = inputs.float()
            codebook_fp32 = codebook.float()
            
            # Expected shape: inputs is [B, N, C]
            B, N, C = inputs.shape

            # Flatten inputs to shape [B*N, C]
            flat_input = inputs_fp32.reshape(B * N, C)

            # Compute squared L2 norm of each codebook vector. Shape: [K]
            codebook_sq = torch.sum(codebook_fp32 * codebook_fp32, dim=1)

            # Compute squared L2 norm of each input vector. Shape: [B*N, 1]
            inputs_sq = torch.sum(flat_input * flat_input, dim=1, keepdim=True)

            # Compute squared Euclidean distance between each input and each codebook vector.
            # Using the identity: ||x - e||^2 = ||x||^2 + ||e||^2 - 2*(x · e)
            # Resulting shape: [B*N, K]
            l2_dis = torch.addmm(input=codebook_sq + inputs_sq,
                                mat1=flat_input,
                                mat2=codebook_fp32.t(),
                                alpha=-2.0, beta=1.0)

            # For each input, find the index of the codebook vector with minimum distance.
            # Also retrieve the corresponding minimum distance.
            min_vals, idx_flat = torch.min(l2_dis, dim=1)

            # Reshape indices and distances back to shape [B, N]
            idx = idx_flat.reshape(B, N)
            distances = min_vals.reshape(B, N)

        # Mark these outputs as non-differentiable.
        ctx.mark_non_differentiable(idx)
        ctx.mark_non_differentiable(distances)

        # Convert distances back to original dtype to maintain compatibility
        distances = distances.to(inputs.dtype)
        
        return idx, distances

    @staticmethod
    def backward(ctx, grad_outputs):
        raise RuntimeError("Backward pass is not defined for VectorQuantization. Use VQStraightThrough instead.")

VQ = VectorQuantization.apply


class VQStraightThrough(Function):
    """
    Custom autograd Function implementing the straight-through estimator for vector quantization.
    
    In the forward pass, it uses the updated VectorQuantization (VQ) to get the nearest codebook
    indices and the corresponding minimum distances (quantization errors). It then retrieves the 
    corresponding codebook embeddings (quantized vectors). In the backward pass, gradients are passed 
    directly (straight-through) to the encoder, while gradients for the codebook are accumulated 
    based on the quantization indices.
    
    Expected input shape: [B, N, C]
      B = Batch size
      N = Number of tokens
      C = Embedding dimension
    Codebook shape: [K, C]
    """
    @staticmethod
    def forward(ctx, inputs, codebook):
        # Since VQ now handles precision internally, we just need to ensure
        # that the subsequent operations also maintain appropriate precision
        
        # Get nearest codebook indices and corresponding quantization errors using updated VQ.
        idx, distances = VQ(inputs, codebook)  # This now runs in full precision
        
        # The following code should also be run in full precision
        with autocast(device_type='cuda', enabled=False):
            # Cast to float32 explicitly
            inputs_fp32 = inputs.float()
            codebook_fp32 = codebook.float()
            
            B, N, C = inputs.shape
            flat_idx = idx.reshape(B * N)
            
            # Save tensors for the backward pass
            ctx.save_for_backward(flat_idx, codebook_fp32)
            
            # Retrieve quantized embeddings via index selection
            codes_flat = torch.index_select(codebook_fp32, dim=0, index=flat_idx)
            codes = codes_flat.reshape(B, N, C)
            
            # Convert back to original dtype
            codes = codes.to(inputs.dtype)
        
        ctx.mark_non_differentiable(flat_idx)
        ctx.mark_non_differentiable(idx)
        ctx.mark_non_differentiable(distances)
        
        return codes, flat_idx, idx, distances

    @staticmethod
    def backward(ctx, grad_outputs, grad_flat_idx, grad_idx, grad_distances):
        grad_inputs, grad_codebook = None, None
        
        # Pass gradients straight-through to the encoder.
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.clone()
        
        # Compute gradients with respect to the codebook.
        if ctx.needs_input_grad[1]:
            flat_idx, codebook = ctx.saved_tensors
            C = codebook.shape[1]
            flat_grad_output = grad_outputs.reshape(-1, C)
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, flat_idx, flat_grad_output)
        
        return grad_inputs, grad_codebook

VQ_ST = VQStraightThrough.apply


class VQEmbedding(nn.Module):
    """
    VQEmbedding class for managing the codebook in a VQ-VAE when the encoder output
    comes from a transformer, i.e. with shape [B, N, C].

    This module creates a learnable embedding table (codebook) with K embeddings,
    each of dimension D. It provides two forward methods:
      - forward: Returns the discrete latent indices for a given encoded input.
      - straight_through_forward: Returns both the quantized vectors (using a
        straight-through estimator) and an alternative quantized representation
        obtained directly via index selection.

    Expected input shape:
      - z_e_x: (B, N, C), where:
            B = Batch size,
            N = Number of tokens (discrete codes),
            C = Embedding dimension (should equal D).
    """
    def __init__(self, K: int, D: int, decay: float = 0.99, unused_reset_threshold: float = 1.0, distance_reset: bool = False):
        """
        Initialize the VQEmbedding module.
        
        Args:
            K (int): Total number of embeddings in the codebook (codebook_size).
            D (int): Dimensionality of each embedding.
            decay (float): EMA decay rate (default: 0.99)
            unused_reset_threshold (float): Threshold below which a code is considered unused.
            distance_reset (bool): Whether to use distance-based codebook resets.
        """
        super(VQEmbedding, self).__init__()
        # Create an embedding layer (the codebook) with K embeddings of dimension D.
        self.vq_embs = nn.Embedding(K, D)
        self.decay = decay
        self.unused_reset_threshold = unused_reset_threshold
        self.distance_reset = distance_reset
        # Initialize with normal distribution
        self.vq_embs.weight.data.normal_(0, 1.0)
        
        # Scale to match expected norm from RMSNorm (approximately sqrt(D))
        expected_norm = math.sqrt(D)
        current_norm = torch.norm(self.vq_embs.weight.data, dim=1, keepdim=True).mean()
        self.vq_embs.weight.data *= (expected_norm / current_norm)
        
        # Register buffers for EMA updates
        self.register_buffer('cluster_size', torch.zeros(K), persistent=True)
        self.register_buffer('embed_sum', torch.zeros(K, D), persistent=True)
        self.register_buffer('ema_initialized', torch.tensor(0, dtype=torch.bool), persistent=True)
        
        # Register buffer for tracking codebook changes
        self.register_buffer('previous_codebook', torch.zeros(K, D), persistent=True)
        self.register_buffer('update_magnitudes', torch.zeros(K), persistent=True)
        
    def forward(self, z_e_x):
        """
        Forward pass that maps continuous encoder outputs to discrete latent indices.
        
        Args:
            z_e_x (Tensor): The continuous encoded output from the transformer.
                            Expected shape: (B, N, C), where C equals the embedding dimension D.
        
        Returns:
            Tensor: Discrete latent indices with shape (B, N), where each element is an index
                    pointing to the closest embedding in the codebook.
        """
        # For transformer output, the shape is already (B, N, C).
        # Use the vector quantization function (VQ) to get the nearest codebook indices.
        latents, _ = VQ(z_e_x, self.vq_embs.weight)  # VQ now returns (idx, distances)
        return latents
    
    def reset_unused_codes_random(self, z_e_x, normalized_embeddings, D, codebook_size):
        """
        Reset unused codebook entries based on low EMA counts.
        
        Args:
            z_e_x (Tensor): Encoder outputs, shape [B, N, D].
            normalized_embeddings (Tensor): The current normalized codebook embeddings, shape [K, D].
            D (int): Embedding dimension.
            codebook_size (int): Total number of codes (K).
            
        Returns:
            Tensor: Updated normalized_embeddings with unused entries replaced.
        """
        # Identify codebook entries with low usage.
        unused_mask = self.cluster_size < self.unused_reset_threshold  # Shape: [K]
        
        # Flatten encoder outputs to shape [B*N, D].
        flat_z_e = z_e_x.reshape(-1, D)
        
        # Sample random encoder outputs for all codebook entries.
        rand_idx = torch.randint(0, flat_z_e.shape[0], (codebook_size,), device=z_e_x.device)
        random_codes = flat_z_e[rand_idx]  # Shape: [K, D]
        
        # Replace normalized embeddings for unused entries using torch.where.
        normalized_embeddings = torch.where(
            unused_mask.unsqueeze(1),  # [K, 1]
            random_codes,
            normalized_embeddings
        )
        # Also update the EMA buffers for these entries.
        new_cluster_size = torch.where(
            unused_mask,
            torch.tensor(1.0, dtype=self.cluster_size.dtype, device=self.cluster_size.device),
            self.cluster_size
        )
        new_embed_sum = torch.where(
            unused_mask.unsqueeze(1),
            random_codes,
            self.embed_sum
        )
        self.cluster_size.copy_(new_cluster_size)
        self.embed_sum.copy_(new_embed_sum)
        return normalized_embeddings
    
    def reset_unused_codes_distance(self, z_e_x, normalized_embeddings, D, distances):
        """
        Reset unused codebook entries using quantization errors (distances) in a fully vectorized manner.
        For each codebook entry (of fixed size K), if its EMA count is below the threshold, it is updated
        with a candidate encoder output selected from those with the highest quantization error.
        
        Args:
            z_e_x (Tensor): Encoder outputs, shape [B, N, D].
            normalized_embeddings (Tensor): Normalized codebook embeddings, shape [K, D].
            D (int): Embedding dimension.
            distances (Tensor): Quantization errors, shape [B, N].
            
        Returns:
            Tensor: Updated normalized_embeddings.
        """

        with autocast(device_type='cuda', enabled=False):

            z_e_x_fp32 = z_e_x.float()
            normalized_embeddings_fp32 = normalized_embeddings.float()
            distances_fp32 = distances.float()
            # Fixed codebook size K.
            K = normalized_embeddings_fp32.shape[0]
            # Flatten encoder outputs and distances.
            flat_z_e = z_e_x_fp32.reshape(-1, D)      # [B*N, D]
            flat_dists = distances_fp32.reshape(-1)     # [B*N]
        
            # Sort encoder outputs by descending quantization error.
            sorted_indices = torch.argsort(flat_dists, descending=True)  # [B*N]
        
            # Compute a fixed-size unused mask (K is fixed).
            unused_mask = self.cluster_size < self.unused_reset_threshold  # [K] boolean
        
            # Compute a rank for each codebook entry: for unused entries, rank them in order of appearance.
            # Since K is fixed, this produces a fixed-shape tensor.
            unused_int = unused_mask.to(torch.int64)           # [K]
            ranks = torch.cumsum(unused_int, dim=0) - 1           # [K]
            # For used entries, force the rank to 0 (these values won't be used).
            ranks = torch.where(unused_mask, ranks, torch.zeros_like(ranks))
        
            # For each codebook entry i, candidate_codes[i] is taken from:
            # flat_z_e[ sorted_indices[ranks[i]] ]
            candidate_codes = flat_z_e[sorted_indices[ranks]]     # [K, D]
        
            # Update only unused entries in normalized_embeddings.
            updated_normalized_embeddings = torch.where(
                unused_mask.unsqueeze(1),  # shape [K, 1]
                candidate_codes,
                normalized_embeddings_fp32
            )
        
            # Update EMA buffers for the unused entries in a vectorized way.
            updated_cluster_size = torch.where(
                unused_mask,
                torch.tensor(1.0, dtype=self.cluster_size.dtype, device=self.cluster_size.device),
                self.cluster_size
            )
            updated_embed_sum = torch.where(
                unused_mask.unsqueeze(1),
                candidate_codes,
                self.embed_sum
            )
        
            self.cluster_size.copy_(updated_cluster_size)
            self.embed_sum.copy_(updated_embed_sum)
        
            return updated_normalized_embeddings.to(normalized_embeddings.dtype)
    
    def reset_unused_codes(self, z_e_x, normalized_embeddings, D, codebook_size, distances):
        if self.distance_reset:
            return self.reset_unused_codes_distance(z_e_x, normalized_embeddings, D, distances)
        else:
            return self.reset_unused_codes_random(z_e_x, normalized_embeddings, D, codebook_size)

    def update_codebook_ema(self, z_e_x, indices, distances):
        """
        Update codebook vectors using Exponential Moving Average (EMA).
        
        This method implements codebook updates using EMA as described in the VQ-VAE-2 paper.
        It then calls reset_unused_codes to reinitialize codes with low usage.
        
        Args:
            z_e_x (Tensor): Encoder output vectors [B, N, C]
                B = batch size
                N = number of codes per sample 
                C = embedding dimension
            indices (Tensor): Indices of nearest codebook entries [B, N]
            distances (Tensor): Quantization errors, shape [B, N].
        """
        with torch.no_grad(), autocast(device_type='cuda', enabled=False):
            # Store the current codebook for computing update magnitudes later
            self.previous_codebook.copy_(self.vq_embs.weight.data)
            
            # Get shapes
            B, N, D = z_e_x.shape  # [B, N, D]
            flat_idx = indices.reshape(B * N)  # [B*N]
            codebook_size = self.vq_embs.weight.shape[0]
            
            # Ensure same dtype for codebook and z_e_x.
            z_e_x_detached = z_e_x.detach().float()
            
            # Create one-hot encodings for indices.
            encodings = F.one_hot(flat_idx, num_classes=codebook_size).float()
            
            # Compute the new cluster size and sum of encoder outputs for each code.
            new_cluster_size = encodings.sum(0)  # [K]
            dw = encodings.t() @ z_e_x_detached.reshape(-1, D)  # [K, D]
            
            # ------------ TORCH.COMPILE FRIENDLY CODE ------------
            is_first_batch = ~self.ema_initialized  # scalar boolean tensor
            initialized_cluster_size = new_cluster_size  # [K]
            initialized_embed_sum = dw  # [K, D]
            
            updated_cluster_size = self.cluster_size.float() * self.decay + new_cluster_size * (1 - self.decay)
            updated_embed_sum = self.embed_sum.float() * self.decay + dw * (1 - self.decay)
            
            new_cluster_size = torch.where(is_first_batch, initialized_cluster_size, updated_cluster_size)
            new_embed_sum = torch.where(is_first_batch.unsqueeze(-1), initialized_embed_sum, updated_embed_sum)
            
            self.cluster_size.copy_(new_cluster_size.to(self.cluster_size.dtype))
            self.embed_sum.copy_(new_embed_sum.to(self.embed_sum.dtype))
            self.ema_initialized.copy_(torch.tensor(True, dtype=torch.bool, device=self.ema_initialized.device))
            # ------------ END TORCH.COMPILE FRIENDLY CODE ------------
            
            # Prevent division by zero.
            effective_cluster_size = self.cluster_size + 1e-3
            
            # Compute normalized embeddings.
            normalized_embeddings = (self.embed_sum / effective_cluster_size.unsqueeze(1))
            
            # Call the reset method to update unused codebook entries.
            normalized_embeddings = self.reset_unused_codes(z_e_x, normalized_embeddings, D, codebook_size, distances)
        
            # Update the codebook weights.
            self.vq_embs.weight.copy_(normalized_embeddings.to(self.vq_embs.weight.dtype))
        
            # Compute update magnitudes (L2 norm of the difference)
            updates = self.vq_embs.weight.data - self.previous_codebook
            self.update_magnitudes.copy_(torch.norm(updates, dim=1))


    def straight_through_forward(self, z_e_x, use_ema=False):
        """
        Forward pass with a straight-through estimator, optionally using EMA updates.
        
        This method quantizes the input vectors and allows gradients to flow through
        the quantization process by using the straight-through approach. It returns:
            - z_q_x: Quantized vectors with shape (B, N, C).
            - zqx_tilde: An alternative quantized representation derived directly by index selection,
                         also with shape (B, N, C).
            - idx: Quantization indices with shape (B, N).
        
        Args:
            z_e_x (Tensor): The continuous encoded output from the transformer.
                            Expected shape: (B, N, C), where C equals the embedding dimension D.
            use_ema (bool): Whether to use EMA updates for codebook.
            
        Returns:
            tuple: (z_q_x, zqx_tilde, idx)
        """
        # Input shape: (B, N, C)
        # Capture all outputs from the straight-through estimator.
        z_q_x, flat_idx, idx, distances = VQ_ST(z_e_x, self.vq_embs.weight.detach())
        
        # Retrieve alternative quantized representation using flat_idx.
        flat_zqx_tilde = torch.index_select(self.vq_embs.weight, dim=0, index=flat_idx)
        zqx_tilde = flat_zqx_tilde.view_as(z_e_x)
        
        # Optionally update the codebook via EMA if enabled.
        if use_ema and self.training:
            self.update_codebook_ema(z_e_x, idx, distances)
        
        return z_q_x, zqx_tilde, idx


@dataclass
class VQVAEConfig(Config):
    r"""
    Configuration class for VQVAE model.
    
    Attributes:
        n_dim (int): Model dimension
        n_head (int): Number of attention heads
        n_grid_layer (int): Number of base transformer layers
        n_latent_layer (int): Number of latent transformer layers
        n_codes (int): Number of discrete codes (must be power of 2)
        pos_dependent_codebook (bool): Whether to use position-dependent codebook (default: True)
        codebook_size (int): Size of codebook (default: 512)
        rope_base_height (int): Base for geometric progression in angle computation for height (default: 10007)
        rope_base_width (int): Base for geometric progression in angle computation for width (default: 5003)
        dropout (float): Dropout probability (default: 0.0)
        max_grid_height (int): Maximum grid height (default: 32)
        max_grid_width (int): Maximum grid width (default: 32)
        n_vocab (int): Vocabulary size (default: 16)
        padding_idx (int | None): Index for padding token (default: n_vocab - 1)
        mask_idx (int | None): Index for masking token (default: n_vocab - 2)
        pad_weight (float): Weight for pad token loss (default: 0.01)
        gamma (float): Focal loss gamma parameter. With \(\gamma=0\) there is no focal modulation, defaulting to 2.0.
        n_layer (int): Total number of transformer layers (grid + latent)
        skip_codebook (bool): Whether to skip the codebook and use continuous latents (default: False)
        use_ema (bool): Whether to use EMA updates for codebook (default: False)
        ema_decay (float): Decay rate for EMA updates (default: 0.99)
        unused_reset_threshold (float): Threshold below which a code is considered unused (default: 1.0)
        distance_reset (bool): Whether to use distance-based codebook resets (default: False)
    """
    n_dim: int = 256
    n_head: int = 4
    n_grid_layer: int = 2
    n_latent_layer: int = 2
    n_codes: int = 128
    codebook_size: int = 512
    rope_base_height: int = 10007  # ~10k, prime
    rope_base_width: int = 5003    # ~5k, prime
    dropout: float = 0.0
    max_grid_height: int = 32
    max_grid_width: int = 32
    n_vocab: int = 16
    padding_idx: int | None = None
    mask_idx: int | None = None
    pad_weight: float = 0.1
    gamma: float = 2.0
    skip_codebook: bool = False
    use_ema: bool = False
    ema_decay: float = 0.99
    unused_reset_threshold: float = 1.0
    distance_reset: bool = False
    return_attn_entropy: bool = False

    def __post_init__(self):
        if self.n_dim % self.n_head != 0:
            raise ValueError("n_dim must be divisible by n_head")
        
        C = self.n_dim // self.n_head
        assert C % 2 == 0, "n_dim // n_head must be divisible by 2"

        head_dim = C // 2  # Actual Head Dimension
        assert head_dim % 2 == 0, "Head dimension must be even"

        # Calculate n_pos from grid dimensions
        self.n_pos = self.max_grid_height * self.max_grid_width
        
        assert is_power_of_two(self.n_pos), "Product of max_grid_height and max_grid_width must be a power of 2"
        # assert is_power_of_two(self.n_codes), "Number of codes must be a power of 2"
        # assert self.n_pos % self.n_codes == 0, "Number of positions must be divisible by the number of codes"

        # Set default padding_idx and eos_idx if not provided
        if self.padding_idx is None:
            self.padding_idx = self.n_vocab - 1
        if self.mask_idx is None:
            self.mask_idx = self.n_vocab - 2

    def __repr__(self) -> str:
        attrs = [f"{key}={getattr(self, key)}" for key in self.__annotations__.keys()]
        computed_attrs = [
            f"n_pos={self.n_pos}",
        ]
        all_attrs = attrs + computed_attrs
        return f"DVAEConfig({', '.join(all_attrs)})"

    def to_dict(self) -> dict:
        """Convert config to a dictionary.
        
        Returns:
            dict: Dictionary containing all config values, including computed attributes
        """
        base_dict = {
            'n_dim': self.n_dim,
            'n_head': self.n_head,
            'n_grid_layer': self.n_grid_layer,
            'n_latent_layer': self.n_latent_layer,
            'n_codes': self.n_codes,
            'codebook_size': self.codebook_size,
            'rope_base_height': self.rope_base_height,
            'rope_base_width': self.rope_base_width,
            'dropout': self.dropout,
            'max_grid_height': self.max_grid_height,
            'max_grid_width': self.max_grid_width,
            'n_vocab': self.n_vocab,
            'padding_idx': self.padding_idx,
            'mask_idx': self.mask_idx,
            'pad_weight': self.pad_weight,
            'gamma': self.gamma,
            'skip_codebook': self.skip_codebook,
            'use_ema': self.use_ema,
            'ema_decay': self.ema_decay,
            'unused_reset_threshold': self.unused_reset_threshold,
            'distance_reset': self.distance_reset,
            'return_attn_entropy': self.return_attn_entropy,
        }
         
        return base_dict

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'VQVAEConfig':
        """Create config from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            VQVAEConfig: New config instance
            
        Raises:
            ValueError: If required fields are missing
        """
        # Extract only the fields that are part of the dataclass
        required_fields = cls.__annotations__.keys()
        config_kwargs = {
            key: config_dict[key] 
            for key in required_fields 
            if key in config_dict
        }
        
        # Create new instance
        return cls(**config_kwargs)

    def compute_latent_bits(self) -> float:
        """Calculate the theoretical information capacity of the discrete latent space in bits.
        
        The model encodes information using n_codes discrete variables, each with codebook_size 
        possible values. The total number of possible configurations is codebook_size^n_codes,
        which corresponds to log2(codebook_size^n_codes) = n_codes * log2(codebook_size) bits.
        
        Returns:
            float: The number of bits that can be encoded in the latent space
        """
        return self.n_codes * math.log2(self.codebook_size)


class VQVAE(nn.Module):
    def __init__(self, config: VQVAEConfig):
        super().__init__()
        self.config = config
        self.skip_codebook = config.skip_codebook
        self.gamma = config.gamma
        self.n_pos = config.n_pos
        self.embd = nn.Embedding(config.n_vocab, config.n_dim)
        self.pad_value = config.padding_idx  # Store padding value here
        self.mask_value = config.mask_idx
        nn.init.normal_(self.embd.weight, mean=0.0, std=0.02)
        

        ## The choice of out_norm is inspired by Llama. We can think of both Encoder and Decoder as Llama models.
        ## With the difference that some of the layers are replaced by TransformerProjection blocks.
        ## Like in Llama, the token embeddings flow through unnormalized until the head is applied.
        ## In my case, we normalise the final output of the encoder as well as that of decoder before applyin their
        ## respective heads. Nothing gets normalised from base to bottleneck and vice versa.

        rope_2d = RoPE2D(
                dim=config.n_dim // config.n_head,  # per-head dimension (e.g., 256//8 = 32)
                max_height=config.max_grid_height,
                max_width=config.max_grid_width,
                base_height=config.rope_base_height,
                base_width=config.rope_base_width)
        
        rope_1d = RotaryPositionalEmbeddings(
            dim=config.n_dim // config.n_head,  # 256//8 = 32, per-head dimension
            max_seq_len=config.n_pos,
            base=config.rope_base_height
        )

        
        self.grid_encoder = Transformer(
            d_model=config.n_dim,
            n_head=config.n_head,
            n_layer=config.n_grid_layer,
            out_norm=False,
            rope=rope_2d,
            return_attn_entropy=config.return_attn_entropy
        )

        self.latent_encoder = LatentTransformer(
            n_latent=config.n_codes,
            d_model=config.n_dim,
            n_head=config.n_head,
            n_layer=config.n_latent_layer,
            out_norm=False,   # This seems to work even when skipping codebook so let's keep it.
            rope=rope_1d,
            return_attn_entropy=config.return_attn_entropy
        )

        self.codebook = VQEmbedding(config.codebook_size,
                                    config.n_dim, 
                                    decay=config.ema_decay, 
                                    unused_reset_threshold=config.unused_reset_threshold,
                                    distance_reset=config.distance_reset)    

        self.latent_decoder = LatentTransformer(
            n_latent=config.n_pos,
            d_model=config.n_dim,
            n_head=config.n_head,
            n_layer=config.n_latent_layer,
            out_norm=False,
            rope=rope_1d,
            return_attn_entropy=config.return_attn_entropy
        )

        self.grid_decoder = Transformer(
            d_model=config.n_dim,
            n_head=config.n_head,
            n_layer=config.n_grid_layer,
            out_norm=True,
            rope=rope_2d,
            return_attn_entropy=config.return_attn_entropy
        )

        self.decoder_head = nn.Linear(config.n_dim, config.n_vocab, bias=False)

        rows = torch.arange(config.max_grid_height, dtype=torch.long)
        cols = torch.arange(config.max_grid_width, dtype=torch.long)
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
            # # Special scaling for the output projection after attention
            # if hasattr(module, 'RESCALE_INIT'):
            #     module.weight.data.div_(math.sqrt(2 * self.config.n_layer))
        elif isinstance(module, nn.Embedding) and not isinstance(module, VQEmbedding) and module is not self.codebook.vq_embs:
            # Skip VQEmbedding modules and the codebook specifically
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.scale)

    def encode(self, x: Tensor, grid_pos_indices: Tensor, latent_pos_indices: Tensor) -> Tensor:
        x_embd = self.embd(x) # [B, S, n_dim]
        grid_encoded, _, grid_attn_entropy = self.grid_encoder(x_embd, positions=grid_pos_indices) # [B, S, n_dim]
        latent_encoded, _, latent_attn_entropy = self.latent_encoder(grid_encoded, positions=latent_pos_indices) # [B, n_codes, n_dim]
        return latent_encoded, grid_attn_entropy + latent_attn_entropy


    def decode(self, x: Tensor, grid_pos_indices: Tensor, latent_pos_indices: Tensor) -> Tensor:    
        latent_decoded, _ , latent_attn_entropy = self.latent_decoder(x, positions=latent_pos_indices)        
        grid_decoded, _ , grid_attn_entropy = self.grid_decoder(latent_decoded, positions=grid_pos_indices)        
        grid_decoded_logits = self.decoder_head(grid_decoded)
        return grid_decoded_logits, latent_attn_entropy + grid_attn_entropy
    

    def apply_mask(self, x: Tensor, mask_percentage: Tensor) -> Tensor:
        x = x.clone()
        # Create a random tensor with values in the range [0,1) for each element in x.
        mask = (torch.rand(x.shape, device=x.device, dtype=torch.float32) < mask_percentage) & (x != self.pad_value)
        x.masked_fill_(mask, self.mask_value)
        return x
    
    def forward(self, x: Tensor, mask_percentage: Tensor = torch.tensor(0.0)) -> Tuple[Tensor, dict, Tensor]:
        B, S = x.size()
        x_masked = self.apply_mask(x, mask_percentage)
        grid_pos_indices = self.grid_pos_indices.expand(B, -1, -1)
        latent_pos_indices = self.latent_pos_indices.expand(B, -1)
    
        z_e_x, encoder_attn_entropy = self.encode(x_masked, grid_pos_indices, latent_pos_indices) # [B, n_codes, n_dim]
                
        if self.skip_codebook:
            decoded_logits, decoder_attn_entropy = self.decode(z_e_x, grid_pos_indices, latent_pos_indices)
            vq_loss = torch.tensor(0.0, device=x.device)
            commitment_loss = torch.tensor(0.0, device=x.device)
            indices = None
        else:
            # Get quantized vectors and indices
            z_q_x_st, z_q_x, indices = self.codebook.straight_through_forward(z_e_x, use_ema=self.config.use_ema) # [B, n_codes, n_dim]
            
            decoded_logits, decoder_attn_entropy = self.decode(z_q_x_st, grid_pos_indices, latent_pos_indices)
            
            # Always calculate both losses for monitoring
            vq_loss = F.mse_loss(z_q_x, z_e_x.detach())
            commitment_loss = F.mse_loss(z_e_x, z_q_x.detach())
    
        ce_loss = self.reconstruction_loss(decoded_logits, x, pad_value=self.pad_value, gamma=self.gamma)

        losses = {
             "ce_loss": ce_loss,  # Weight normalized loss
             "vq_loss": vq_loss.detach() if self.config.use_ema else vq_loss, # Detach to prevent gradients when using EMA
             "commitment_loss": commitment_loss,
             "attn_entropy_loss": encoder_attn_entropy + decoder_attn_entropy
        }
        
        # Return usage_stats as a third return value
        return decoded_logits, losses, indices


    def reconstruction_loss(self, decoded_logits: Tensor, x: Tensor, pad_value: int = -1, pad_weight: float = 1.0, gamma: float = 0.0) -> Tensor:
        """
        Compute the reconstruction loss using focal loss weighted cross-entropy per sample, with 
        padded tokens handled separately.

        When gamma == 0.0, the focal modulation factor is effectively 1 (i.e. standard cross-entropy loss).
        This avoids unnecessary computation.

        Args:
            decoded_logits (Tensor): Predicted logits of shape [B, S, V].
            x (Tensor): Target tokens of shape [B, S].
            pad_value (int): Token value for padding tokens.
            pad_weight (float): Weight for pad token loss. Valid tokens have weight 1.0.
            gamma (float): Focusing parameter for focal loss. If 0.0, no focal modulation is applied (default: 0.0).

        Returns:
            Tensor: Average reconstruction loss per sample.
        """
        # Create a weight tensor: assign pad tokens pad_weight, and valid tokens a weight of 1.0.
        weights = torch.where(
            x == pad_value,
            torch.full_like(x, pad_weight, dtype=decoded_logits.dtype),
            torch.ones_like(x, dtype=decoded_logits.dtype)
        )
        
        # Compute the standard cross-entropy loss per token without reduction.
        ce_loss = F.cross_entropy(
            decoded_logits.view(-1, decoded_logits.size(-1)),
            x.view(-1),
            reduction='none'
        )
        
        # If gamma > 0, apply focal loss modulation; otherwise, use the cross-entropy loss directly.
        if gamma > 0.0:
            # Estimate the probability for the true class: p_t ≈ exp(-ce_loss)
            p_t = torch.exp(-ce_loss)
            # Compute the focal modulation factor: (1 - p_t)^gamma.
            modulating_factor = (1 - p_t) ** gamma
            loss = modulating_factor * ce_loss
        else:
            loss = ce_loss
        
        # Compute the total loss: each token's loss is scaled by its corresponding weight.
        total_loss = (loss * weights.view(-1)).sum()
        # Normalize by the sum of static weights (this includes valid and pad tokens).
        total_weight = weights.sum()
        
        normalized_loss = total_loss / total_weight
        return normalized_loss
    
    def initialize_codebook_with_kmeans(self, data_loader, device='cuda', max_datapoints=2_000_000, batch_size=100_000):
        """
        Initialize codebook using k-means clustering on encoder outputs from a pre-trained model.
        
        Args:
            data_loader: DataLoader containing representative data samples
            device: Device to run computation on
            max_datapoints: Maximum number of data points to use for k-means clustering
            batch_size: Batch size for k-means clustering
        """
 
        print(f"Collecting latent vectors for codebook initialization (max: {max_datapoints} points)...")
        latent_vectors = []
        total_vectors = 0
        
        # Set model to eval mode
        self.eval()

        batch_size = data_loader.batch_size
        num_codes = self.config.n_codes

        num_iters = max_datapoints // (batch_size * num_codes)
        
        with torch.no_grad():
            # Wrap data_loader with progress bar
            pbar = tqdm(data_loader, total=num_iters)
            for batch in pbar:
                x = batch[0].to(device)  # Adjust depending on your dataloader format
                B, S = x.size()
                
                grid_pos_indices = self.grid_pos_indices.expand(B, -1, -1)
                latent_pos_indices = self.latent_pos_indices.expand(B, -1)
                
                # Ensure encoder outputs are in full precision for k-means
                with autocast(device_type='cuda', enabled=False):
                    z_e_x, _ = self.encode(x, grid_pos_indices, latent_pos_indices)
                    # Convert to float32 for scikit-learn compatibility
                    batch_vectors = z_e_x.float().reshape(-1, self.config.n_dim).cpu()
                
                latent_vectors.append(batch_vectors)
                total_vectors += batch_vectors.size(0)
                
                # Update progress bar description
                pbar.set_description(f"Collected {total_vectors}/{max_datapoints} vectors")
                
                # Break if we've exceeded the maximum
                if total_vectors >= max_datapoints:
                    pbar.close()
                    print(f"Reached maximum number of datapoints ({max_datapoints})")
                    break
        
        # Concatenate all batches
        latent_vectors = torch.cat(latent_vectors, dim=0)
        
        if total_vectors < max_datapoints:
            print(f"Dataloader exhausted. Using all available {total_vectors} latent vectors for clustering")
        else:
            print(f"Using {total_vectors} latent vectors for k-means clustering")
        
        # Perform k-means clustering with progress bar
        print(f"Running k-means clustering with {self.config.codebook_size} centroids...")
        kmeans = MiniBatchKMeans(
            n_clusters=self.config.codebook_size,
            random_state=0,
            verbose=1,
            max_iter=300,
            batch_size=batch_size,  # Process in smaller batches
            max_no_improvement=50
        )
        
        # Make sure the tensor is in float32 before converting to numpy
        kmeans.fit(latent_vectors.numpy())
        
        # Initialize codebook with cluster centroids
        centroids = torch.from_numpy(kmeans.cluster_centers_).float()
        self.codebook.vq_embs.weight.data.copy_(centroids)
        
        # Return to training mode
        self.train()
        print("Codebook initialized with k-means centroids!")
