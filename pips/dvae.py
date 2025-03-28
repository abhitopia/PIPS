#%%
import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical, RelaxedOneHotCategorical
from torch.nn import functional as F
from dataclasses import dataclass
from torch.amp import autocast
from .kld import compute_decomposed_kld, monte_carlo_kld, approximate_kld_loss


def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

def is_perfect_square(n):
    if n < 0:
        return False
    root = math.isqrt(n)  # or int(math.sqrt(n))
    return root * root == n

@dataclass
class Config:
    pass


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
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w2.RESCALE_INIT = True

        # Note that it adds extra params, but I don't care about it.
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

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
        Args:
            x (Tensor): input tensor to normalize

        Returns:
            Tensor: The output tensor after applying RMSNorm.
        """
        # computation is in fp32
        x_fp32 = x.float()
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return x_normed * self.scale


class MultiHeadAttention(nn.Module):
    def __init__(self, n_dim, n_head, dropout, rope=None):
        super().__init__()
        # self.config = config
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
    def __init__(self, d_model, n_head, rope=None, dim_feedforward=None, dropout=0.0):
        super().__init__()
        self.mha = MultiHeadAttention(n_dim=d_model, n_head=n_head, dropout=dropout, rope=rope)
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
        attn_out, new_kv_cache = self.mha(normed_queries, normed_context, normed_context, positions=positions, attn_mask=attn_mask, kv_cache=kv_cache, return_kv_cache=return_kv_cache)  # shape (B, Tq, D)
        queries = queries + self.dropout(attn_out)

        # Feed-forward
        queries = queries + self.dropout(self.ff(self.norm_ff(queries)))  # shape (B, Tq, D)
        return queries, new_kv_cache

    
class Transformer(nn.Module):
    def __init__(self, d_model, n_head, n_layer, dim_feedforward=None, rope=None, out_norm=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_head=n_head, dim_feedforward=dim_feedforward, rope=rope)
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
        
        # Stack of cross-attn blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_head=n_head, dim_feedforward=dim_feedforward, rope=rope)
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

        # positions = self.positions.expand(B, -1)
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


class AttnCodebook(nn.Module):
    """
    A single-head attention-based codebook module.
    
    The latent encoder outputs are treated as queries, and the codebook consists
    of learned keys and values. This allows the module to compute an attention 
    weighted sum over the codebook entries.
    
    Args:
        d_model (int): The model dimension.
        codebook_size (int): The number of codebook entries.
    
    Shapes:
        - queries: [B, N, d_model] where B=batch size, N=number of tokens.
        - codebook_keys: [d_model, codebook_size]
        - codebook_values: [codebook_size, d_model]
    """
    def __init__(self, d_model: int, codebook_size: int, use_exp_relaxed=False, dim_feedforward=None, rope=None, normalise_kq: bool = False,
                 decay: float = 0.99999, epsilon: float = 1e-5, codebook_ema_update: bool = True):
        super().__init__()
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.use_exp_relaxed = use_exp_relaxed
        self.rope = rope
        self.normalise_kq = normalise_kq

        self.decay = decay
        self.epsilon = epsilon
        self.codebook_ema_update = codebook_ema_update

        self.norm_context = RMSNorm(dim=d_model)
        self.norm_queries = RMSNorm(dim=d_model)

        # Shared codebook embeddings: shape [codebook_size, d_model]
        self.codebook = nn.Parameter(torch.randn(codebook_size, d_model))
        # EMA buffers to keep track of cluster sizes and the running average of codebook embeddings.
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size), persistent=False)
        self.register_buffer("ema_codebook", self.codebook.data.clone(), persistent=False)

        ## Attention Stuff
        # Projection layers to generate keys and values from the codebook.
        # These layers project the codebook from [codebook_size, d_model] to the same shape.
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=True)
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = torch.sqrt(torch.tensor(self.d_model))
        self.c_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        ## Feedforward Stuff
        self.ff = SwiGLUFFN(dim=self.d_model, hidden_dim=dim_feedforward if dim_feedforward is not None else 4*self.d_model)
        self.norm_ff = RMSNorm(dim=d_model)


    def single_head_attention(self, queries: Tensor, keys: Tensor, values: Tensor, tau: Tensor, positions: Tensor, gumbel_noise_scale: Tensor = torch.tensor(0.0)) -> Tensor:
        """
        Single-head attention pass.
        
        Args:
            queries (Tensor): Query tensors of shape [B, N, d_model].
            keys (Tensor): Key tensors of shape [C, d_model].
            values (Tensor): Value tensors of shape [C, d_model].
            tau (Tensor): Temperature parameter for controlling distribution sharpness.
            positions (Tensor): Position indices for rotary position embeddings.
            gumbel_noise_scale (Tensor): Scale of Gumbel noise to add when sampling=False. Value between 0 and 1.
                                         Only applied when self.sampling is False.
        
        Returns:
            Tuple[Tensor, Tensor, Tensor]: (attention_output, log_alpha, z)
        """
        # Project queries: [B, N, d_model]
        q = self.query_proj(queries).unsqueeze(1) # [B, 1, N, d_model]
        k = self.key_proj(keys).unsqueeze(0).expand(q.size(0), -1, -1).unsqueeze(1) # [B, 1, C, d_model]
        v = self.value_proj(values).unsqueeze(0).expand(q.size(0), -1, -1) # [B, C, d_model]

        qT = q.size(2)
        kT = k.size(2)

        q = self.rope(q, positions[:, :qT].unsqueeze(1)).squeeze(1)
        k = self.rope(k, positions[:, :kT].unsqueeze(1)).squeeze(1)


        if self.normalise_kq:
            ## Normalise the keys and queries so network doesn't fight back the decrease in temperature
            q = q / torch.norm(q, dim=-1, keepdim=True)
            k = k / torch.norm(k, dim=-1, keepdim=True)

        # Compute scaled dot-product attention.
        log_alpha = torch.matmul(q, k.mT) # [B, N, C]
        log_alpha = log_alpha / self.scale # [B, N, C]
        
        # Sample Gumbel noise: -log(-log(uniform(0,1)))
        gumbel = -torch.log(-torch.log(torch.rand_like(log_alpha) + 1e-10) + 1e-10)
        # Add scaled Gumbel noise to logits before temperature scaling
        log_alpha_tau = (log_alpha + gumbel_noise_scale * gumbel) / tau
        z = torch.softmax(log_alpha_tau, dim=-1)


        attn_output = torch.matmul(z, v) # [B, N, d_model]
        y = self.c_proj(attn_output) # [B, N, d_model]

        return y, log_alpha, log_alpha_tau, z
    

    def forward(self, latents: Tensor, tau: Tensor, residual_scaling: Tensor, positions: Tensor, gumbel_noise_scale: Tensor = torch.tensor(0.0)) -> Tensor:
        """
        Forward pass through the codebook.

        Args:
            latents (Tensor): The input latents.
            tau (Tensor): The temperature for the Gumbel-Softmax distribution.
            residual_scaling (Tensor): The scaling factor for the residual connection.
            positions (Tensor): The positions of the latents.
            gumbel_noise_scale (Tensor): Scale of Gumbel noise to add when sampling=False. Value between 0 and 1.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple of (codebook_output, log_alpha, z), where:
                codebook_output (Tensor): The output of the codebook.
                log_alpha (Tensor): The unnormalized logits used for distribution over the codebook.
                z (Tensor): The soft assignment code (either sampled or computed deterministically via softmax).
        """

        normed_context = self.norm_context(self.codebook) # [C, d_model]
        normed_queries = self.norm_queries(latents) # [B, N, d_model]

        attn_output, log_alpha, log_alpha_tau, z = self.single_head_attention(
                                                            queries=normed_queries, 
                                                            keys=normed_context, 
                                                            values=normed_context, 
                                                            tau=tau,
                                                            positions=positions,
                                                            gumbel_noise_scale=gumbel_noise_scale)

        # Notice that this mixes the latents with the codebook embeddings.
        # But this also means that attn_output is not a function of log_alpha only.
        # As such, we add a residual scaling factor to the residual connection.
        attn_output = residual_scaling * latents + attn_output # Residual connection

        codebook_output = attn_output + self.ff(self.norm_ff(attn_output)) # Residual connection

        # EMA update for the codebook
        if self.training and self.codebook_ema_update:
            self._update_codebook(latents, z)

        return codebook_output, log_alpha, log_alpha_tau, z
    


    def _update_codebook(self, latents: Tensor, z: Tensor):
        """
        Update the codebook embeddings using exponential moving average (EMA).
        
        Args:
            latents (Tensor): Input latents of shape [B, N, d_model].
            z (Tensor): Soft assignments (probabilities) of shape [B, N, codebook_size].
        """
        with torch.no_grad():
            # Reshape to merge batch and token dimensions: [B*N, d_model] and [B*N, codebook_size]
            flat_latents = latents.reshape(-1, self.d_model)
            flat_z = z.reshape(-1, self.codebook_size)

            # Compute the batch statistics: sum of assignment probabilities and weighted latent sums
            cluster_size_batch = flat_z.sum(dim=0)  # [codebook_size]
            codebook_update_batch = flat_z.t() @ flat_latents  # [codebook_size, d_model]

            # Update the EMA buffers
            self.ema_cluster_size.mul_(self.decay).add_(cluster_size_batch, alpha=1 - self.decay)
            self.ema_codebook.mul_(self.decay).add_(codebook_update_batch, alpha=1 - self.decay)

            # Normalize the EMA codebook
            n = self.ema_cluster_size.sum()
            # Avoid division by zero by adding a small constant epsilon
            cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.codebook_size * self.epsilon) * n

            # Update the codebook parameter (using .data to avoid gradient interference)
            self.codebook.data.copy_(self.ema_codebook / cluster_size.unsqueeze(1))

class GumbelCodebook(nn.Module):
    def __init__(self, d_model, codebook_size, n_codes, position_dependent: bool = False, use_exp_relaxed=False, sampling: bool = True):
        super().__init__()
        self.position_dependent = position_dependent
        self.codebook_size = codebook_size
        self.use_exp_relaxed = use_exp_relaxed
        self.sampling = sampling

        # If position-dependent is enabled, create separate head and mapping for each latent position.
        if self.position_dependent:
            # Head: one per latent position.
            self.head = nn.Parameter(torch.randn(n_codes, d_model, codebook_size))
            # Unique positional mapping applied to quantized outputs.
            self.positional_mapping = nn.Parameter(torch.randn(n_codes, d_model, d_model))
        else:
            # Shared head across all positions.
            self.head = nn.Linear(d_model, codebook_size, bias=False)
            self.positional_mapping = None

        self.codebook = nn.Linear(codebook_size, d_model, bias=False)

    def sample(self, log_alpha: Tensor, tau: Tensor) -> Tensor:
        """Sample from either RelaxedOneHotCategorical or ExpRelaxedCategorical."""
        assert tau > 0.0, "Temperature must be greater than 0.0"
        # tau is already a tensor, so we simply use it directly.
        if self.use_exp_relaxed:
            # We need to exponentiate the sample to get the correct sample from the distribution.
            return ExpRelaxedCategorical(tau, logits=log_alpha).rsample().exp()
        else:
            return RelaxedOneHotCategorical(tau, logits=log_alpha).rsample()
        

    def forward(self, logits: Tensor, tau: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through the Gumbel-Softmax codebook.
        
        Args:
            logits (Tensor): Input tensor of shape [B, N, D].
            tau (Tensor): Temperature parameter for quantization as a scalar tensor.
                When sampling mode is enabled (self.sampling=True), tau is used as the temperature in the Gumbel-Softmax sampling.
                When sampling mode is disabled (self.sampling=False), tau is used to scale the logits before softmax.
            
        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple of (quantized, log_alpha, z), where:
                quantized (Tensor): The quantized vector after projection via the codebook.
                log_alpha (Tensor): The logits after projecting through the head.
                z (Tensor): The soft assignment code (either sampled or computed deterministically via softmax).
        """
        if not self.training:
            return self.inference(logits)

        # If using a position-dependent head, compute logits using a per-position projection.
        if self.position_dependent:
            # Assume logits shape is [B, N, d_model] and head is [N, d_model, codebook_size].
            log_alpha = torch.einsum("bnd,ndc->bnc", logits, self.head)  # [B, N, codebook_size]
        else:
            log_alpha = self.head(logits)  # [B, N, codebook_size]

        if self.sampling:
            # Sample using the (Gumbel-Softmax) distribution.
            z = self.sample(log_alpha, tau)  # [B, N, C]
        else:
            # Apply softmax with temperature scaling.
            z = torch.softmax(log_alpha / tau, dim=-1)

        quantized = self.codebook(z)  # [B, N, d_model]

        if self.positional_mapping is not None:
            # Apply a unique mapping for each latent position via batched matrix multiplication.
            quantized = torch.einsum("bnd,ndm->bnm", quantized, self.positional_mapping)

        return quantized, log_alpha, z

    def inference(self, logits):

        """
        Inference pass through the Gumbel-Softmax codebook.
        This method returns a hard one-hot encoding based on the argmax of the logits.
        
        Args:
            logits (Tensor): Input tensor of shape [B, N, d_model].
            
        Returns:
            Tuple[Tensor, Tensor, Tensor]: (quantized, log_alpha, hard_one_hot)
        """
        # This is used to get the hard assignments from the logits

        # Project to codebook space: [B, N, C]
        if self.position_dependent:
            log_alpha = torch.einsum("bnd,ndc->bnc", logits, self.head)
        else:
            log_alpha = self.head(logits)
                
        # Take hard assignments: [B, N]
        hard_idx = torch.argmax(log_alpha, dim=-1)
        
        # More efficient approach - create one-hot tensor directly on the correct device
        hard_one_hot = torch.zeros_like(log_alpha)
        hard_one_hot.scatter_(-1, hard_idx.unsqueeze(-1), 1.0)
        
        # Get quantized output via the codebook
        quantized = self.codebook(hard_one_hot)

        if self.positional_mapping is not None:
            quantized = torch.einsum("bnd,ndm->bnm", quantized, self.positional_mapping)
        
        return quantized, log_alpha, hard_one_hot


@dataclass
class GridDVAEConfig(Config):
    r"""
    Configuration class for GridDVAE model.
    
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
        use_exp_relaxed (bool): Whether to use exponentially relaxed Gumbel-Softmax (default: False)
        use_monte_carlo_kld (bool): Whether to use Monte Carlo KLD (default: False)
        gamma (float): Focal loss gamma parameter. With \(\gamma=0\) there is no focal modulation, defaulting to 2.0.
    """
    n_dim: int
    n_head: int
    n_grid_layer: int
    n_latent_layer: int
    n_codes: int
    codebook_size: int = 512
    rope_base_height: int = 10007  # ~10k, prime
    rope_base_width: int = 5003    # ~5k, prime
    dropout: float = 0.0
    max_grid_height: int = 32
    max_grid_width: int = 32
    n_vocab: int = 16
    padding_idx: int | None = None
    mask_idx: int | None = None
    pad_weight: float = 0.01,
    use_exp_relaxed: bool = False,
    use_monte_carlo_kld: bool = False,
    gamma: float = 2.0
    init_mode: str = "normal"
    skip_codebook: bool = False
    normalise_kq: bool = False
    use_pure_logits_for_loss: bool = False
    codebook_ema_update: bool = True

    def __post_init__(self):
        if self.n_dim % self.n_head != 0:
            raise ValueError("n_dim must be divisible by n_head")
        
        C = self.n_dim // self.n_head
        assert C % 2 == 0, "n_dim // n_head must be divisible by 2"

        head_dim = C // 2  # Actual Head Dimension
        assert head_dim % 2 == 0, "Head dimension must be even"

        # Calculate n_pos from grid dimensions
        self.n_pos = self.max_grid_height * self.max_grid_width

        assert self.init_mode in ["normal", "xavier"], "Invalid initialization mode"
        
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
            'use_exp_relaxed': self.use_exp_relaxed,
            'use_monte_carlo_kld': self.use_monte_carlo_kld,
            'gamma': self.gamma,
            'init_mode': self.init_mode,
            'skip_codebook': self.skip_codebook,
            'normalise_kq': self.normalise_kq,
            'use_pure_logits_for_loss': self.use_pure_logits_for_loss,
            'codebook_ema_update': self.codebook_ema_update
        }
        
        # Add computed attributes if they exist
        computed_attrs = ['n_pos']
        for attr in computed_attrs:
            if hasattr(self, attr):
                base_dict[attr] = getattr(self, attr)
                
        return base_dict

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'GridDVAEConfig':
        """Create config from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            GridDVAEConfig: New config instance
            
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


class GridDVAE(nn.Module):
    def __init__(self, config: GridDVAEConfig):
        super().__init__()
        self.config = config
        self.use_monte_carlo_kld = config.use_monte_carlo_kld
        self.use_exp_relaxed = config.use_exp_relaxed
        self.skip_codebook = config.skip_codebook
        self.gamma = config.gamma
        self.n_pos = config.n_pos
        self.embd = nn.Embedding(config.n_vocab, config.n_dim)
        self.pad_value = config.padding_idx  # Store padding value here
        self.mask_value = config.mask_idx
        self.use_pure_logits_for_loss = config.use_pure_logits_for_loss
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

        rope_codebook = RotaryPositionalEmbeddings(
            dim=config.n_dim // 1,
            max_seq_len=config.n_pos,
            base=config.rope_base_height
        )

        
        self.grid_encoder = Transformer(
            d_model=config.n_dim,
            n_head=config.n_head,
            n_layer=config.n_grid_layer,
            out_norm=False,
            rope=rope_2d
        )

        self.latent_encoder = LatentTransformer(
            n_latent=config.n_codes,
            d_model=config.n_dim,
            n_head=config.n_head,
            n_layer=config.n_latent_layer,
            out_norm=False,
            rope=rope_1d
        )

        self.codebook = AttnCodebook(d_model=config.n_dim, 
                                    codebook_size=config.codebook_size,
                                    use_exp_relaxed=config.use_exp_relaxed,
                                    rope=rope_codebook,
                                    normalise_kq=config.normalise_kq,
                                    codebook_ema_update=config.codebook_ema_update)

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

        rows = torch.arange(config.max_grid_height, dtype=torch.long)
        cols = torch.arange(config.max_grid_width, dtype=torch.long)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
        grid_pos_indices = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).unsqueeze(0)
        latent_pos_indices = torch.arange(config.n_pos).unsqueeze(0)

        self.register_buffer("latent_pos_indices", latent_pos_indices, persistent=False)
        self.register_buffer('grid_pos_indices', grid_pos_indices, persistent=False)

        # Apply weight initialization on registered modules.
        self.apply(self._init_weights)
        # Additionally, initialize any raw nn.Parameters.
        self.initialize_all_parameters()

    def _init_weights(self, module):
        # Get initialization mode from config (if present). Default is "normal".
        # Set self.config.init_mode = 'xavier' if you prefer Xavier initialization.
        init_mode = getattr(self.config, "init_mode", "normal")
        
        if isinstance(module, nn.Linear):
            if init_mode == "xavier":
                torch.nn.init.xavier_normal_(module.weight)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            module.weight._initialized = True
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                module.bias._initialized = True
        elif isinstance(module, nn.Embedding):
            if init_mode == "xavier":
                torch.nn.init.xavier_normal_(module.weight)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            module.weight._initialized = True
        elif isinstance(module, nn.LayerNorm):
            # Often LayerNorm weights are initialized to ones and biases to zeros.
            torch.nn.init.ones_(module.weight)
            module.weight._initialized = True
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                module.bias._initialized = True

    def initialize_all_parameters(self):
        """
        Initialize all parameters in the model recursively.
        This method is meant to also initialize raw nn.Parameter attributes that are not part
        of a submodule (and hence not handled by self.apply).
        """
        init_mode = getattr(self.config, "init_mode", "normal")
        for name, param in self.named_parameters():
            # If the parameter has already been initialized (flagged via _initialized), skip.
            if hasattr(param, "_initialized"):
                continue
            if param.ndim >= 2:
                if init_mode == "xavier":
                    torch.nn.init.xavier_normal_(param)
                else:
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
            param._initialized = True

        # Update ema_codebook to match the codebook after parameter initialization.
        self.codebook.ema_codebook.copy_(self.codebook.codebook.data)

    def encode(self, x: Tensor, grid_pos_indices: Tensor, latent_pos_indices: Tensor) -> Tensor:
        x_embd = self.embd(x)
        grid_encoded, _ = self.grid_encoder(x_embd, positions=grid_pos_indices)
        latent_encoded, _ = self.latent_encoder(grid_encoded, positions=latent_pos_indices)
        return latent_encoded


    def decode(self, x: Tensor, grid_pos_indices: Tensor, latent_pos_indices: Tensor) -> Tensor:    
        latent_decoded, _ = self.latent_decoder(x, positions=latent_pos_indices)        
        grid_decoded, _ = self.grid_decoder(latent_decoded, positions=grid_pos_indices)        
        grid_decoded_logits = self.decoder_head(grid_decoded)
        return grid_decoded_logits
    

    def apply_mask(self, x: Tensor, mask_percentage: Tensor) -> Tensor:
        x = x.clone()
        # Create a random tensor with values in the range [0,1) for each element in x.
        mask = (torch.rand(x.shape, device=x.device, dtype=torch.float32) < mask_percentage) & (x != self.pad_value)
        x.masked_fill_(mask, self.mask_value)
        return x
    



    def forward(self, x: Tensor, q_z_marg: Optional[Tensor] = None, tau: Tensor = torch.tensor(1.0), mask_percentage: Tensor = torch.tensor(0.0), residual_scaling: Tensor = torch.tensor(0.0), gumbel_noise_scale: Tensor = torch.tensor(0.0)) -> Tuple[Tensor, dict, Tensor]:
        B, S = x.size()
        x_masked = self.apply_mask(x, mask_percentage)
        grid_pos_indices = self.grid_pos_indices.expand(B, -1, -1)
        latent_pos_indices = self.latent_pos_indices.expand(B, -1)
    
        encoded_logits = self.encode(x_masked, grid_pos_indices, latent_pos_indices)
    
        quantized, log_alpha, log_alpha_tau, z = self.codebook(encoded_logits, tau=tau, residual_scaling=residual_scaling, positions=latent_pos_indices, gumbel_noise_scale=gumbel_noise_scale)
            
        log_alpha_loss = log_alpha if self.use_pure_logits_for_loss else log_alpha_tau

        if self.use_monte_carlo_kld:
            kld_losses = monte_carlo_kld(log_alpha, tau=tau, reduction='mean', use_exp_relaxed=self.use_exp_relaxed)
            q_z_marg_updated = q_z_marg
        else:
            kld_losses, q_z_marg_updated = compute_decomposed_kld(log_alpha_loss, q_z_marg, reduction='mean')
    
        # Calculate codebook-related losses efficiently
        diversity_losses = self.compute_diversity_losses(log_alpha_loss)

        if self.skip_codebook:
            decoded_logits = self.decode(encoded_logits, grid_pos_indices, latent_pos_indices)
        else:
            decoded_logits = self.decode(quantized, grid_pos_indices, latent_pos_indices)
    
        ce_loss = self.reconstruction_loss(decoded_logits, x, pad_value=self.pad_value, gamma=self.gamma)
    
        losses = {
             "ce_loss": ce_loss,  # Weight normalized loss
             **diversity_losses,  # Include all diversity losses
             **kld_losses.to_dict()
        }
        return decoded_logits, log_alpha_tau, losses, q_z_marg_updated
    

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
    

    def compute_diversity_losses(self, log_alpha_tau: torch.Tensor):
        """
        Computes diversity losses from log_alpha.
        
        Returns a dictionary with:
        - diversity_entropy: average per-token entropy (we want this low)
        - diversity_sample: negative entropy of aggregated per-sample distribution
        - diversity_position: negative entropy of aggregated per-position distribution
        - diversity_usage: KL divergence between overall codebook usage and uniform distribution
        """
        # Use a single small epsilon value for numerical stability
        epsilon = 1e-10
        
        # Compute log probabilities and probabilities in one go
        log_p = F.log_softmax(log_alpha_tau, dim=-1)
        p = torch.exp(log_p)

        # Entropy Loss: average per-token entropy.
        # We can use the log_p we already computed
        token_entropy = -(p * log_p).sum(dim=-1)  # [B, N]
        diversity_entropy = token_entropy.mean()  # Positive; lower is better.

        # Sample Loss: aggregated distribution per sample (average over tokens).
        q = p.mean(dim=1)  # [B, C]
        log_q = torch.log(q + epsilon)
        H_q = -(q * log_q).sum(dim=-1)  # [B]
        diversity_sample = -H_q.mean()  # Negative; minimizing encourages higher entropy.

        # Position Loss: aggregated distribution per token position (average over batch).
        r = p.mean(dim=0)  # [N, C]
        log_r = torch.log(r + epsilon)
        H_r = -(r * log_r).sum(dim=-1)  # [N]
        diversity_position = -H_r.mean()  # Negative; minimizing encourages higher entropy.

        # Usage Loss: measures if all codebook entries are being utilized equally
        # First, compute the aggregated usage by summing over batch and position dimensions
        # This gives us the total "usage count" for each codebook entry
        codebook_usage = p.sum(dim=(0, 1))  # [C]
        
        # Normalize to get a proper probability distribution
        codebook_usage = codebook_usage / codebook_usage.sum()
        
        # Compute KL divergence with uniform distribution
        # KL(usage||uniform) = Σ(usage_i * log(usage_i / (1/C)))
        log_codebook_size = math.log(self.config.codebook_size)
        
        # Simplified computation using properties of logarithms
        usage_log_usage = codebook_usage * torch.log(codebook_usage + epsilon)
        diversity_usage = usage_log_usage.sum() + log_codebook_size
        
        return {
            "diversity_entropy": diversity_entropy, 
            "diversity_sample": diversity_sample, 
            "diversity_position": diversity_position, 
            "diversity_usage": diversity_usage
        }


#%%
