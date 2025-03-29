from dataclasses import dataclass
import math
import torch
from torch.autograd import Function
from torch.nn import functional as F
from torch import Tensor, nn
from typing import List, Optional, Tuple
from torch.amp import autocast



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
        torch.nn.init.normal_(self.latent_tokens, mean=0.0, std=0.02)
        
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



# Reference: https://github.com/Vrushank264/VQVAE-PyTorch/tree/main

class VectorQuantization(Function):
    """
    Custom autograd Function for vector quantization in a VQ-VAE.
    
    This function maps each input vector (of shape [B, N, C]) to the index of its closest
    embedding in the codebook (of shape [K, C]) using a squared L2 distance.
    
    Note: This function is non-differentiable; its backward pass is disabled.
    
    Expected input shape: [B, N, C]
      B = Batch size
      N = Number of tokens (or discrete codes)
      C = Embedding dimension
    """
    @staticmethod
    def forward(ctx, inputs, codebook):
        # Hard dimensions: inputs is expected to be [B, N, C]
        B, N, C = inputs.shape
        
        # Flatten the inputs to shape [B*N, C]
        flat_input = inputs.reshape(B * N, C)
        
        # Compute squared L2 norm of each codebook vector.
        # codebook: shape [K, C] --> codebook_sq: shape [K]
        codebook_sq = torch.sum(codebook * codebook, dim=1)
        
        # Compute squared L2 norm of each input vector.
        # flat_input: [B*N, C] --> inputs_sq: [B*N, 1]
        inputs_sq = torch.sum(flat_input * flat_input, dim=1, keepdim=True)
        
        # Compute squared Euclidean distance between each input and each codebook vector.
        # Using the identity: ||x - e||^2 = ||x||^2 + ||e||^2 - 2*(x · e)
        # Here, torch.addmm computes: (codebook_sq + inputs_sq) + 2 * (flat_input @ codebook.T)
        # Since we're only selecting the minimum, the sign is unimportant.
        # Resulting shape: [B*N, K]
        l2_dis = torch.addmm(input=codebook_sq + inputs_sq,
                             mat1=flat_input,
                             mat2=codebook.t(),
                             alpha=2.0, beta=1.0)
        
        # For each input vector, find the index of the codebook vector with minimum distance.
        # idx_flat: shape [B*N]
        _, idx_flat = torch.min(l2_dis, dim=1)
        
        # Reshape indices back to hard dimensions [B, N]
        idx = idx_flat.reshape(B, N)
        
        # Mark these indices as non-differentiable.
        ctx.mark_non_differentiable(idx)
        
        return idx

    @staticmethod
    def backward(ctx, grad_outputs):
        raise RuntimeError("Backward pass is not defined for VectorQuantization. "
                           "Use VQStraightThrough instead.")

VQ = VectorQuantization.apply


class VQStraightThrough(Function):
    """
    Custom autograd Function implementing the straight-through estimator for vector quantization.
    
    In the forward pass, it uses the above VectorQuantization (VQ) to get the nearest codebook
    indices, then retrieves the corresponding codebook embeddings (quantized vectors). In the
    backward pass, gradients are passed directly (straight-through) to the encoder, while gradients
    for the codebook are accumulated based on the quantization indices.
    
    Expected input shape: [B, N, C]
      B = Batch size
      N = Number of tokens
      C = Embedding dimension
    Codebook shape: [K, C]
    """
    @staticmethod
    def forward(ctx, inputs, codebook):
        # Hard dimensions: inputs is expected to be [B, N, C]
        B, N, C = inputs.shape
        
        # Get nearest codebook indices using VectorQuantization.
        idx = VQ(inputs, codebook)  # idx has shape [B, N]
        
        # Flatten indices to shape [B*N] for later use.
        flat_idx = idx.reshape(B * N)
        
        # Save the flattened indices and the codebook for use in the backward pass.
        ctx.save_for_backward(flat_idx, codebook)
        ctx.mark_non_differentiable(flat_idx)
        
        # Retrieve quantized embeddings via index selection.
        # This gives a tensor of shape [B*N, C].
        codes_flat = torch.index_select(codebook, dim=0, index=flat_idx)
        
        # Reshape back to hard dimensions [B, N, C].
        codes = codes_flat.reshape(B, N, C)
        
        # Return both the codes and the original indices (not flattened)
        return codes, flat_idx, idx

    @staticmethod
    def backward(ctx, grad_outputs, grad_flat_idx, grad_idx):
        grad_inputs, grad_codebook = None, None
        
        # Pass the gradients to the encoder inputs directly using the straight-through method.
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.clone()
        
        # Compute gradients with respect to the codebook.
        if ctx.needs_input_grad[1]:
            # grad_outputs: gradient with respect to the quantized codes, shape [B, N, C]
            flat_idx, codebook = ctx.saved_tensors
        
            # Get the embedding dimension from the codebook.
            C = codebook.shape[1]

            # Flatten grad_outputs to shape [B*N, C] to match flat_idx.
            flat_grad_output = grad_outputs.reshape(-1, C)
            
            # Initialize gradient for the codebook with zeros.
            grad_codebook = torch.zeros_like(codebook)
            
            # Accumulate gradients for each codebook vector using the saved indices.
            grad_codebook.index_add_(0, flat_idx, flat_grad_output)
        
        return grad_inputs, grad_codebook


VQ_ST =  VQStraightThrough.apply


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
    def __init__(self, K: int, D: int):
        """
        Initialize the VQEmbedding module.
        
        Args:
            K (int): Total number of embeddings in the codebook.
            D (int): Dimensionality of each embedding.
        """
        super(VQEmbedding, self).__init__()
        # Create an embedding layer (the codebook) with K embeddings of dimension D.
        self.vq_embs = nn.Embedding(K, D)
        
        # Initialize with normal distribution
        self.vq_embs.weight.data.normal_(0, 1.0)
        
        # Scale to match expected norm from RMSNorm (approximately sqrt(D))
        expected_norm = math.sqrt(D)
        current_norm = torch.norm(self.vq_embs.weight.data, dim=1, keepdim=True).mean()
        self.vq_embs.weight.data *= (expected_norm / current_norm)
        
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
        # For transformer output, the shape is already (B, N, C) so no permutation is needed.
        # Use the vector quantization function (VQ) to get the nearest codebook indices.
        # VQ is the custom autograd function (VectorQuantization.apply).
        latents = VQ(z_e_x, self.vq_embs.weight)  # Resulting shape: (B, N)
        return latents
    
    def straight_through_forward(self, z_e_x):
        """
        Forward pass with a straight-through estimator.
        
        This method quantizes the input vectors and allows gradients to flow through
        the quantization process by using the straight-through approach. It returns:
            - z_q_x: Quantized vectors with shape (B, N, C).
            - zqx_tilde: An alternative quantized representation derived directly by index selection,
                         also with shape (B, N, C).
            - idx: Quantization indices with shape (B, N).
        
        Args:
            z_e_x (Tensor): The continuous encoded output from the transformer.
                            Expected shape: (B, N, C), where C equals the embedding dimension D.
        
        Returns:
            tuple: (z_q_x, zqx_tilde, idx)
        """
        # Input shape: (B, N, C)
        # Apply the straight-through vector quantization.
        # VQ_ST is the straight-through variant (VQStraightThrough.apply) that allows gradient flow.
        # It now returns quantized codes (z_q_x), flat indices (flat_idx), and original indices (idx)
        z_q_x, flat_idx, idx = VQ_ST(z_e_x, self.vq_embs.weight.detach())
        
        # Directly index the codebook using flat_idx to obtain the alternative quantized representation.
        # This returns a tensor of shape (B*N, D)
        flat_zqx_tilde = torch.index_select(self.vq_embs.weight, dim=0, index=flat_idx)
        
        # Reshape the flat tensor back to the input shape: (B, N, C)
        zqx_tilde = flat_zqx_tilde.view_as(z_e_x)  # New shape: (B, N, C)
        
        # Return the quantized codes, the alternative quantized representation, and the indices
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
        }
        
        # Add computed attributes if they exist
        computed_attrs = ['n_pos']
        for attr in computed_attrs:
            if hasattr(self, attr):
                base_dict[attr] = getattr(self, attr)
                
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
            rope=rope_2d
        )

        self.latent_encoder = LatentTransformer(
            n_latent=config.n_codes,
            d_model=config.n_dim,
            n_head=config.n_head,
            n_layer=config.n_latent_layer,
            out_norm=True,   # This seems to work even when skipping codebook so let's keep it.
            rope=rope_1d
        )


        self.codebook = VQEmbedding(config.n_codes, config.n_dim)    

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
        grid_encoded, _ = self.grid_encoder(x_embd, positions=grid_pos_indices) # [B, S, n_dim]
        latent_encoded, _ = self.latent_encoder(grid_encoded, positions=latent_pos_indices) # [B, n_codes, n_dim]
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
    
    def forward(self, x: Tensor, mask_percentage: Tensor = torch.tensor(0.0)) -> Tuple[Tensor, dict, Tensor]:
        B, S = x.size()
        x_masked = self.apply_mask(x, mask_percentage)
        grid_pos_indices = self.grid_pos_indices.expand(B, -1, -1)
        latent_pos_indices = self.latent_pos_indices.expand(B, -1)
    
        z_e_x = self.encode(x_masked, grid_pos_indices, latent_pos_indices) # [B, n_codes, n_dim]
        
        usage_stats = None
        
        if self.skip_codebook:
            decoded_logits = self.decode(z_e_x, grid_pos_indices, latent_pos_indices)
            vq_loss = torch.tensor(0.0, device=x.device)
            commitment_loss = torch.tensor(0.0, device=x.device)
        else:
            # Get quantized vectors and indices
            z_q_x_st, z_q_x, indices = self.codebook.straight_through_forward(z_e_x) # [B, n_codes, n_dim]
            
            decoded_logits = self.decode(z_q_x_st, grid_pos_indices, latent_pos_indices)
            vq_loss = F.mse_loss(z_q_x, z_e_x.detach())
            commitment_loss = F.mse_loss(z_e_x, z_q_x.detach())
    
        ce_loss = self.reconstruction_loss(decoded_logits, x, pad_value=self.pad_value, gamma=self.gamma)

        losses = {
             "ce_loss": ce_loss,  # Weight normalized loss
             "vq_loss": vq_loss,
             "commitment_loss": commitment_loss
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
    