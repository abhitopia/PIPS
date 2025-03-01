#%%
import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from dataclasses import dataclass
from torch.amp import autocast


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
        base (int): Base for geometric progression in angle computation.
    """

    def __init__(
        self,
        dim: int,
        max_height: int = 1024,
        max_width: int = 1024,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension 'dim' must be even to split for 2D RoPE."

        self.dim = dim
        self.half_dim = dim // 2

        # Initialize two RotaryPositionalEmbeddings for each positional dimension
        self.rope_height = RotaryPositionalEmbeddings(
            dim=self.half_dim,
            max_seq_len=max_height,
            base=base
        )
        self.rope_width = RotaryPositionalEmbeddings(
            dim=self.half_dim,
            max_seq_len=max_width,
            base=base
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


class Codebook(nn.Module):
    def __init__(self, d_model, codebook_size):
        super().__init__()
        self.head = nn.Linear(d_model, codebook_size, bias=False)
        self.codebook = nn.Linear(codebook_size, d_model, bias=False)

    def sample(self, soft_code, logits, reinMax: bool = False):
          # Straight through
        index = soft_code.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(
            soft_code, memory_format=torch.legacy_contiguous_format
        ).scatter_(-1, index, 1.0)
        
        if reinMax:
            # ReinMax: Algorithm 2 in https://arxiv.org/pdf/2304.08612
            # Notice that I use pi_0 from gumbel instead of the softmax, this is deliberate
            # The noise in gumbel softmax better captures the stochasticity of sampling
            # For example, even in STE (Algorithm 1), the authors don't use gumbel softmax as pi_0
            # However, even the official gumbel softmax implementation in PyTorch uses the softmax with gumbel noise
            pi_0 = soft_code # Step 1
            D = y_hard # Step 2
            pi_1 = (D + pi_0) / 2
            pi_1 = F.softmax((pi_1.log() - logits).detach() - logits, dim=-1)
            pi_2 = 2 * pi_1 - 0.5 * pi_0
            hard_code = pi_2 - pi_2.detach() + D
        else:
            hard_code = y_hard - soft_code.detach() + soft_code

        return hard_code

    def forward(self, logits, tau: float = 1.0, hardness: Tensor = torch.tensor(0.0), reinMax: bool = False):
        logits = self.head(logits)

        # Compute both versions of soft_code.
        soft_code_softmax = F.softmax(logits / tau, dim=-1)
        soft_code_gumbel = F.gumbel_softmax(logits, tau=tau, hard=False)
        
        # Use hardness to select between softmax and gumbel-softmax.
        # (Assuming hardness < 0 means use softmax, otherwise use gumbel-softmax)
        cond_soft = hardness < 0
        # Expand the condition to match the shape of the soft code if needed.
        cond_soft = cond_soft.expand_as(soft_code_softmax) if soft_code_softmax.dim() > 0 else cond_soft
        soft_code = torch.where(cond_soft, soft_code_softmax, soft_code_gumbel)
        
        # Compute the hard code branch.
        hard_code = self.sample(soft_code, logits=logits, reinMax=reinMax)
        
        # If hardness > 0, use a linear combination of the hard and soft codes.
        # Otherwise, just use soft_code.
        cond_code = hardness > 0
        combined = hardness * hard_code + (1 - hardness) * soft_code
        # Again, expand the condition if necessary.
        cond_code = cond_code.expand_as(soft_code) if soft_code.dim() > 0 else cond_code
        code = torch.where(cond_code, combined, soft_code)
    
        return self.codebook(code), soft_code, code

    def forward_no_compile(self, logits, tau: float = 1.0, hardness: float = 0.0, reinMax: bool = False):
        logits = self.head(logits)
        
        if hardness < 0:
            # Compute both softmax and gumbel-softmax versions
            soft_code = F.softmax(logits/tau, dim=-1)
        else:
            soft_code = F.gumbel_softmax(logits, tau=tau, hard=False)
            
        # For the hard code path, also use tensor operations
        if hardness > 0:
            hard_code = self.sample(soft_code, logits=logits, reinMax=reinMax)
            code = hardness * hard_code + (1 - hardness) * soft_code
        else:
            code = soft_code

        return self.codebook(code), soft_code, code

    @staticmethod
    def kld_disentanglement_loss(q_z_x, q_z_marg=None, momentum=0.99, eps=1e-8):
        """
        The Beta-TCVAE paper(https://arxiv.org/pdf/1802.04942) splits the KLD term as

        1/N KLD(q(z|x) | p(z)) = MI + TC + DWKL

        where, 
            MI = KLD(q(z,x)|q(z)p(x))
            TC = KLD(q(z)|∏[j] q(z_j))
            DWKL = sum_j KLD(q(z_j) | p(z_i))

        Computing q(z, x), q(z) and q(z_i) are all intractable. This happens because the
        authors compute average across all the dataset which involves p(x_n) and makes life
        complicated. I suspect, this is useful for when the dataset size is small and fixed.
        However, I want to operate in the large/infinite data regime. So I will try to compute
        this first per sample x_n (like it done for VAE) and average for the batch later.

        Using similar derivation as in the paper, we get following

        KL_n(q(z| x_n) | p(z)) = MI_n + TC_n + DWKL_n
        where 
            MI_n   = E{q(z|x_n)}[log q(z|x_n)/q(z)]
            TC_n   = E{q(z|x_n)}[log q(z)/ ∏{j} q(z_j)]
            DWKL_n = Sum_j E{q(z|x_n)}[log q(z_j)/p(z_j)]

        Notice that since, expectation is a sum over all latents, this will be C^N
        different values making it intractable. So regular VAE (computing LHS directly),
        we have to assume assume:
        - The latent vector z = (z_1, z_2, ..., z_N) factorizes:
                q(z|x) = ∏[j=1 to N] q(z_j|x)
        - The prior also factorizes:
                p(z) = ∏[j=1 to N] p(z_j)

        The encoder gives us access to q(z|x_n), but we don't have access to aggregate posterior
        q(z), or it's marginal per latent q(z_i) which are global quantities (not dependent on the batch).
        The original paper uses MWS and MSS sampling to estimate these which are too complicated and ineffecient. 

        We can use Monte Carlo sampling to estimate q(z_i) (And not q(z)) as following

        q(z_i) = sum_n q(z_i |x_n) p(x_n)  # Per mini-batch of size B
               = 1/ B sum_n q(z_i | x_n)   # (N, C)          

        Because a batch is not a good representation of p(x), we can use EMA to update this over time.

        But both MI_n and TC_n still involve q(z) which remains intractable. We cannot assume that q(z)
        factorises to prod_j q(z), because if we did, then TC_n would vanish. In fact, this is the term
        that encourages disentanglement. What should we do?

        My idea here is that we approximate MI_n with the assumption that q(z) factorises, and then use
        the following identity to estimate TC_n

        TC_n = KL_n - MI_n (approx) - DWKL_n

        where 
        
        MI_n (approx) = E{q(z|x_n)}[log q(z|x_n)/∏{j} q(z_j)]
                      = E{q(z|x_n)}[log ∏{j} (q(z_j|x_n)/ q(z_j)]
                      = sum_j E{q(z|x_n)} log(q(z_j|x_n) / q(z_j))
                      = sum_j E{q(z_j|x_n)} log(q(z_j|x_n) / q(z_j))

        Compute disentanglement losses—Full KL, Mutual Information (MI), 
        Dimension-wise KL (DWKL), and Total Correlation (TC)—using an exponential moving 
        average (EMA) to estimate the aggregated posterior q(z).

        This function assumes:
        - The latent vector z = (z_1, z_2, ..., z_N) factorizes:
                q(z|x) = ∏[j=1 to N] q(z_j|x)
        - The prior also factorizes:
                p(z) = ∏[j=1 to N] p(z_j)

        For specific x, therefore, the full KL (per sample) is defined as:
            KL(q(z|x) || p(z)) = Σ[j=1 to N] KL(q(z_j|x) || p(z_j))

        The paper provides a decomposition of the "average KL" across all samples in the dataset
        which makes life complicated because we don't have access to p(x_n). Instead, following is the 
        rederivation of the decomposition (like usually done for VAEs) for each specific sample x_i.

        KL(q(z|x_i) || p(z)) = KL(q(z|x_i) || p(z))

        = E{q(z|x_i)} [log q(z|x_i) - log p(z)]

        = E{q(z|x_i)} [log q(z|x_i) - log ∏ p(z_j) + log q(z) - log q(z) + log ∏ q(z_j) - log ∏ q(z_j) ]

        = E{q(z|x_i)} [log q(z|x_i)/q(z) + log q(z)/∏ q(z_j) + log ∏ q(z_j)/p(z_j) ]

        = Σ{z} [q(z|x_i) * (log q(z|x_i)/log q(z) + log q(z)/∏ q(z_j) + Σ{j} log q(z_j)/p(z_j) )]

        = Σ{z} [q(z|x_i) * (log q(z|x_i)/log q(z) + log q(z)/∏ q(z_j) + Σ{z_j} log q(z_j)/p(z_j) )]


        MI = Σ{z} [q(z|x_i) * log q(z|x_i)/log q(z)]
        TC = Σ{z} [q(z|x_i) * (log q(z) - Σ{j} log q(z_j)]
        DWKL = Σ{z} [q(z|x_i) * Σ{z_j} log q(z_j)/p(z_j)]
      

        Also, following the decomposition:
            Full KL = MI + TC + DWKL,
        where:
            - MI = Σ[j=1 to N] KL(q(z_j|x) || q(z_j))  (averaged over samples),
            - DWKL = Σ[j=1 to N] KL(q(z_j) || p(z_j))    (computed from an estimate of q(z), 
                                                        ideally a global quantity),
            - TC = Full KL - MI - DWKL.

        To ensure consistency, we compute Full KL and MI per sample (summing over latent dimensions, then averaging over the batch) 
        and compute DWKL from the aggregated marginal. Because a minibatch may not reliably capture the global posterior, 
        we use an EMA to keep a running estimate of q(z).

        Args:
            q_z_x (Tensor): Soft one-hot distributions from Gumbel-softmax, of shape (B, N, C), where:
                B = batch size,
                N = number of discrete latent codes,
                C = codebook size.
            q_z_marg: Optional tensor of shape (N, C) containing the running estimate of q(z)
            momentum (float): The decay rate for the EMA; typical values are near 0.99
            eps (float): A small constant for numerical stability
            apply_relu (bool): Whether to apply ReLU to ensure non-negative losses

        Returns:
            Tuple containing:
            - Dictionary of KL-related losses
            - Updated marginal q(z)
        """
        # -----------------------------------------------
        # Retrieve dimensions.
        # B: batch size, N: number of latent codes, C: codebook size.
        # -----------------------------------------------
        B, N, C = q_z_x.shape

        # -----------------------------------------------
        # Step 1: Compute the current aggregated marginal posterior q(z_j) from the minibatch.
        # For each latent dimension j, compute q_z_current[j] as the average over the batch.
        # q_z_marg_current has shape (N, C). Each row is a valid probability distribution (sums to 1).
        # -----------------------------------------------
        q_z_marg_batch = q_z_x.mean(dim=0)  # Shape: (N, C)

        # -----------------------------------------------
        # Step 1b: Update the running estimate of q(z_j) using EMA.
        # If q_z_marg_running is provided, update it; otherwise, initialize it with q_z_marg_current.
        # The formula is: q_z_running_new = momentum * q_z_running_old + (1 - momentum) * q_z_current
        # This provides a smoother estimate of the global q(z) than using the current batch alone.
        # -----------------------------------------------
        q_z_marginal = (momentum * q_z_marg + (1 - momentum) * q_z_marg_batch) if q_z_marg is not None else q_z_marg_batch
        q_z_marginal_detached = q_z_marginal.detach()  # Detach for return value

        # Compute log probabilities
        log_q_z_x = torch.log(q_z_x + eps)  # Shape: (B, N, C)
        # Since p(z_j) is assumed to be uniform over C possible codes, each entry is 1/C.
        log_uniform = torch.full_like(log_q_z_x, -math.log(C))  # Shape: (1, N, C)

        # -----------------------------------------------
        # Step 2: Compute the Full KL Divergence per sample.
        #
        # For each sample i and latent dimension j, compute:
        #   KL(q(z_j|x_i) || p(z_j)) = Σ[c=1 to C] q(z_j=c|x_i) * 
        #                                ( log(q(z_j=c|x_i)) - log(1/C) )
        #
        # This gives a tensor of shape (B, N) (KL per latent, per sample).
        # Then, sum over the latent dimensions (for each sample) to get the full KL per sample.
        # Finally, average over the batch to obtain the final full KL loss.
        # -----------------------------------------------
        kl_per_latent = (q_z_x * (log_q_z_x - log_uniform)).sum(dim=-1)  # Shape: (B, N)
        full_kl_batch = kl_per_latent.sum(dim=1)  # Sum over latent dimensions → Shape: (B,)


        # -----------------------------------------------
        # Step 3: Compute Dimension-wise KL (DWKL) from aggregated posteriors.
        #   DWKL_n = Σ[j=1 to N] E{q(z|x_n)}[log q(z_j)/p(z_j)]
        #          = Σ[j=1 to N] E{q(z_j |x_n)}[log q(z_j)/p(z_j)]   (others sum to 1)
        log_q_z_marg = torch.log(q_z_marginal + eps).unsqueeze(0)  # Shape: (1, N, C)

        # For each latent dimension j, using the running aggregated posterior marginal q_z_running,
        # compute:
        #   DWKL(j) = KL(q(z_j) || p(z_j)) = Σ[c=1 to C] q_z_x[j, c] *
        #                                   ( log(q_z[j, c]) - log(1/C) )
        #
        # This yields a tensor of shape (N,), one value per latent dimension.
        # Sum over all latent dimensions to yield the total DWKL.
        # -----------------------------------------------
        dwkl_per_latent  = (q_z_x * (log_q_z_marg - log_uniform)).sum(dim=-1)  # Shape: (B, N)
        dwkl_batch = dwkl_per_latent.sum(dim=-1)  #  (B,) 

        # -----------------------------------------------
        # Step 3: Compute Mutual Information (MI)
        # NOTE: As explained above, assume here that q(z) factorises into q(z_j)
        log_q_z = torch.log(q_z_marginal + eps)  # Shape: (N, C) (Assume q(z) factorises)

        # For each sample i and latent dimension j, compute:
        #   MI_component = KL(q(z_j|x_i) || q(z_j)) = Σ[c=1 to C] q(z_j=c|x_i) * 
        #                     ( log(q(z_j=c|x_i)) - log(q_z_running[j, c]) )
        #
        # q_z_running is the EMA estimate for the aggregated posterior.
        # This gives a tensor of shape (B, N).
        # Sum over latent dimensions for each sample, then average over the batch.
        # -----------------------------------------------
        mi_per_latent = (q_z_x * (log_q_z_x - log_q_z.unsqueeze(0))).sum(dim=-1)  # Shape: (B, N)
        mi_batch = mi_per_latent.sum(dim=1)  # Sum over latent → Shape: (B,)


        # -----------------------------------------------
        # Step 5: Compute Total Correlation (TC) loss.
        # Estimating TC directly is challenging because it requires the joint distribution of all latent codes.
        # But we cannot also use the approximation q(z) = prod_j q(z_j) as otherwise TC will vanish
        # We can solve for TC:
        #   TC_n = KL_n - MI_n - DWKL_n
        # -----------------------------------------------
        tc_batch = full_kl_batch - mi_batch - dwkl_batch  # (B,) for each sample
       
        # -----------------------------------------------
        # Return the computed losses (averaged per sample over batch) and the updated running aggregated posterior.
        # -----------------------------------------------


        return {
            "mi_loss": mi_batch.mean(), 
            "dwkl_loss": dwkl_batch.mean(), 
            "tc_loss": tc_batch.mean(), 
            "kl_loss": full_kl_batch.mean()
        }, q_z_marginal_detached


@dataclass
class GridDVAEConfig(Config):
    """Configuration class for GridDVAE model.
    
    Attributes:
        n_dim (int): Model dimension
        n_head (int): Number of attention heads
        n_base_layers (int): Number of base transformer layers
        n_latent_layers (int): Number of latent transformer layers
        n_codes (int): Number of discrete codes (must be power of 2)
        codebook_size (int): Size of codebook (default: 512)
        rope_base (int): Base for rotary position encoding (default: 10_000)
        dropout (float): Dropout probability (default: 0.0)
        max_grid_height (int): Maximum grid height (default: 32)
        max_grid_width (int): Maximum grid width (default: 32)
        n_vocab (int): Vocabulary size (default: 16)
        padding_idx (int | None): Index for padding token (default: n_vocab - 1)
        eos_idx (int | None): Index for end-of-sequence token (default: n_vocab - 2)
    """
    n_dim: int
    n_head: int
    n_grid_layer: int
    n_latent_layer: int
    n_codes: int
    codebook_size: int = 512
    rope_base: int = 10_000
    dropout: float = 0.0
    max_grid_height: int = 32
    max_grid_width: int = 32
    n_vocab: int = 16
    padding_idx: int | None = None
    mask_idx: int | None = None
    pad_weight: float = 0.01

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
        assert is_power_of_two(self.n_codes), "Number of codes must be a power of 2"
        assert self.n_pos % self.n_codes == 0, "Number of positions must be divisible by the number of codes"

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
        # Get all explicitly defined fields
        base_dict = {
            'n_dim': self.n_dim,
            'n_head': self.n_head,
            'n_grid_layer': self.n_grid_layer,
            'n_latent_layer': self.n_latent_layer,
            'n_codes': self.n_codes,
            'codebook_size': self.codebook_size,
            'rope_base': self.rope_base,
            'dropout': self.dropout,
            'max_grid_height': self.max_grid_height,
            'max_grid_width': self.max_grid_width,
            'n_vocab': self.n_vocab,
            'padding_idx': self.padding_idx,
            'mask_idx': self.mask_idx,
            'pad_weight': self.pad_weight
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


class GridDVAE(nn.Module):
    def __init__(self, config: GridDVAEConfig):
        super().__init__()
        self.config = config
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

        
        # Keep the base transformer blocks
        # self.encoder_base = Transformer(config, out_norm=False)
        self.grid_encoder = Transformer(
            d_model=config.n_dim,
            n_head=config.n_head,
            n_layer=config.n_grid_layer,
            out_norm=False
        )

        self.latent_encoder = LatentTransformer(
            n_latent=config.n_codes,
            d_model=config.n_dim,
            n_head=config.n_head,
            n_layer=config.n_latent_layer,
            out_norm=False
        )

        self.codebook = Codebook(config.n_dim, config.codebook_size)

        self.latent_decoder = LatentTransformer(
            n_latent=config.n_pos,
            d_model=config.n_dim,
            n_head=config.n_head,
            n_layer=config.n_latent_layer,
            out_norm=False
        )

        self.grid_decoder = Transformer(
            d_model=config.n_dim,
            n_head=config.n_head,
            n_layer=config.n_grid_layer,
            out_norm=True
        )

        self.decoder_head = nn.Linear(config.n_dim, config.n_vocab, bias=False)

        rows = torch.arange(config.max_grid_height, dtype=torch.long)
        cols = torch.arange(config.max_grid_width, dtype=torch.long)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
        grid_pos_indices = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).unsqueeze(0)
        latent_pos_indices = torch.arange(config.n_pos).unsqueeze(0)

        self.register_buffer("latent_pos_indices", latent_pos_indices, persistent=False)
        self.register_buffer('grid_pos_indices', grid_pos_indices, persistent=False)
        # self.apply(self._init_weights) # This initialisation seems to be terrible for overfitting.


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # if hasattr(module, 'RESCALE_INIT'):
            #     # 2 * (total_layer/2) because each encoder and decoder has a transformer block layers
            #     # and each transformer block has 2 residual connections.
            #     std *= self.config.total_layers ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    

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
    

    def apply_mask(self, x: Tensor, mask_percentage: float = 0.0) -> Tensor:
        mask = (torch.rand_like(x.float()) < mask_percentage) & (x != self.pad_value)
        x.masked_fill_(mask, self.mask_value)
        return x


    def forward(self, x: Tensor, q_z_marg: Optional[Tensor] = None, tau: float = 1.0, hardness: Tensor = torch.tensor(0.0), mask_percentage: float = 0.0, reinMax: bool = False) -> Tuple[Tensor, dict, Tensor]:
        B, S = x.size()
        x = self.apply_mask(x, mask_percentage)

        grid_pos_indices = self.grid_pos_indices.expand(B, -1, -1)
        latent_pos_indices = self.latent_pos_indices.expand(B, -1)

        encoded_logits = self.encode(x, grid_pos_indices, latent_pos_indices)

        encoded_logits, soft_code, code = self.codebook(encoded_logits, tau=tau, hardness=hardness, reinMax=reinMax)

        kld_losses, q_z_marg = self.codebook.kld_disentanglement_loss(soft_code, q_z_marg=q_z_marg, momentum=0.99, eps=1e-8)

        decoded_logits = self.decode(encoded_logits, grid_pos_indices, latent_pos_indices)
        
        ce_loss = self.reconstruction_loss(decoded_logits, x, pad_value=self.pad_value)

        losses = {
            "ce_loss": ce_loss/self.config.n_pos,  # Per sample per token
            **{k: v/self.config.n_codes for k, v in kld_losses.items()}, # Per sample per latent
        }
        return decoded_logits, soft_code, losses, q_z_marg

    def reconstruction_loss(self, decoded_logits: Tensor, x: Tensor, pad_value: int = -1, pad_weight: float = 0.01) -> Tensor:
        """
        Compute the reconstruction loss using cross-entropy per sample, with pad tokens weighted differently.

        Args:
            decoded_logits (Tensor): Predicted logits of shape [B, S, V]
            x (Tensor): Target tokens of shape [B, S]
            pad_value (int): Token value for padding tokens
            pad_weight (float): Weight for pad token loss (default: 0.01 = 1% of normal weight)

        Returns:
            Tensor: Average reconstruction loss per sample, with weighted pad tokens
        """
        # Create a weight tensor where pad tokens have pad_weight and others have 1.0
        weights = torch.where(x == pad_value, 
                            torch.full_like(x, pad_weight, dtype=decoded_logits.dtype),
                            torch.ones_like(x, dtype=decoded_logits.dtype))
        
        # Compute per-token cross entropy loss
        per_token_loss = F.cross_entropy(
            decoded_logits.view(-1, decoded_logits.size(-1)),
            x.view(-1),
            reduction='none'
        )
        
        # Apply weights and average per sample
        weighted_loss = (per_token_loss * weights.view(-1)).sum() / x.size(0)
        
        return weighted_loss
    





#%%
