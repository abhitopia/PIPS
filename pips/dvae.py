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

# Define the RotaryPositionalEmbeddings class
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


# Define the RoPE2D class
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


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


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


class SelfAttention(nn.Module):
    def __init__(self, config: Config, rope: Optional[RoPE2D]=None):
        super().__init__()
        self.config = config
        self.rope = rope
        assert config.n_dim % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_dim, 3 * config.n_dim, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_dim, config.n_dim, bias=False)

        # regularization
        self.n_head = config.n_head
        self.n_dim = config.n_dim
        self.dropout = config.dropout

    def forward(self,
            x: Tensor, 
            attn_mask: Optional[Tensor],
            positions: Optional[Tensor] = None,
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
        B, T, C = x.size()
        qkv = self.c_attn(x)        # qkv: (B, T, 3 * C)
        q, k, v = qkv.split(self.n_dim, dim=2)

        # Reshape for multi-head attention, but do not transpose yet!
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  

        # Apply Rope2D to q and k
        if self.rope is not None and positions is not None:
            k = self.rope(k, positions.unsqueeze(1))
            q = self.rope(q, positions.unsqueeze(1))

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
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # Expand to [B, 1, S, S]

        # attn_output: (B, n_head, T, head_dim)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)

        # Reshape back to (B, T, C)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.c_proj(attn_output)

        # Zero out NaN values, so they don't affect future computations
        # I have also verified that the it doesn't matter what the nan values are set to
        # y = torch.nan_to_num(y, nan=0.0)

        return y, new_kv_cache
    

class TransformerBlock(nn.Module):
    def __init__(self, config: Config, rope: Optional[RoPE2D]):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.rmsnorm = RMSNorm(config.n_dim)
        self.attn = SelfAttention(config, rope)
        self.normed_mlp = nn.Sequential(
                            RMSNorm(config.n_dim),
                            SwiGLUFFN(config.n_dim, 4 * config.n_dim))

    def forward(self, x: Tensor, 
            attn_mask: Optional[Tensor], 
            positions: Optional[Tensor] = None,
            kv_cache: Optional[Tuple[Tensor, Tensor]] = None, 
            return_kv_cache: bool = False) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:

        attn_output, new_kv_cache = self.attn(self.rmsnorm(x), attn_mask=attn_mask, positions=positions, kv_cache=kv_cache, return_kv_cache=return_kv_cache)
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.normed_mlp(x))
        return x, new_kv_cache
    

class Transformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        rope_2d = RoPE2D(config.n_dim // config.n_head,
                        max_height=config.max_grid_height,
                        max_width=config.max_grid_width,
                        base=config.rope_base)
        self.blocks = nn.ModuleList([TransformerBlock(config, rope=rope_2d) for _ in range(config.n_layers)])
        self.rms_out = RMSNorm(config.n_dim)


    def forward(self,
            x: Tensor, 
            attn_mask: Tensor, 
            positions: Tensor, 
            kv_cache: Optional[List[Tuple[Tensor, Tensor]]] = None,
            return_kv_caches: bool = False
        ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:

        loop_kv_caches: List[Tuple[Tensor, Tensor]] = []

        for i, block in enumerate(self.blocks):
            x, new_kv_cache = block( x, 
                attn_mask=attn_mask,
                positions=positions, 
                kv_cache=kv_cache[i] if kv_cache is not None else None,
                return_kv_cache=return_kv_caches
            )

            # Ensure new_kv_cache is not None before appending
            if return_kv_caches and new_kv_cache is not None:
                loop_kv_caches.append(new_kv_cache)

        x = self.rms_out(x)
        return x, loop_kv_caches
    

class AttentionPool(nn.Module):
    """
    Transforms variable-length sequences to fixed-length sequences using attention.
    Input: BxSxD -> Output: BxKxD, where K is a fixed number of learned queries.
    """
    def __init__(self, dim: int, num_queries: int):
        super().__init__()
        self.num_queries = num_queries
        self.dim = dim
        
        # Learned query vectors
        self.queries = nn.Parameter(torch.randn(num_queries, dim))
        
        # Projection layers
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape [B, S, D]
            attn_mask (Optional[Tensor]): Attention mask of shape [1, 1, S], [B, 1, S], or [B, K, S]
        Returns:
            Tensor: Output tensor of shape [B, K, D]
        """
        B, S, D = x.shape
        assert D == self.dim, f"Expected input dimension {self.dim}, got {D}."

        # Project the queries, keys, and values
        q = self.q_proj(self.queries).unsqueeze(0)  # [1, K, D]
        k = self.k_proj(x)  # [B, S, D]
        v = self.v_proj(x)  # [B, S, D]

        # Check attn_mask dimensions if provided
        if attn_mask is not None:
            assert attn_mask.shape in [(1, 1, S), (B, 1, S), (B, self.num_queries, S)], \
                f"Expected attn_mask shape [(1, 1, S), (B, 1, S), (B, {self.num_queries}, S)], got {attn_mask.shape}"

        # Compute attention
        attn_output = F.scaled_dot_product_attention(
            q,                          # [1, K, D]
            k,                          # [B, S, D]
            v,                          # [B, S, D]
            attn_mask=attn_mask,        # Apply attention mask if provided
            scale=self.dim ** -0.5
        )  # [B, K, D]
        
        return self.out_proj(attn_output)
    

class StackedPooling(nn.Module):
    """
    Applies multiple AttentionPool blocks in succession, reducing or increasing sequence length
    from S -> S1 -> S2 -> ... -> SK step by step.
    """
    def __init__(self, dim: int, pool_sizes: List[int]):
        """
        Args:
            dim (int): Embedding dimension.
            pool_sizes (List[int]): The output sequence lengths for each compression stage.
                                    Example: [256, 64, 8]
                                    means Stage1: S -> 256, Stage2: 256 -> 64, Stage3: 64 -> 8.
        """
        super().__init__()
        self.dim = dim
        self.pool_sizes = pool_sizes
        
        # Create one AttentionPool module for each size in pool_sizes
        self.pool_layers = nn.ModuleList([
            AttentionPool(dim, num_queries=size) for size in pool_sizes
        ])

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): shape [B, S, D] input to the first stage
            attn_mask (Optional[Tensor]): Attention mask of shape [1, 1, S] or [B, 1, S] to be applied to the first AttentionPool
        Returns:
            Tensor: shape [B, final_size, D] after all stages
        """
        B, S, D = x.shape

        # Assert correct mask shape if provided
        if attn_mask is not None:
            assert attn_mask.shape in [(1, 1, S), (B, 1, S)], f"Expected attn_mask shape [(1, 1, S), (B, 1, S)], got {attn_mask.shape}"

        for i, pool in enumerate(self.pool_layers):
            if i == 0 and attn_mask is not None:
                # Apply the attention mask only to the first AttentionPool
                x = pool(x, attn_mask=attn_mask)
            else:
                x = pool(x)  # compress from current S -> next S'
        return x



@dataclass
class GridDVAEConfig(Config):
    n_dim: int
    n_head: int
    n_layers: int
    n_codes: int  # Directly specify the number of codes
    codebook_size: int = 512
    rope_base: int = 10_000
    dropout: float = 0.0
    max_grid_height: int = 32  # New default value
    max_grid_width: int = 32   # New default value
    n_vocab: int = 16

    def __post_init__(self):
        if self.n_dim % self.n_head != 0:
            raise ValueError("n_dim must be divisible by n_head")
        
        C = self.n_dim // self.n_head
        assert C % 2 == 0, "n_dim // n_head must be divisible by 2"

        head_dim = C // 2  # Actual Head Dimension. 

        # This is to ensure Rope2D can be applied
        assert head_dim % 2 == 0, "Head dimension must be even"

        # Calculate n_pos from grid dimensions
        self.n_pos = self.max_grid_height * self.max_grid_width
        
        assert is_power_of_two(self.n_pos), "Product of max_grid_height and max_grid_width must be a power of 2"
        assert is_power_of_two(self.n_codes), "Number of codes must be a power of 2"
        assert self.n_pos % self.n_codes == 0, "Number of positions must be divisible by the number of codes"

        self.compression_factor = self.n_pos // self.n_codes
        self.pool_sizes = [int(self.n_pos / (2**i)) for i in range(int(math.log2(self.compression_factor)) + 1)]

    def __repr__(self) -> str:
        attrs = [f"{key}={getattr(self, key)}" for key in self.__annotations__.keys()]
        computed_attrs = [
            f"n_pos={self.n_pos}",
            f"compression_factor={self.compression_factor}",
            f"pool_sizes={self.pool_sizes}"
        ]
        all_attrs = attrs + computed_attrs
        return f"DVAEConfig({', '.join(all_attrs)})"



def gumbel_softmax(logits: Tensor, tau: float=1, hard: bool=False, dim: int=-1) -> Tensor:
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

	"""

    if tau <= 0:
        raise ValueError("Temperature must be positive")
    

    gumbels = -torch.empty_like(logits).exponential_().log()  # # Generates Gumbel(0,1) noise
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau) Applies temperature scaling
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class GridDVAE(nn.Module):
    def __init__(self, config: GridDVAEConfig):
        super().__init__()
        self.config = config
        self.n_pos = config.n_pos
        self.embd = nn.Embedding(config.n_vocab, config.n_dim)
        self.encoder_base = Transformer(config=config)
        self.encoder_bottleneck = StackedPooling(dim=config.n_dim, pool_sizes=config.pool_sizes[1:])
        self.encoder_head = nn.Linear(config.n_dim, config.codebook_size)
        self.codebook = nn.Parameter(torch.randn(config.codebook_size, config.n_dim))
        self.decoder_bottleneck = StackedPooling(dim=config.n_dim, pool_sizes=config.pool_sizes[::-1][1:])
        self.decoder_base = Transformer(config=config)
        self.decoder_head = nn.Linear(config.n_dim, config.n_vocab, bias=False)
        pos_indices = self.create_grid_position_tensor(
                            config.max_grid_height,
                            config.max_grid_width, 
                            requires_grad=False).unsqueeze_(0)

        # persistent=False prevents it from saved to statedict
        self.register_buffer('pos_indices', pos_indices, persistent=False)

        self.q_z_marg = None

    @staticmethod
    def create_grid_position_tensor(height: int, width: int, requires_grad=True) -> torch.Tensor:
        """
        Creates a position tensor for a grid of size height x width.
        Returns tensor of shape (height*width, 2) in row-major order.
        
        Args:
            height (int): Number of rows
            width (int): Number of columns
            requires_grad (bool): Whether the tensor requires gradients
            
        Returns:
            Tensor: Position tensor of shape (height*width, 2) containing (row, col) indices,
                with dtype=torch.long
        """
        rows = torch.arange(height, dtype=torch.long)
        cols = torch.arange(width, dtype=torch.long)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
        positions = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)
        
        # Convert to float only if requires_grad is True
        if requires_grad:
            positions = positions.float().requires_grad_(True)
        
        return positions

    def create_random_mask(self, B: int, S: int, mask_percentage: float, same_mask_for_all: bool = False) -> Optional[Tensor]:
        """
        Creates a random boolean mask for the input sequence.

        Args:
            B (int): Batch size.
            S (int): Sequence length.
            mask_percentage (float): Fraction of tokens to mask (set to False).
            same_mask_for_all (bool): If True, apply the same mask to all samples in the batch.

        Returns:
            Optional[Tensor]: A boolean mask of shape [B, 1, S] or [1, 1, S] with True for unmasked and False for masked tokens,
                              or None if no masking is applied.
        """
        # if mask_percentage == 0:
        #     return None  # No masking
        assert mask_percentage < 1, "mask_percentage of 1 would mask all tokens, which is not allowed."


        device = self.pos_indices.device if hasattr(self, "pos_indices") else torch.device("cpu")

        if same_mask_for_all:
            mask = torch.rand(1, 1, S, device=device) > mask_percentage
        else:
            mask = torch.rand(B, 1, S, device=device) > mask_percentage
        return mask

    def encode(self, x: Tensor, attn_mask: Optional[Tensor] = None, tau: float = 0.9, hard: bool = True, reinMax: bool = False) -> Tensor:
        if reinMax:
            assert hard, "ReinMax requires hard sampling"

        B, S = x.size()

        # Convert into x: (B, S, D) using self.embd
        x = self.embd(x)

        # Ensure the position indices tensor is on the same device as x
        positions = self.pos_indices.to(x.device).expand(B, -1, -1)

        assert S == self.n_pos, f"Input Sequence must be of length {self.n_pos}"

        # Pass to the encoder network with the attention mask
        encoded, _ = self.encoder_base(x, attn_mask, positions)

        # Compress to Codebook Space with the attention mask
        encoded_compressed = self.encoder_bottleneck(encoded, attn_mask)

        # Map n_codes to logits
        encoded_logits = self.encoder_head(encoded_compressed)
        # Use gumbel softmax to sample from the Codebook
        soft_code = F.gumbel_softmax(encoded_logits, tau=tau, hard=False)

        hard_code = soft_code  # Default to soft code
        if hard:
            # Straight through.
            index = soft_code.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(
                encoded_logits, memory_format=torch.legacy_contiguous_format
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
                pi_1 = F.softmax((pi_1.log() - encoded_logits).detach() - encoded_logits, dim=-1)
                pi_2 = 2 * pi_1 - 0.5 * pi_0
                hard_code = pi_2 - pi_2.detach() + D
            else:
                hard_code = y_hard - soft_code.detach() + soft_code

        return hard_code, soft_code

    def decode(self, code: Tensor) -> Tensor:
        B, n_codes, _ = code.size()

        # Lookup Codebook - Matrix multiplication between one-hot codes and codebook
        code_words = torch.einsum('bnc,cd->bnd', code, self.codebook)  # (B, n_codes, n_dim)

        # Decompress from Codebook Space
        z_prime = self.decoder_bottleneck(code_words)

        # Ensure that pos_indices is on the same device as the code tensor.
        positions = self.pos_indices.to(code.device).expand(B, -1, -1)

        # Pass through decoder network
        decoded, _ = self.decoder_base(z_prime, None, positions)

        # Convert to logits
        decoded_logits = self.decoder_head(decoded)

        return decoded_logits

    def forward(self, x: Tensor, tau: float = 0.9, hard: bool = True, mask_percentage: float = 0.0, reinMax: bool = False) -> Tensor:
        # Create a random boolean mask
        attn_mask = self.create_random_mask(x.size(0), x.size(1), mask_percentage, same_mask_for_all=True)
        code, soft_code = self.encode(x, attn_mask, tau, hard, reinMax)

        # Compute the KL disentanglement loss using the internal q_z_running
        kld_losses = self.kld_disentanglement_loss(soft_code)

        # Compute the reconstruction loss
        decoded_logits = self.decode(code)
        reconstruction_loss = self.reconstruction_loss(decoded_logits, x)

        # Return the reconstruction loss and the disentanglement losses
        return decoded_logits, reconstruction_loss, kld_losses
    

    def reconstruction_loss(self, decoded_logits: Tensor, x: Tensor) -> Tensor:
        """
        Compute the reconstruction loss using cross-entropy per sample (and not per token)
        """
        return F.cross_entropy(decoded_logits.view(-1, decoded_logits.size(-1)), x.view(-1), reduction='sum') / x.size(0)
    

    def kld_disentanglement_loss(self, q_z_x, momentum=0.99, eps=1e-8):
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

        = Σ{z} [q(z|x_i) * (log q(z|x_i)/log q(z) + log q(z)/∏ q(z_j) + log ∏ q(z_j)/p(z_j) )]

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
            code_soft (Tensor): Soft one-hot distributions from Gumbel-softmax, of shape (B, N, C), where:
                B = batch size,
                N = number of discrete latent codes,
                C = codebook size.
            q_z_running (Tensor, optional): The running estimate of the aggregated posterior q(z) 
                from previous minibatches, of shape (N, C). If None, it will be initialized to the current batch's mean.
            momentum (float): The decay rate for the EMA; typical values are near 0.99.
            eps (float): A small constant for numerical stability.

        Returns:
            mi_loss (Tensor): Average Mutual Information loss per sample (sum over latents, then averaged over batch).
            dwkl_loss (Tensor): Total Dimension-wise KL loss computed from the EMA aggregated posterior (summed over latents).
            tc_loss (Tensor): Total Correlation loss: TC = Full KL - MI - DWKL.
            full_kl_loss (Tensor): Average full KL loss per sample (sum over latents, then averaged over batch).
            q_z_running (Tensor): The updated EMA estimate of the aggregated posterior q(z).
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

        q_z_marginal = ( momentum * self.q_z_marg + (1 - momentum) * q_z_marg_batch ) if self.q_z_marg is not None else q_z_marg_batch
        self.q_z_marg = q_z_marginal.detach()                                        # Detach the local variable from the graph


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
            "mi_loss": F.relu(mi_batch.mean()), 
            "dwkl_loss": F.relu(dwkl_batch.mean()), 
            "tc_loss": F.relu(tc_batch.mean()), 
            "kl_loss": F.relu(full_kl_batch.mean())
        }



#%%
