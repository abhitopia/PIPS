from functools import partial
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from torch import Tensor
from pips.grid_dataset import GridDataset
from torch.amp import autocast


# -----------------------------
# 1) A Simple Multihead Cross-Attn + Feedforward Block
# -----------------------------

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


class CrossAttnBlock(nn.Module):
    """
    A single Transformer-style block that:
      - Performs multi-head cross-attention (Q from 'queries', K/V from 'context').
      - Applies a feedforward MLP.
      - Uses residual connections + LayerNorm.
    We assume batch_first=True shape conventions: (B, T, D).
    """
    def __init__(self, d_model, n_head, dim_feedforward=512, dropout=0.0, rope=None):
        super().__init__()
        # self.cross_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout, rope=rope)
        self.norm_context = RMSNorm(d_model)
        self.norm_queries = RMSNorm(d_model)
        # self.ff = nn.Sequential(
        #     nn.Linear(d_model, dim_feedforward),
        #     nn.ReLU(),
        #     nn.Linear(dim_feedforward, d_model)
        # )
        self.ff = SwiGLUFFN(d_model, dim_feedforward)
        self.norm2 = RMSNorm(d_model)

    def forward(self, queries, context, positions: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        """
        queries: (B, Tq, D)
        context: (B, Tc, D)
        Returns: (B, Tq, D)
        """
        # Cross-attention: Q = queries, K=V = context
        normed_context = self.norm_context(context)
        normed_queries = self.norm_queries(queries)
        attn_out, _ = self.cross_attn(normed_queries, normed_context, normed_context, positions=positions, attn_mask=attn_mask)  # shape (B, Tq, D)
        x = queries + attn_out

        # Feed-forward
        ff_out = self.ff(self.norm2(x))  # shape (B, Tq, D)
        # out = self.norm2(x + ff_out)
        return ff_out

# -----------------------------
# 2) Latent Encoder
#    - 32 learnable tokens cross-attend over the full 1024-token input.
# -----------------------------
class LatentEncoder(nn.Module):
    def __init__(self,
                 d_model=64,
                 n_latent=32,
                 n_head=4,
                 num_layers=2,
                 dim_feedforward=256):
        super().__init__()
        # Learnable latent tokens: shape (1, n_latent, d_model)
        self.latent_tokens = nn.Parameter(torch.randn(1, n_latent, d_model))


        rope = RotaryPositionalEmbeddings(
            dim=d_model // n_head,
            max_seq_len=1024,
            base=10_000
        )

        positions = torch.arange(rope.max_seq_len).unsqueeze(0)

        self.register_buffer("positions", positions, persistent=False)
        # Stack of cross-attn blocks
        self.blocks = nn.ModuleList([
            CrossAttnBlock(d_model, n_head, dim_feedforward, rope=rope)
            for _ in range(num_layers)
        ])

        self.rms_norm = RMSNorm(d_model)

    def forward(self, x):
        """
        x: (B, S, d_model) = input embeddings for S=1024 tokens
        Return: (B, n_latent, d_model)
        """
        B = x.size(0)

        # Broadcast the learnable latents to match batch size
        latents = self.latent_tokens.expand(B, -1, -1)  # (B, n_latent, d_model)

        positions = self.positions.expand(B, -1)
        for block in self.blocks:
            latents = block(latents, x, positions=positions)  # cross-attn with input x

        latents = self.rms_norm(latents)
        return latents

# -----------------------------
# 4) Full Autoencoder: Embedding -> LatentEncoder -> LatentDecoder -> Output
# -----------------------------
class BottleneckAutoencoder(nn.Module):
    def __init__(self,
                 vocab_size=16,
                 d_model=64,
                 n_latent=32,
                 n_head=4,
                 num_enc_layers=2,
                 num_dec_layers=2):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        # Input embedding for tokens
        self.embed = nn.Embedding(vocab_size, d_model)

        # Encoder
        self.encoder = LatentEncoder(d_model=d_model,
                                     n_latent=n_latent,
                                     n_head=n_head,
                                     num_layers=num_enc_layers,
                                     dim_feedforward=4*d_model)  # e.g. 256 if d_model=64

        # Decoder
        self.decoder = LatentEncoder(d_model=d_model,
                                     n_latent=1024,
                                     n_head=n_head,
                                     num_layers=num_dec_layers,
                                     dim_feedforward=4*d_model)

        # Final projection back to vocab size
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        x: (B, 1024) of token indices
        Returns: (B, 1024, vocab_size)
        """
        # 1. Embed input tokens
        emb = self.embed(x)  # (B, 1024, d_model)

        # 2. Latent encoding
        latents = self.encoder(emb)  # (B, 32, d_model)

        # 3. Latent decoding
        decoded_emb = self.decoder(latents)  # (B, 1024, d_model)

        # 4. Project to vocab logits
        logits = self.output_proj(decoded_emb)  # (B, 1024, vocab_size)

        return logits

# -----------------------------
# 5) Simple Training Loop on Random Data
# -----------------------------
def main():
    # Hyperparameters
    BATCH_SIZE = 8
    SEQ_LEN = 1024
    VOCAB_SIZE = 16
    D_MODEL = 64
    N_LATENT = 16
    BASE_LR = 1e-3
    WARMUP_STEPS = 100
    MAX_STEPS = 1000
    MIN_LR_FACTOR = 0.01  # Final LR will be BASE_LR * MIN_LR_FACTOR

    max_size = SEQ_LEN
    padding_idx = 15
    eos_idx = 14
    batch_size = BATCH_SIZE
    permute_train = False


    # Create training dataloader
    collate_fn_train = partial(GridDataset.collate_fn, 
                             pad_value=padding_idx, 
                             eos_value=eos_idx,  # Add eos_value parameter
                             permute=permute_train,  # Use the permute_train parameter
                             max_size=max_size
                             )
    train_dataset = GridDataset(train=True)

    # num_workers = min(8, os.cpu_count() or 1)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn_train,
        shuffle=permute_train,  ## True if training only if permute_train is True
        num_workers=0,
        persistent_workers=False,
        # worker_init_fn=worker_init_fn,
        drop_last=True
    )

    for idx, batch in enumerate(train_loader):
        x, _ = batch
        if idx == 3:
            break

    print("batch:", x)
    # Create random data: shape (B, SEQ_LEN) of token indices
    # Each entry is in [0..VOCAB_SIZE-1]
    # x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    # Model
    model = BottleneckAutoencoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_latent=N_LATENT,
        n_head=4,
        num_enc_layers=2,
        num_dec_layers=2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR)
    
    # Learning rate schedule with warmup and cosine decay
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            # Linear warmup
            return float(step) / float(max(1, WARMUP_STEPS))
        else:
            # Cosine decay from 1.0 to MIN_LR_FACTOR
            progress = float(step - WARMUP_STEPS) / float(max(1, MAX_STEPS - WARMUP_STEPS))
            return MIN_LR_FACTOR + 0.5 * (1.0 - MIN_LR_FACTOR) * (1.0 + np.cos(np.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)

    model.train()

    for step in range(MAX_STEPS):
        optimizer.zero_grad()

        # Forward
        logits = model(x)  # (B, 1024, vocab_size)

        # Reshape for cross-entropy:
        # CrossEntropyLoss expects: (B*T, vocab_size) vs (B*T) for targets
        loss = criterion(logits.view(-1, VOCAB_SIZE), x.view(-1))

        # Calculate accuracy (excluding padding tokens)
        predictions = logits.argmax(dim=-1)  # (B, 1024)
        mask = (x != padding_idx)  # Create mask to exclude padding tokens
        correct = ((predictions == x) * mask).sum()
        total = mask.sum()
        accuracy = (correct / total).item() * 100  # Convert to percentage

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step {step}/{MAX_STEPS}, Loss: {loss.item():.4f}, "
                  f"Accuracy: {accuracy:.2f}%, LR: {current_lr:.6f}")

if __name__ == "__main__":
    main()
