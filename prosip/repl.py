from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import Tensor, autocast
import torch.nn as nn
from prosip.utils import get_activation_fn
from torch.nn import functional as F



def _build_mlp(input_dim, output_dim, num_hidden_layers, hidden_dim, activation, dropout):
    layers = []
    assert num_hidden_layers >= 0, "Number of hidden layers must be non-negative"

    # First hidden layer.
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(activation())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    # Additional hidden layers if required.
    for _ in range(num_hidden_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    # Final output layer.
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


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
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        return device

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


class HyperNetwork(nn.Module):
    def __init__(self, task_embedding_dim, target_dim_A, target_dim_B, rank, mlp_layers, mlp_dim, activation, dropout):
        """
        Args:
            task_embedding_dim (int): Size of the task embedding.
            target_dim_A (int): Output dimension for A matrix (typically ffn_dim).
            target_dim_B (int): Input dimension for B matrix (typically embed_dim).
            rank (int): Rank for the low-rank factorization.
            mlp_layers (int): Number of hidden layers in the MLP.
            mlp_dim (int): Dimension of the hidden layers.
            activation (callable): Activation function class (e.g. nn.ReLU).
            dropout (float): Dropout rate between layers.
        """
        super(HyperNetwork, self).__init__()
        self.target_dim_A = target_dim_A
        self.target_dim_B = target_dim_B
        self.rank = rank

        # Build two MLPs to generate the parameters for the low-rank matrices.
        self.fc_A = _build_mlp(task_embedding_dim, target_dim_A * rank,
                                    mlp_layers, mlp_dim, activation, dropout)
        self.fc_B = _build_mlp(task_embedding_dim, rank * target_dim_B,
                                    mlp_layers, mlp_dim, activation, dropout)


    def forward(self, task_embedding):
        """
        Args:
            task_embedding: Tensor of shape [batch_size, task_embedding_dim]
        Returns:
            A_params: Tensor of shape [batch_size, target_dim_A, rank]
            B_params: Tensor of shape [batch_size, rank, target_dim_B]
        """
        A = self.fc_A(task_embedding)  # [batch_size, target_dim_A * rank]
        B = self.fc_B(task_embedding)  # [batch_size, rank * target_dim_B]
        A_params = A.view(-1, self.target_dim_A, self.rank)
        B_params = B.view(-1, self.rank, self.target_dim_B)
        return A_params, B_params


class DynamicLoRALinear(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        subroutine_dim: int,
        rank: int,
        mlp_layers: int,
        mlp_dim: int,
        activation: str,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.rank = rank
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.subroutine_dim = subroutine_dim

        self.base = nn.Linear(dim_in, dim_out, bias=bias)
        if rank > 0:
            self.hyper = HyperNetwork(
                task_embedding_dim=self.subroutine_dim,
                target_dim_A=dim_out,
                target_dim_B=dim_in,
                rank=rank,
                mlp_layers=mlp_layers,
                mlp_dim=mlp_dim,
                activation=get_activation_fn(activation, return_module=True),
                dropout=dropout,
            )

    def forward(self, x: Tensor, subroutine: Optional[Tensor] = None) -> Tensor:
            # canonicalize both 2-D and 3-D into a [B, T, D] problem
            base_out = self.base(x)
            if subroutine is None or self.rank == 0:
                return base_out

            is_flat = (base_out.dim() == 2)
            if is_flat:
                # assume x: [BT, D_in]; treat as B=BT, T=1
                base_out = base_out.unsqueeze(1)                # → [BT, 1, D_in]
                subroutine = subroutine.unsqueeze(1)

            # now both x and (if present) subroutine are [B, T, D_in]
            B, T, D = subroutine.shape

            flat_sub = subroutine.view(B*T, D)
            A_raw, B_raw = self.hyper(flat_sub)
            A = A_raw.view(B, T, self.dim_out, self.rank)
            Bm= B_raw.view(B, T, self.rank, self.dim_in)

            inter = torch.einsum('bti,btri->btr', x, Bm)
            adapt = torch.einsum('btr,btor->bto', inter, A)
            out = base_out + adapt
            
            # if we started flat, squeeze back to [BT, D_out]
            if is_flat:
                out = out.reshape(-1, self.dim_out)

            return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_dim, n_head, dropout, rank=0, rope=None, mlp_layers=1, mlp_dim=128, activation="gelu"):
        super().__init__()
        self.rope = rope
        self.n_dim = n_dim
        self.n_head = n_head
        self.dropout = dropout
        assert n_dim % n_head == 0


        lora_kwargs = {
            "rank": rank,
            "subroutine_dim": n_dim,
            "mlp_layers": mlp_layers,
            "mlp_dim": mlp_dim,
            "activation": activation,
            "dropout": dropout
        }
        # key, query, value projections for all heads, but in a batch
        self.q_proj = DynamicLoRALinear(n_dim, n_dim, **lora_kwargs)
        self.k_proj = DynamicLoRALinear(n_dim, n_dim, **lora_kwargs)
        self.v_proj = DynamicLoRALinear(n_dim, n_dim, **lora_kwargs)
        self.c_proj = DynamicLoRALinear(n_dim, n_dim, **lora_kwargs)

    def forward(self,
            q: Tensor, 
            k: Tensor,
            v: Tensor,
            subroutine: Optional[Tensor] = None,
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
        
        q = self.q_proj(q, subroutine)
        v = self.v_proj(v, subroutine)
        k = self.k_proj(k, subroutine)

        # Reshape for multi-head attention, but do not transpose yet!
        q = q.view(B, qT, self.n_head, D // self.n_head).transpose(1, 2) # (B, n_head, T, head_dim)
        k = k.view(B, kT, self.n_head, D // self.n_head).transpose(1, 2) 
        v = v.view(B, vT, self.n_head, D // self.n_head).transpose(1, 2)  

        # Apply Rope2D to q and k
        if self.rope is not None:
            positions = torch.arange(max(kT, qT), device=k.device).unsqueeze(0).expand(B, -1)
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
        y = self.c_proj(attn_output, subroutine)

        # Zero out NaN values, so they don't affect future computations
        # I have also verified that the it doesn't matter what the nan values are set to
        # y = torch.nan_to_num(y, nan=0.0)

        return y, new_kv_cache


class DynamicTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, rank, num_heads, mlp_layers, mlp_dim,
                 activation, dropout=0.0, rope=None, ffn_dim=None):
        """
        Args:
            embed_dim (int): Embedding dimension for the transformer.
            ffn_dim (int): Inner dimension of the feed-forward network.
            task_embedding_dim (int): Dimension for task embeddings.
            rank (int): Low-rank dimension for dynamic LoRA.
            num_heads (int): Number of attention heads.
            activation (str): Activation function name (e.g. "relu", "gelu").
            dropout (float): Dropout rate.
        """
        super(DynamicTransformerEncoderLayer, self).__init__()
        self.embed_dim = embed_dim

        lora_kwargs = {
            "rank": rank,
            "mlp_layers": mlp_layers,
            "mlp_dim": mlp_dim,
            "activation": activation,
            "dropout": dropout
        }

        if ffn_dim is None:
            ffn_dim = embed_dim * 4
        
        self.self_attn = MultiHeadAttention(n_dim=embed_dim, 
                                            n_head=num_heads, 
                                            rope=rope,
                                            **lora_kwargs)
                                            
        self.norm1 = nn.RMSNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-Forward Network components.
        self.ffn_linear1 = DynamicLoRALinear(embed_dim, ffn_dim, subroutine_dim=embed_dim, **lora_kwargs, bias=False)
        self.ffn_linear2 = DynamicLoRALinear(ffn_dim, embed_dim, subroutine_dim=embed_dim, **lora_kwargs, bias=False)
        # Using a helper to get the activation module.
        self.activation = get_activation_fn(activation, return_module=True)()
        self.norm2 = nn.RMSNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, subroutine):
        """
        Args:
            src: Tensor of shape [B, seq_len, embed_dim]
            task_embedding: Tensor of shape [B, seq_len, task_embedding_dim]
                           (i.e. each token has its own task embedding)
        Returns:
            Tensor of shape [B, seq_len, embed_dim]
        """
        # --- Self-Attention Sublayer with Pre-Norm ---
        normed_src = self.norm1(src)
        attn_output, _ = self.self_attn(normed_src, normed_src, normed_src, subroutine=subroutine)
        src = src + self.dropout1(attn_output)

        # --- Feed-Forward Sublayer with Dynamic LoRA and Pre-Norm ---
        out_ffn1 = self.ffn_linear1(self.norm2(src), subroutine)
        out_activated = self.activation(out_ffn1)
        out_ffn2 = self.ffn_linear2(out_activated, subroutine)
        src = src + self.dropout2(out_ffn2)
        return src


class SubroutineExecutor(nn.Module):
    def __init__(self, num_layers, embed_dim, rank, num_heads, mlp_layers, mlp_dim,
                 activation, dropout=0.0, rope=None):
        """
        Args:
            num_layers (int): Number of transformer encoder layers.
            embed_dim (int): Embedding dimension.
            rank (int): Low-rank dimension for dynamic LoRA.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(SubroutineExecutor, self).__init__()

   
        self.layers = nn.ModuleList([
            DynamicTransformerEncoderLayer(embed_dim=embed_dim, 
                                           rank=rank, 
                                           num_heads=num_heads, 
                                           activation=activation,
                                           dropout=dropout,
                                           mlp_layers=mlp_layers,
                                           mlp_dim=mlp_dim,
                                           rope=rope)
            for _ in range(num_layers)
        ])
        # self.norm = nn.RMSNorm(embed_dim)
        self.norm = nn.Identity()

    def forward(self, state, subroutine):
        """
        Args:
            state: Tensor of shape [batch_size, seq_len, embed_dim]
            subroutine: Tensor of shape [batch_size, seq_len, embed_dim]
        Returns:
            Processed next state tensor of shape [batch_size, seq_len, embed_dim]
        """
        for layer in self.layers:
            state = layer(state, subroutine)
        return self.norm(state)
    

class SubroutineGeneratorLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, activation, ffn_dim=None, dropout=0.0, rope=None):
        super(SubroutineGeneratorLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_dim=embed_dim, n_head=num_heads, dropout=dropout, rope=rope, rank=0)
        self.norm_kv = nn.RMSNorm(embed_dim)
        self.norm_q = nn.RMSNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        if ffn_dim is None:
            ffn_dim = embed_dim * 4

        # Feed-Forward Network components.
        self.ffn_linear1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.ffn_linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.activation = get_activation_fn(activation, return_module=True)()
        self.norm2 = nn.RMSNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, state, program, iter_embd):
        # state: BxSxD | program: BxD | iter_embd: BxD

        # Concatenate program and iteration embeddings
        queries = self.norm_q(state) # BxSxD
        keys = self.norm_kv(torch.cat([program.unsqueeze(1), iter_embd.unsqueeze(1)], dim=1)) # Bx2xD
        values = keys


        # --- Self-Attention Sublayer with Pre-Norm ---
        attn_output, _ = self.self_attn(queries, keys, values) # BxSxD
        # Add the residual connection.
        state = state + self.dropout1(attn_output) # BxSxD

        # --- Feed-Forward Sublayer with Pre-Norm ---
        out_ffn1 = self.ffn_linear1(self.norm2(state)) # BxSxD
        out_activated = self.activation(out_ffn1) # BxSxD
        out_ffn2 = self.ffn_linear2(out_activated) # BxSxD
        # Add residual connection.
        subroutine = state + self.dropout2(out_ffn2) # BxSxD 
        return subroutine
    

class SubroutineGenerator(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, activation, max_iterations, ffn_dim=None, dropout=0.0, rope=None):
        super(SubroutineGenerator, self).__init__()

        self.iteration_embedding = nn.Embedding(max_iterations, embed_dim)

        self.layers = nn.ModuleList([
            SubroutineGeneratorLayer(embed_dim=embed_dim, 
                                     num_heads=num_heads, 
                                     activation=activation, 
                                     ffn_dim=ffn_dim, 
                                     dropout=dropout,
                                     rope=rope)
            for _ in range(num_layers)
        ])  

        # self.norm = nn.RMSNorm(embed_dim)
        self.norm = nn.Identity()

    
    def forward(self, state, program, iteration: int):
        # state: BxSxD | program: BxD | iteration: int

        batch_size = state.shape[0]
        iteration_embedding = self.iteration_embedding(torch.tensor(iteration, device=state.device)).unsqueeze(0).expand(batch_size, -1)
    
        for layer in self.layers:
            state = layer(state, program, iteration_embedding)
        return self.norm(state)



class Interpreter(nn.Module):
    def __init__(self, n_layer_exec, n_layer_gen, n_dim, n_head, activation, max_iterations, dropout, rank, mlp_layers, mlp_dim, rope=None):
        super(Interpreter, self).__init__()


        common_kwargs = {
            "embed_dim": n_dim,
            "num_heads": n_head,
            "activation": activation,
            "dropout": dropout,
            "rope": rope
        }
         # Create a proper iteration embedding
        self.subroutine_generator = SubroutineGenerator(
            num_layers=n_layer_gen,
            **common_kwargs,
            max_iterations=max_iterations,
            ffn_dim=None)
        
        lora_kwargs = {
            "rank": rank,
            "mlp_layers": mlp_layers,
            "mlp_dim": mlp_dim,
        }

        self.subroutine_executor = SubroutineExecutor(
            num_layers=n_layer_exec,
            **common_kwargs,
            **lora_kwargs)
        
        self.output_projection = DynamicLoRALinear(
            dim_in=n_dim,
            dim_out=n_dim,
            subroutine_dim=n_dim,
            activation=activation,
            **lora_kwargs)
        
        self.norm_out_proj = nn.RMSNorm(n_dim)
        self.norm_subroutine = nn.RMSNorm(n_dim)

    def forward(self, prev_state, program, iteration: int):
        # input: BxSxD | program: BxD | iteration: int

        # Generate the raw subroutine
        subroutine_raw = self.subroutine_generator(prev_state, program, iteration)

        # Normalize the subroutine before using it for LoRA generation/application
        subroutine_normed = self.norm_subroutine(subroutine_raw)

        # Executor uses the normalized subroutine
        next_state = self.subroutine_executor(prev_state, subroutine_normed)

        # Extract the output state from the next_state
        output_iter = self.output_projection(self.norm_out_proj(next_state), subroutine_normed)

        return next_state, output_iter


class StateConstructor(nn.Module):
    def __init__(self, n_layer_exec, n_layer_gen, n_dim, n_head, activation, dropout, rank, mlp_layers, mlp_dim, rope=None):
        super(StateConstructor, self).__init__()
        common_kwargs = {
            "embed_dim": n_dim,
            "num_heads": n_head,
            "activation": activation,
            "dropout": dropout,
            "rope": rope
        }

        lora_kwargs = {
            "rank": rank,
            "mlp_layers": mlp_layers,
            "mlp_dim": mlp_dim,
        }

        self.subroutine_generator = SubroutineGenerator(
            num_layers=n_layer_gen,
            **common_kwargs,
            max_iterations=1, # Only one iteration for the state constructor
            )

        self.subroutine_executor = SubroutineExecutor(
            num_layers=n_layer_exec,
            **common_kwargs,
            **lora_kwargs)
        
        self.norm_subroutine = nn.RMSNorm(n_dim)

    def forward(self, input, program):
        # input: BxSxD | program: BxD

        # Generate and normalize the subroutine
        subroutine_raw = self.subroutine_generator(input, program, 0)
        subroutine_normed = self.norm_subroutine(subroutine_raw)

        # Execute using the normalized subroutine
        state = self.subroutine_executor(input, subroutine_normed)
        return state
        


@dataclass
class REPLConfig:

    # Base Transformer parameters
    n_dim: int = 256
    n_head: int = 4
    dropout: float = 0.0
    activation: str = "relu"
    num_iterations: int = 12
    n_state: int = 1024 # Number of tokens in the state
    use_rope: bool = True
    rope_base: int = 10000

    # Split layer configuration
    n_layer_exec: int = 2  # Number of layers for SubroutineExecutor
    n_layer_gen: int = 2   # Number of layers for SubroutineGenerator

    # LoRA parameters
    rank: int = 8
    mlp_layers: int = 2
    mlp_dim: int = 128


class REPL(nn.Module):
    def __init__(self, config: REPLConfig):
        super(REPL, self).__init__()

        self.rope = RotaryPositionalEmbeddings(dim=config.n_dim // config.n_head, 
                                               max_seq_len=1024, base=config.rope_base) if config.use_rope else None

        common_kwargs = {
            "n_layer_exec": config.n_layer_exec,
            "n_layer_gen": config.n_layer_gen,
            "n_dim": config.n_dim,
            "n_head": config.n_head,
            "activation": config.activation,
            "dropout": config.dropout,
            "rope": self.rope
        }

        lora_kwargs = {
            "rank": config.rank,
            "mlp_layers": config.mlp_layers,
            "mlp_dim": config.mlp_dim,
        }
        self.state_constructor = StateConstructor(**common_kwargs,
                                                  **lora_kwargs)
        self.interpreter = Interpreter( **common_kwargs,
                                        max_iterations=100,   # Set to 100 to ease loading old checkpoints,
                                        **lora_kwargs)
        self.num_iterations = config.num_iterations
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters using standard practices for transformer models.
        - Linear layers: Normal distribution with std=0.02
        - Layer norm: Weight=1.0
        - Embeddings: Normal distribution with std=0.02
        - Custom nn.Parameter: Initialized based on shape
        """
        def init_weights(module):
            if isinstance(module, nn.Linear):
                # Initialize linear weights with normal distribution
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                # Initialize layer norm parameters
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.Embedding):
                # Initialize embedding weights
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RotaryPositionalEmbeddings) and hasattr(module, 'reset_parameters'):
                # Use module's own reset_parameters if available
                module.reset_parameters()
            else:
                # Handle custom parameters that are directly attached to the module
                for name, param in module.named_parameters(recurse=False):
                    print(f"Initializing {name} of shape {param.shape}")
                    if param.requires_grad:
                        if param.dim() > 1:
                            # For matrices/tensors, use normal initialization
                            nn.init.normal_(param, mean=0.0, std=0.02)
                        else:
                            # For vectors (1D tensors), initialize to zeros
                            nn.init.zeros_(param)
            
        self.apply(init_weights)

    def forward(self, input, program):
        # input is BxSxD | program is BxD
        
        outputs = []
        prev_state = self.state_constructor(input, program)
        
        # Process through transformer encoder
        for it in range(self.num_iterations):
            # Generate the subroutine for this iteration
            next_state, output_iter = self.interpreter(prev_state, program, it)
            outputs.append(output_iter)
            prev_state = next_state  # Update the previous state

        outputs = torch.stack(outputs, dim=0) # shape: [num_iterations, batch, seq_len, n_dim]
        return outputs

