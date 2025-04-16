from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from prosip.utils import get_activation_fn



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


class HyperNetwork(nn.Module):
    def __init__(self, task_embedding_dim, target_dim_A, target_dim_B, rank, num_hidden_layers, hidden_dim, activation, dropout):
        """
        Args:
            task_embedding_dim (int): Size of the task embedding.
            target_dim_A (int): Output dimension for A matrix (typically ffn_dim).
            target_dim_B (int): Input dimension for B matrix (typically embed_dim).
            rank (int): Rank for the low-rank factorization.
            num_hidden_layers (int): Number of hidden layers in the MLP.
            hidden_dim (int): Dimension of the hidden layers.
            activation (callable): Activation function class (e.g. nn.ReLU).
            dropout (float): Dropout rate between layers.
        """
        super(HyperNetwork, self).__init__()
        self.target_dim_A = target_dim_A
        self.target_dim_B = target_dim_B
        self.rank = rank

        # Build two MLPs to generate the parameters for the low-rank matrices.
        self.fc_A = _build_mlp(task_embedding_dim, target_dim_A * rank,
                                    num_hidden_layers, hidden_dim, activation, dropout)
        self.fc_B = _build_mlp(task_embedding_dim, rank * target_dim_B,
                                    num_hidden_layers, hidden_dim, activation, dropout)



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
    

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int):
        """
        Args:
            base_layer: A preexisting nn.Linear layer.
            rank: The rank for the low-rank update.
        """
        super(LoRALinear, self).__init__()
        self.base_layer = base_layer  # Expected shape: [out_features, in_features]
        self.rank = rank
        self.out_features = base_layer.out_features
        self.in_features = base_layer.in_features

    def forward(self, x, A_params, B_params):
        """
        Args:
            x: Tensor of shape [B, seq_len, in_features]
            A_params: Tensor of shape [B, seq_len, out_features, rank]
            B_params: Tensor of shape [B, seq_len, rank, in_features]
        Returns:
            Tensor of shape [B, seq_len, out_features]
        """
        # Compute the base output using the original layer.
        base_out = self.base_layer(x)  # Shape: [B, seq_len, out_features]

        B, seq_len, in_features = x.size()
        # Flatten the first two dimensions to process per-token.
        x_flat = x.view(B * seq_len, in_features)  # [B*seq_len, in_features]
        # For B_params: shape [B, seq_len, rank, in_features] -> flatten to [B*seq_len, rank, in_features]
        B_flat = B_params.view(B * seq_len, self.rank, in_features)
        # For A_params: shape [B, seq_len, out_features, rank] -> flatten to [B*seq_len, out_features, rank]
        A_flat = A_params.view(B * seq_len, self.out_features, self.rank)
        
        # Compute intermediate: for each token, do: x_i [1, in_features] @ B_flat_i^T -> [1, rank]
        # Use torch.bmm over the flattened batch.
        x_flat_unsq = x_flat.unsqueeze(1)  # [B*seq_len, 1, in_features]
        intermediate = torch.bmm(x_flat_unsq, B_flat.transpose(1, 2))  # [B*seq_len, 1, rank]
        # Compute adaptation: [B*seq_len, 1, rank] @ A_flat_i^T -> [B*seq_len, 1, out_features]
        adaptation = torch.bmm(intermediate, A_flat.transpose(1, 2))  # [B*seq_len, 1, out_features]
        adaptation = adaptation.squeeze(1).view(B, seq_len, self.out_features)
        
        return base_out + adaptation
    

class DynamicTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, ffn_dim, task_embedding_dim, rank, num_heads,
                 activation="relu", dropout=0.0, lora_mlp_layers=1, lora_mlp_dim=128):
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
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                               dropout=dropout, batch_first=True, bias=False)
        self.norm1 = nn.RMSNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-Forward Network components.
        self.ffn_linear1 = nn.Linear(embed_dim, ffn_dim)
        self.ffn_linear2 = nn.Linear(ffn_dim, embed_dim)
        # Using a helper to get the activation module.
        self.activation = get_activation_fn(activation, return_module=True)()
        self.norm2 = nn.RMSNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # Wrap ffn_linear1 with a dynamic LoRA adapter.
        self.lora_linear1 = LoRALinear(self.ffn_linear1, rank)
        # Hypernetwork to generate LoRA parameters; note the dimensions:
        # - For A: target dimension is ffn_dim.
        # - For B: target dimension is embed_dim.
        self.hypernetwork = HyperNetwork(task_embedding_dim=task_embedding_dim,
                                         target_dim_A=ffn_dim,
                                         target_dim_B=embed_dim,
                                         rank=rank,
                                         num_hidden_layers=lora_mlp_layers,
                                         hidden_dim=lora_mlp_dim,
                                         activation=get_activation_fn(activation, return_module=True),
                                         dropout=dropout)

    def forward(self, src, task_embedding):
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
        attn_output, _ = self.self_attn(normed_src, normed_src, normed_src)
        src = src + self.dropout1(attn_output)

        # --- Feed-Forward Sublayer with Dynamic LoRA and Pre-Norm ---
        normed_src = self.norm2(src)
        B, seq_len, _ = normed_src.shape
        # Flatten the token dimension for the hypernetwork.
        task_flat = task_embedding.view(B * seq_len, -1)  # [B*seq_len, task_embedding_dim]
        # Obtain LoRA parameters from hypernetwork.
        A_params_flat, B_params_flat = self.hypernetwork(task_flat)
        # Reshape to per-token shape.
        A_params = A_params_flat.view(B, seq_len, self.ffn_linear1.out_features, self.lora_linear1.rank)
        B_params = B_params_flat.view(B, seq_len, self.lora_linear1.rank, self.embed_dim)
        # Apply LoRA-adapted first FFN layer using per-token parameters.
        out_ffn1 = self.lora_linear1(normed_src, A_params, B_params)
        out_activated = self.activation(out_ffn1)
        out_ffn2 = self.ffn_linear2(out_activated)
        src = src + self.dropout2(out_ffn2)
        return src


class SubroutineExecutor(nn.Module):
    def __init__(self, num_layers, embed_dim, ffn_dim, task_embedding_dim, rank, num_heads=8, dropout=0.1, activation="relu", lora_mlp_layers=1, lora_mlp_dim=128):
        """
        Args:
            num_layers (int): Number of transformer encoder layers.
            embed_dim (int): Embedding dimension.
            ffn_dim (int): Inner dimension for feed-forward networks.
            task_embedding_dim (int): Dimension of the task embeddings.
            rank (int): Low-rank dimension for dynamic LoRA.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(SubroutineExecutor, self).__init__()
        self.layers = nn.ModuleList([
            DynamicTransformerEncoderLayer(embed_dim=embed_dim, 
                                           ffn_dim=ffn_dim, 
                                           task_embedding_dim=task_embedding_dim, 
                                           rank=rank, 
                                           num_heads=num_heads, 
                                           activation=activation,
                                           dropout=dropout,
                                           lora_mlp_layers=lora_mlp_layers,
                                           lora_mlp_dim=lora_mlp_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(embed_dim)

    def forward(self, state, subroutine):
        """
        Args:
            state: Tensor of shape [batch_size, seq_len, embed_dim]
            subroutine_embedding: Tensor of shape [batch_size, seq_len, embed_dim]
        Returns:
            Processed next state tensor of shape [batch_size, seq_len, embed_dim]
        """
        for layer in self.layers:
            state = layer(state, subroutine)
        return self.norm(state)
    

class SubroutineGeneratorLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, activation, ffn_dim=None, dropout=0.0):
        super(SubroutineGeneratorLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True, bias=False)
        self.norm_kv = nn.RMSNorm(embed_dim)
        self.norm_q = nn.RMSNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        if ffn_dim is None:
            ffn_dim = embed_dim * 4

        # Feed-Forward Network components.
        self.ffn_linear1 = nn.Linear(embed_dim, ffn_dim)
        self.ffn_linear2 = nn.Linear(ffn_dim, embed_dim)
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
        out_ffn1 = self.ffn_linear1(state) # BxSxD
        out_activated = self.activation(out_ffn1) # BxSxD
        out_ffn2 = self.ffn_linear2(out_activated) # BxSxD
        # Add residual connection.
        subroutine = state + self.dropout2(out_ffn2) # BxSxD 
        return subroutine
    

class SubroutineGenerator(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, activation, max_iterations, ffn_dim=None, dropout=0.0):
        super(SubroutineGenerator, self).__init__()

        self.iteration_embedding = nn.Embedding(max_iterations, embed_dim)

        self.layers = nn.ModuleList([
            SubroutineGeneratorLayer(embed_dim=embed_dim, 
                                     num_heads=num_heads, 
                                     activation=activation, 
                                     ffn_dim=ffn_dim, 
                                     dropout=dropout)
            for _ in range(num_layers)
        ])  

        self.norm = nn.RMSNorm(embed_dim)

    
    def forward(self, state, program, iteration: int):
        # state: BxSxD | program: BxD | iteration: int

        batch_size = state.shape[0]
        iteration_embedding = self.iteration_embedding(torch.tensor(iteration, device=state.device)).unsqueeze(0).expand(batch_size, -1)
    
        for layer in self.layers:
            state = layer(state, program, iteration_embedding)
        return self.norm(state)



class Interpreter(nn.Module):
    def __init__(self, n_layer, n_dim, n_head, activation, max_iterations, ffn_dim, dropout, lora_rank, mlp_layers, mlp_dim):
        super(Interpreter, self).__init__()

         # Create a proper iteration embedding
        self.subroutine_generator = SubroutineGenerator(
            num_layers=n_layer,
            embed_dim=n_dim,
            num_heads=n_head,
            activation=activation,
            max_iterations=max_iterations,
            ffn_dim=ffn_dim,
            dropout=dropout)

        self.subroutine_executor = SubroutineExecutor(
            num_layers=n_layer,
            embed_dim=n_dim,
            ffn_dim=ffn_dim,
            task_embedding_dim=n_dim,
            rank=lora_rank,
            num_heads=n_head,
            dropout=dropout,
            activation=activation,   
            lora_mlp_layers=mlp_layers,
            lora_mlp_dim=mlp_dim)
        
        self.output_extractor = SubroutineExecutor(
            num_layers=n_layer,  
            embed_dim=n_dim,
            ffn_dim=ffn_dim,
            task_embedding_dim=n_dim,
            rank=lora_rank,
            num_heads=n_head,
            dropout=dropout,
            activation=activation,
            lora_mlp_layers=mlp_layers,
            lora_mlp_dim=mlp_dim)
        
    def forward(self, prev_state, program, iteration: int):
        # input: BxSxD | program: BxD | iteration: int

        subroutine = self.subroutine_generator(prev_state, program, iteration)
        next_state = self.subroutine_executor(prev_state, subroutine)

        # Extract the output state from the next_state
        output_iter = self.output_extractor(next_state, subroutine)

        return next_state, output_iter


class StateConstructor(nn.Module):
    def __init__(self, n_layer, n_dim, n_head, activation, ffn_dim, dropout, lora_rank, mlp_layers, mlp_dim):
        super(StateConstructor, self).__init__()

           # Create a proper iteration embedding
        self.subroutine_generator = SubroutineGenerator(
            num_layers=n_layer,
            embed_dim=n_dim,
            num_heads=n_head,
            activation=activation,
            max_iterations=1, # Only one iteration for the state constructor
            dropout=dropout)

        self.subroutine_executor = SubroutineExecutor(
            num_layers=n_layer,
            embed_dim=n_dim,
            ffn_dim=ffn_dim,
            task_embedding_dim=n_dim,
            rank=lora_rank,
            num_heads=n_head,
            dropout=dropout,
            activation=activation,   
            lora_mlp_layers=mlp_layers,
            lora_mlp_dim=mlp_dim)
    
    def forward(self, input, program):
        # input: BxSxD | program: BxD
        subroutine = self.subroutine_generator(input, program, 0)
        state = self.subroutine_executor(input, subroutine)
        return state
        


@dataclass
class REPLConfig:

    # Base Transformer parameters
    n_layer: int = 2
    n_dim: int = 256
    n_head: int = 4
    dropout: float = 0.0
    activation: str = "relu"
    max_iterations: int = 12
    ffn_dim: Optional[int] = None

    # LoRA parameters
    lora_rank: int = 8
    mlp_layers: int = 2
    mlp_dim: int = 128


    def __post_init__(self):
        if self.ffn_dim is None:
            self.ffn_dim = self.n_dim * 4


class REPL(nn.Module):
    def __init__(self, config: REPLConfig):
        super(REPL, self).__init__()
        self.state_constructor = StateConstructor(n_layer=config.n_layer, 
                                                  n_dim=config.n_dim, 
                                                  n_head=config.n_head, 
                                                  activation=config.activation, 
                                                  ffn_dim=config.ffn_dim, 
                                                  dropout=config.dropout, 
                                                  lora_rank=config.lora_rank, 
                                                  mlp_layers=config.mlp_layers, 
                                                  mlp_dim=config.mlp_dim)
        self.interpreter = Interpreter(n_layer=config.n_layer, 
                                        n_dim=config.n_dim, 
                                        n_head=config.n_head, 
                                        activation=config.activation, 
                                        max_iterations=config.max_iterations, 
                                        ffn_dim=config.ffn_dim, 
                                        dropout=config.dropout, 
                                        lora_rank=config.lora_rank, 
                                        mlp_layers=config.mlp_layers, 
                                        mlp_dim=config.mlp_dim)
       

    def forward(self, input, program, num_iterations=1):
        # input is BxSxD | program is BxD
        outputs = []
        prev_state = self.state_constructor(input, program)
        
        # Process through transformer encoder
        for it in range(num_iterations):
            # Generate the subroutine for this iteration
            next_state, output_iter = self.interpreter(prev_state, program, it)
            outputs.append(output_iter)
            prev_state = next_state  # Update the previous state

        outputs = torch.stack(outputs, dim=0) # shape: [num_iterations, batch, seq_len, n_dim]
        return outputs

