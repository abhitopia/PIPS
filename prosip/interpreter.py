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
    def __init__(self, task_embedding_dim, target_dim_A, target_dim_B, rank,
                 num_hidden_layers=1, hidden_dim=128, activation=nn.ReLU, dropout=0.0):
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
        self.base_layer = base_layer  # expects weight of shape [out_features, in_features]
        self.rank = rank
        self.out_features = base_layer.out_features
        self.in_features = base_layer.in_features

    def forward(self, x, A_params, B_params):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, in_features]
            A_params: Tensor of shape [batch_size, out_features, rank]
            B_params: Tensor of shape [batch_size, rank, in_features]
        Returns:
            Output tensor of shape [batch_size, seq_len, out_features]
        """
        # Compute base output: [batch_size, seq_len, out_features]
        base_out = self.base_layer(x)
        # Compute intermediate: x @ B_params^T for each sample.
        # B_params.transpose(1,2) has shape [batch_size, in_features, rank].
        intermediate = torch.bmm(x, B_params.transpose(1, 2))  # [batch_size, seq_len, rank]
        # Then compute adaptation: intermediate @ A_params^T, A_params^T shape: [batch_size, rank, out_features]
        adaptation = torch.bmm(intermediate, A_params.transpose(1, 2))  # [batch_size, seq_len, out_features]
        return base_out + adaptation




class DynamicTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, ffn_dim, task_embedding_dim, rank, num_heads,  activation="relu", dropout=0.0, lora_mlp_layers=1, lora_mlp_dim=128):
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
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True, bias=False)
        self.norm1 = nn.RMSNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-Forward Network components.
        self.ffn_linear1 = nn.Linear(embed_dim, ffn_dim)
        self.ffn_linear2 = nn.Linear(ffn_dim, embed_dim)
        self.activation = get_activation_fn(activation, return_module=True)()
        self.norm2 = nn.RMSNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # Wrap ffn_linear1 with a dynamic LoRA adapter.
        self.lora_linear1 = LoRALinear(self.ffn_linear1, rank)
        # Hypernetwork to generate LoRA parameters for ffn_linear1.
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
            src: Input tensor of shape [batch_size, seq_len, embed_dim]
            task_embedding: Tensor of shape [batch_size, task_embedding_dim]
        Returns:
            Tensor of shape [batch_size, seq_len, embed_dim]
        """
        # --- Self-Attention Sublayer with Pre-Norm ---
        # Normalize the input before the attention.
        normed_src = self.norm1(src)
        attn_output, _ = self.self_attn(normed_src, normed_src, normed_src)
        # Add the residual connection.
        src = src + self.dropout1(attn_output)

        # --- Feed-Forward Sublayer with Dynamic LoRA and Pre-Norm ---
        # Normalize before entering the feed-forward sublayer.
        normed_src = self.norm2(src)
        # Generate dynamic LoRA parameters for the batch.
        A_params, B_params = self.hypernetwork(task_embedding)
        # Apply the LoRA-adapted first FFN layer.
        out_ffn1 = self.lora_linear1(normed_src, A_params, B_params)
        out_activated = self.activation(out_ffn1)
        out_ffn2 = self.ffn_linear2(out_activated)
        # Add residual connection.
        src = src + self.dropout2(out_ffn2)
        return src

    

class BaseDynamicTransformerEncoder(nn.Module):
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
        super(BaseDynamicTransformerEncoder, self).__init__()
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

    def forward(self, src, task_embedding):
        """
        Args:
            src: Tensor of shape [seq_len, batch_size, embed_dim]
            task_embedding: Tensor of shape [batch_size, task_embedding_dim]
        Returns:
            Processed tensor of shape [seq_len, batch_size, embed_dim]
        """
        for layer in self.layers:
            src = layer(src, task_embedding)
        return self.norm(src)



@dataclass
class InterpreterConfig:

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

    # Dynamic Program Embedding parameters
    def __post_init__(self):
        if self.ffn_dim is None:
            self.ffn_dim = self.n_dim * 4


class Interpreter(nn.Module):
    def __init__(self, config: InterpreterConfig):
        super(Interpreter, self).__init__()
        self.config = config
        
        # Create a proper iteration embedding
        self.iteration_embedding = nn.Embedding(config.max_iterations, config.n_dim)
        
        # Layer to combine program and iteration embeddings
        self.dynamic_prog_embedding = _build_mlp(
                                                input_dim=2*config.n_dim, 
                                                output_dim=config.n_dim, 
                                                num_hidden_layers=config.mlp_layers, 
                                                hidden_dim=config.mlp_dim, 
                                                activation=get_activation_fn(config.activation, return_module=True), 
                                                dropout=config.dropout)
        
        self.base_transformer = BaseDynamicTransformerEncoder(
            num_layers=config.n_layer,
            embed_dim=config.n_dim,
            ffn_dim=config.ffn_dim,
            task_embedding_dim=config.n_dim,
            rank=config.lora_rank,
            num_heads=config.n_head,
            dropout=config.dropout,
            activation=config.activation,
            lora_mlp_layers=config.mlp_layers,
            lora_mlp_dim=config.mlp_dim)

    def forward(self, x, program, num_iterations=1):
        # x is BxSxD
        # program is BxP
        batch_size = x.shape[0]
        output = []
        # Process through transformer encoder
        for it in range(num_iterations):
            # Get iteration embedding for this iteration
            iter_emb = self.iteration_embedding(torch.tensor(it, device=x.device))
            iter_emb = iter_emb.unsqueeze(0).expand(batch_size, -1)  # [batch_size, n_dim]
            
            # Concatenate iteration embedding with program embedding
            combined_embedding = torch.cat([program, iter_emb], dim=1)
            
            # Transform to get dynamic program embedding for this iteration
            iteration_program = self.dynamic_prog_embedding(combined_embedding)
            
            # Use the iteration-specific program embedding for the transformer
            x = self.base_transformer(x, iteration_program)
            
            output.append(x.clone())
        
        output = torch.stack(output, dim=0) # shape: [num_iterations, batch, seq_len, n_dim]
        return output

