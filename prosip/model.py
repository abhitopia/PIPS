from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from prosip.grid_autoencoder import GridAutoEncoder, GridAutoEncoderConfig
from prosip.interpreter import Interpreter, InterpreterConfig
from prosip.trajectory_loss import vectorized_monotonic_trajectory_loss

@dataclass
class ProSIPConfig:
    # Common config
    n_dim: int = 256
    n_head: int = 4
    activation: str = "gelu"
    n_vocab: int = 16
    n_layer_encoder: int = 4
    n_layer_interpreter: int = 2
    dropout: float = 0.0
    program_vocab: int = 2048

    # Autoencoder config
    latent_height: int = 8
    latent_width: int = 8
    conv_block_size: int = 2
    n_conv_blocks: int = 2
    grid_height: int = 32
    grid_width: int = 32
    encode_norm: bool = True
    decode_norm: bool = True

    # Trajectory loss config
    margin: float = 0.0
    zero_margin_at_end: bool = True

    # LoRA parameters
    lora_rank: int = 8
    lora_mlp_layers: int = 2
    lora_mlp_dim: Optional[int] = None

    def __post_init__(self):

        if self.lora_mlp_dim is None:
            self.lora_mlp_dim = self.n_dim

        self.autoencoder_config = GridAutoEncoderConfig(
            n_vocab=self.n_vocab,
            n_dim=self.n_dim,
            n_layer=self.n_layer_encoder,
            n_head=self.n_head,
            latent_height=self.latent_height,
            latent_width=self.latent_width,
            conv_block_size=self.conv_block_size,
            n_conv_blocks=self.n_conv_blocks,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            encode_norm=self.encode_norm,
            decode_norm=self.decode_norm,
            activation=self.activation,
            dropout=self.dropout
        )
        self.interpreter_config = InterpreterConfig(
            n_layer=self.n_layer_interpreter,
            n_dim=self.n_dim,
            n_head=self.n_head,
            dropout=self.dropout,
            activation=self.activation,
            ffn_dim=self.n_dim * 4,
            lora_rank=self.lora_rank,
            lora_mlp_layers=self.lora_mlp_layers,
            lora_mlp_dim=self.lora_mlp_dim
        )

class ProSIPModel(nn.Module):
    def __init__(self, config: ProSIPConfig):
        super().__init__()
        self.config = config
        self.trajectory_margin = config.margin
        self.zero_margin_at_end = config.zero_margin_at_end

        self.token_embedding = nn.Embedding(config.n_vocab, config.n_dim)
        self.program_embedding = nn.Embedding(config.program_vocab, config.n_dim)
        self.autoencoder = GridAutoEncoder(config.autoencoder_config)
        self.interpreter = Interpreter(config.interpreter_config)

    def forward(self, input_grids: torch.Tensor, output_grids: torch.Tensor, program_ids: torch.Tensor, num_iterations: int = 1) -> torch.Tensor:
        # tasks is BxNx2xHxW
        B, H, W = input_grids.shape

        encoded_input_grids = self.autoencoder.encode(input_grids) # B, n_latent, n_dim
        encoded_output_grids = self.autoencoder.encode(output_grids) # B, n_latent, n_dim

        program_embeds = self.program_embedding(program_ids) # B, n_dim
        
        # Now reshape them to (B, N*n_latent, n_dim)
        intermediate_embeddings = self.interpreter.forward(encoded_input_grids, program_embeds, num_iterations)

        loss = vectorized_monotonic_trajectory_loss(intermediate_embeddings, 
                                                    encoded_input_grids, 
                                                    encoded_output_grids,
                                                    margin=self.trajectory_margin,
                                                    zero_margin_at_end=self.zero_margin_at_end)

        print("intermediate_embeddings.shape", intermediate_embeddings.shape)
        print("loss", loss)

        decoded_output_grids = self.autoencoder.decode(encoded_output_grids)
        decoded_input_grids = self.autoencoder.decode(encoded_input_grids)

        print(decoded_output_grids.shape)
        print(decoded_input_grids.shape)

        pass


if __name__ == "__main__":
    P = 2048
    V = 16
    B = 4 
    N = 3
    H = 32
    W = 32
    S = H * W

    # Create a batch of demonstrations
    input_grids = torch.randint(0, V, (B, H, W))
    output_grids = torch.randint(0, V, (B, H, W))
    program_batch = torch.randint(0, P, (B,))

    prosip_config = ProSIPConfig()

    model = ProSIPModel(prosip_config)
    
    output = model(input_grids, output_grids, program_batch, num_iterations=2)