from dataclasses import dataclass
from functools import partial
import os
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pips.data import DatasetType
from prosip.grid_autoencoder import GridAutoEncoder, GridAutoEncoderConfig
from prosip.repl import REPL, REPLConfig
from prosip.task_dataset import ExampleDataset, collate_fn_project, worker_init_fn
from prosip.trajectory_loss import vectorized_monotonic_trajectory_loss

@dataclass
class ProSIPConfig:
    # Common config
    n_dim: int = 256
    n_head: int = 4
    activation: str = "gelu"
    n_vocab: int = 16
    n_layer_encoder: int = 4
    n_layer_interpreter_exec: int = 2
    n_layer_interpreter_gen: int = 2
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

    # LoRA parameters
    lora_rank: int = 8
    lora_mlp_layers: int = 2
    lora_mlp_dim: Optional[int] = None

    def __post_init__(self):

        if self.lora_mlp_dim is None:
            self.lora_mlp_dim = self.n_dim

        self.padding_idx = self.n_vocab - 1

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
        self.repl_config = REPLConfig(
            n_layer_exec=self.n_layer_interpreter_exec,
            n_layer_gen=self.n_layer_interpreter_gen,
            n_dim=self.n_dim,
            n_head=self.n_head,
            dropout=self.dropout,
            activation=self.activation,
            ffn_dim=self.n_dim * 4,
            lora_rank=self.lora_rank,
            mlp_layers=self.lora_mlp_layers,
            mlp_dim=self.lora_mlp_dim,
            n_state=self.latent_height * self.latent_width
        )

class ProSIPModel(nn.Module):
    def __init__(self, config: ProSIPConfig):
        super().__init__()
        self.config = config
        self.trajectory_margin = config.margin

        self.token_embedding = nn.Embedding(config.n_vocab, config.n_dim)
        self.program_embedding = nn.Embedding(config.program_vocab, config.n_dim)
        self.autoencoder = GridAutoEncoder(config.autoencoder_config)
        self.interpreter = REPL(config.repl_config)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters for ProSIPModel.
        - Embeddings: Normal distribution with std=0.02
        - Also calls reset_parameters on submodules that have it
        """
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.program_embedding.weight, mean=0.0, std=0.02)
        
        # Call reset_parameters on submodules that have their own implementations
        self.autoencoder.reset_parameters()
        self.interpreter.reset_parameters()

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
                                                    margin=self.trajectory_margin)

        print("intermediate_embeddings.shape", intermediate_embeddings.shape)
        print("loss", loss)

        decoded_output_grids = self.autoencoder.decode(encoded_output_grids)
        decoded_input_grids = self.autoencoder.decode(encoded_input_grids)

        print(decoded_output_grids.shape)
        print(decoded_input_grids.shape)


def create_dataloaders(
    batch_size: int,
    data_multiplier: int = 1,
    ds_type: DatasetType = DatasetType.ALL,
    padding_idx: int = 15,
    num_train_examples: int = None,
    num_val_examples: int = None,
    max_grid_height: int = 32,
    max_grid_width: int = 32
):
    train_ex_ds = ExampleDataset(dataset_type=ds_type, 
                                 data_multiplier=data_multiplier, 
                                 max_examples=num_train_examples,
                                 is_test=False)
    
    val_ex_ds = ExampleDataset(dataset_type=ds_type, 
                               data_multiplier=1,
                               max_examples=num_val_examples,
                               is_test=True)
    
    tokenizer = train_ex_ds.get_program_tokenizer()


    collate_fn = partial(collate_fn_project, 
                    program_tokenizer=tokenizer,
                    pad_value=padding_idx,
                    max_height=max_grid_height,
                    max_width=max_grid_width,
                    flatten=False)

    num_workers = min(8, os.cpu_count() or 1)


    train_loader = DataLoader(
        train_ex_ds, 
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_ex_ds, 
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        shuffle=False,
        drop_last=True,
        persistent_workers=True,
    )

    print("Number of batches in training set: ", len(train_loader))
    print("Number of batches in validation set: ", len(val_loader))
    return train_loader, val_loader, tokenizer


if __name__ == "__main__":

    train_loader, val_loader, tokenizer = create_dataloaders(batch_size=4, 
                                                  data_multiplier=1, 
                                                  ds_type=DatasetType.ALL, 
                                                  padding_idx=15,
                                                  max_grid_height=32, 
                                                  max_grid_width=32,
                                                  num_val_examples=1000)
    batch = next(iter(train_loader))

    input_grids, output_grids, program_ids, attributes = batch

    print(input_grids.shape)
    print(output_grids.shape)
    print(program_ids.shape)
    print(attributes)

    prosip_config = ProSIPConfig(program_vocab=len(tokenizer))

    model = ProSIPModel(prosip_config)
    output = model(input_grids, output_grids, program_ids, num_iterations=2)
    output = model(input_grids, output_grids, program_ids, num_iterations=2)