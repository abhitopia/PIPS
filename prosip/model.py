import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional, List
from dataclasses import asdict, dataclass, field
from functools import partial
from rich import print

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch import Tensor
from torch.serialization import add_safe_globals  # Add this import at the top

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import LearningRateMonitor



from pips.misc.checkpoint_with_wandb_sync import ModelCheckpointWithWandbSync
from pips.misc.custom_progress_bar import CustomRichProgressBar
from pips.data import DatasetType
from pips.misc.acceleration_config import AccelerationConfig

from prosip.grid_autoencoder import GridAutoEncoder, GridAutoEncoderConfig
from prosip.repl import REPL, REPLConfig
from prosip.utils import normalize_state_dict, load_model_weights
from prosip.task_dataset import ExampleDataset, TaskDataset, Tokenizer, worker_init_fn
from prosip.trajectory_loss import vectorized_monotonic_trajectory_loss

@dataclass
class ProSIPConfig:
    n_dim: int = 256
    n_head: int = 4
    activation: str = "gelu"
    n_vocab: int = 16
    n_layer_encoder: int = 4
    n_layer_interpreter_exec: int = 2
    n_layer_interpreter_gen: int = 2
    dropout: float = 0.0
    program_vocab: int = 2048
    num_iterations: int = 1

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
    rope_base: int = 4000

    # LoRA parameters
    rank: int = 8
    mlp_layers: int = 2
    mlp_dim: Optional[int] = None

    def __post_init__(self):

        if self.mlp_dim is None:
            self.mlp_dim = self.n_dim

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
            rank=self.rank,
            mlp_layers=self.mlp_layers,
            mlp_dim=self.mlp_dim,
            n_state=self.latent_height * self.latent_width,
            num_iterations=self.num_iterations,
            rope_base=self.rope_base,
            use_rope=True
        )

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)

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

    def freeze_autoencoder(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_program_embeddings(self):
        for param in self.program_embedding.parameters():
            param.requires_grad = False
        
    def unfreeze_program_embeddings(self):
        for param in self.program_embedding.parameters():
            param.requires_grad = True

    def forward(self, input_grids: torch.Tensor, output_grids: torch.Tensor, program_ids: torch.Tensor) -> torch.Tensor:
        # tasks is BxNx2xHxW
        encoded_input_grids = self.autoencoder.encode(input_grids, mask_percentage=torch.tensor(0, device=input_grids.device)) # B, n_latent, n_dim
        encoded_output_grids = self.autoencoder.encode(output_grids, mask_percentage=torch.tensor(0, device=output_grids.device)) # B, n_latent, n_dim
        program_embeds = self.program_embedding(program_ids) # B, n_dim
        
        # Now reshape them to (B, N*n_latent, n_dim)
        intermediate_embeddings = self.interpreter.forward(encoded_input_grids, program_embeds)
        last_embeddings = intermediate_embeddings[-1]

        trajectory_loss = vectorized_monotonic_trajectory_loss(intermediate_embeddings, 
                                                    encoded_input_grids, 
                                                    encoded_output_grids,
                                                    margin=self.trajectory_margin)

        decoded_input_grids = self.autoencoder.decode(encoded_input_grids)
        decoded_output_grids = self.autoencoder.decode(encoded_output_grids) # B, H, W
        predicted_output_grids = self.autoencoder.decode(last_embeddings)

        # Calculate Cross Entropy Loss between input_grids and decoded_input_grids
        input_reconstruction_loss = nn.functional.cross_entropy(decoded_input_grids.view(-1, self.config.n_vocab), input_grids.view(-1))
        output_reconstruction_loss = nn.functional.cross_entropy(decoded_output_grids.view(-1, self.config.n_vocab), output_grids.view(-1))
        reconstruction_loss = (input_reconstruction_loss + output_reconstruction_loss) / 2
        prediction_loss = nn.functional.cross_entropy(predicted_output_grids.view(-1, self.config.n_vocab), output_grids.view(-1))
        latent_alignment_loss = nn.functional.mse_loss(last_embeddings, encoded_output_grids)

        input_predictions = decoded_input_grids.argmax(dim=-1)  # Shape: [B, H, W]
        output_predictions = decoded_output_grids.argmax(dim=-1)  # Shape: [B, H, W]
        predicted_predictions = predicted_output_grids.argmax(dim=-1)  # Shape: [B, H, W]

        predictions = {
            "input": input_predictions,
            "output": output_predictions,
            "predicted": predicted_predictions
        }

        loss_dict = {
            "reconstruction_loss": reconstruction_loss,
            "prediction_loss": prediction_loss,
            "trajectory_loss": trajectory_loss,
            "alignment_loss": latent_alignment_loss
        }

        return predictions, loss_dict

def create_dataloaders(
    batch_size: int,
    group_by_program: bool = False,
    data_multiplier: int = 1,
    ds_type: DatasetType = DatasetType.ALL,
    padding_idx: int = 15,
    num_train_examples: int = None,
    num_val_examples: int = None,
    max_grid_height: int = 32,
    max_grid_width: int = 32,
    num_workers: int = min(8, os.cpu_count() or 1)
):
    

    if group_by_program:
        train_ex_ds = TaskDataset(dataset_type=ds_type, 
                                  max_tasks=num_train_examples // 3 if num_train_examples is not None else None,
                                  data_multiplier=data_multiplier)
        
        val_ex_ds = TaskDataset(dataset_type=ds_type, 
                                  max_tasks=num_val_examples if num_val_examples is not None else None,
                                  data_multiplier=data_multiplier)
    else:
        train_ex_ds = ExampleDataset(dataset_type=ds_type, 
                                    data_multiplier=data_multiplier, 
                                    max_examples=num_train_examples,
                                    is_test=False)
        
        val_ex_ds = ExampleDataset(dataset_type=ds_type, 
                                data_multiplier=1,
                                max_examples=num_val_examples,
                                is_test=True)
    
    tokenizer = train_ex_ds.get_program_tokenizer()


    kwargs = {
        "program_tokenizer": tokenizer,
        "pad_value": padding_idx,
        "max_height": max_grid_height,
        "max_width": max_grid_width,
        "flatten": False
    }

    if group_by_program:
        collate_fn_train = partial(train_ex_ds.collate_fn_project, **kwargs, is_test=False)
        collate_fn_val = partial(val_ex_ds.collate_fn_project, **kwargs, is_test=True)
        train_batch_size = batch_size // 3
        val_batch_size = batch_size * 2
    else:
        collate_fn_train = partial(ExampleDataset.collate_fn_project, **kwargs)
        collate_fn_val = partial(ExampleDataset.collate_fn_project, **kwargs)
        train_batch_size = batch_size
        val_batch_size = batch_size * 2
   

    train_loader = DataLoader(
        train_ex_ds, 
        collate_fn=collate_fn_train,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=True,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_ex_ds, 
        collate_fn=collate_fn_val,
        batch_size=val_batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        shuffle=False,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    print(f"Train batch size: {train_batch_size * 3 if group_by_program else train_batch_size}, Val batch size: {val_batch_size}")
    print("Number of batches in training set: ", len(train_loader))
    print("Number of batches in validation set: ", len(val_loader))
    return train_loader, val_loader, tokenizer


@dataclass
class ProSIPExperimentConfig:
    """Configuration for training hyperparameters and schedules"""
    # Model configuration
    model_config: ProSIPConfig = field(default_factory=ProSIPConfig)
    
    # General training parameters
    seed: int | None = None  # None means random seed
    batch_size: int = 64
    max_steps: int = 1_000_000
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    
    # Learning rate parameters
    learning_rate: float = 1e-4  # Consistent with Dalle-E paper
    lr_min: float = 1e-6  # Minimum learning rate to be reached after decay
    program_embedding_lr_multiplier: float = 0.1 # Multiplier for embedding LR relative to main LR
    warmup_steps_lr: int = 1_000
    decay_steps_lr: int | None = None
    adamw_betas_1: float = 0.9
    adamw_betas_2: float = 0.999
    weight_decay: float = 1e-4  # Consistent with Dalle-E paper
    
    # Commitment loss parameters
    beta_reconstruction: float = 1.0
    beta_prediction: float = 1.0
    beta_trajectory: float = 1.0
    beta_alignment: float = 0.0
    # Dataset parameters
    dataset: DatasetType = DatasetType.ALL
    group_by_program: bool = False
    limit_training_samples: int | None = None  # Limit the number of training samples. None means use all samples.
    limit_validation_samples: int | None = None  # Limit the number of validation samples. None means use all samples.
    data_multiplier: int = 2  # How many times to repeat the dataset
    
    # Other parameters
    model_src: str | None = None
    train_only_program_embeddings: bool = False
    freeze_autoencoder: bool = False
    em_start_epoch: int | None = None

    def __post_init__(self):
        if self.accumulate_grad_batches < 1:
            raise ValueError("accumulate_grad_batches must be >= 1")
            
        # Generate random seed if none provided
        if self.seed is None:
            self.seed = np.random.randint(0, 2**32 - 1)

        # Cap warmup steps at max_steps and ensure warmup + transition <= max_steps
      
        # For learning rate (special case - using original name)
        self.warmup_steps_lr = min(self.warmup_steps_lr, self.max_steps)
        if self.decay_steps_lr is None:
            self.decay_steps_lr = self.max_steps - self.warmup_steps_lr

    def set_em_start_step(self, dataloader_length: int):
        if self.em_start_epoch is None:
            self.em_start_step = None
        else:
            self.em_start_step = (self.em_start_epoch * dataloader_length // self.accumulate_grad_batches)

    def to_dict(self) -> dict:
        """Convert config to a dictionary."""
        config_dict = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            # Handle nested VQVAEConfig
            if field == 'model_config':
                config_dict[field] = value.to_dict()
            # Handle DatasetType enum
            elif isinstance(value, DatasetType):
                config_dict[field] = value.name
            else:
                config_dict[field] = value
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ProSIPExperimentConfig':
        """Create config from a dictionary."""
        # Create a copy to avoid modifying the input
        config = config_dict.copy()
        
        # Handle nested VQVAEConfig
        if isinstance(config.get('model_config'), dict):
            config['model_config'] = ProSIPConfig.from_dict(config['model_config'])
        
        # Handle DatasetType fields
        if 'dataset' in config:
            config['dataset'] = DatasetType[config['dataset']]
            
        return cls(**config)

    @staticmethod
    def from_checkpoint(checkpoint_path: str) -> 'ProSIPExperimentConfig':
        # Add our custom classes to safe globals
        add_safe_globals([ProSIPExperimentConfig, ProSIPConfig])
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        config = ckpt['hyper_parameters']['experiment_config']
        assert isinstance(config, ProSIPExperimentConfig)
        return config
    

class ProSIPTrainingModule(pl.LightningModule):
    def __init__(self, experiment_config: ProSIPExperimentConfig, tokenizer: Tokenizer, compile_model: bool = False):
        super(ProSIPTrainingModule, self).__init__()
        self.experiment_config = experiment_config
        self.model_config = experiment_config.model_config
        self.model = None
        self.pad_idx = experiment_config.model_config.padding_idx
        self.learning_rate = experiment_config.learning_rate
        self.compile_model = compile_model
        self.tokenizer = tokenizer
        self.save_hyperparameters()


    def configure_model(self):
        """
        Compile the model after device placement.
        This gets called in on_fit_start so that the model is already on GPU.
        """
        if self.model is not None:
            return

        self.model = ProSIPModel(self.model_config)
        
        if self.compile_model:
            print("Compiling model using torch.compile...")
            self.model = torch.compile(
                self.model,
                fullgraph=True,
                mode="reduce-overhead",
                backend="inductor"
            )
        else:
            print("Model compilation disabled; skipping torch.compile.")
    
    def on_train_start(self):
        super().on_train_start()
        # Count model parameters after model is fully initialized
        model_parameters = sum(p.numel() for n, p in self.named_parameters() if p.requires_grad and 'program_embedding' not in n)
        print(f"Number of parameters in the model (excluding program embedding): {model_parameters}")
        print(f"Expected size of the model: {model_parameters * 4 / 10**6} MB")

        if self.experiment_config.freeze_autoencoder:
            print("NOTE: Freezing autoencoder")
            self.model.freeze_autoencoder()
        
        if self.experiment_config.train_only_program_embeddings:
            print("NOTE: Freezing everything except program embeddings")
            self.model.freeze()
            self.model.unfreeze_program_embeddings()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Handle loading checkpoints with different state dict keys and handle mismatched codebook sizes."""
        if self.model is None:
            self.configure_model()
        ckpt_state_dict = checkpoint["state_dict"]
        current_state_dict = self.state_dict()
        new_state_dict = normalize_state_dict(current_state_dict, ckpt_state_dict)
        checkpoint["state_dict"] = new_state_dict
        print(f"Loaded all model weights")

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        # Check if we should use EM training
        current_step = self.trainer.global_step
        em_start_step = self.experiment_config.em_start_step

        if em_start_step is None or current_step < em_start_step:
            return
        
        # Determine if this is an E-step or M-step
        is_e_step = ((current_step - em_start_step) % 2 == 0)
        if is_e_step:
            # E-step: freeze program embeddings
            self.model.unfreeze()
            self.model.freeze_program_embeddings()
        else:
            # M-step: freeze everything except program embeddings
            self.model.freeze()
            self.model.unfreeze_program_embeddings()

    def compute_accuracies(self, predictions: Tensor, target: Tensor):
         # Calculate token accuracy excluding padding tokens.
        non_padding_mask = (target != self.pad_idx)  # Shape: [B, H, W]
        # Simple token accuracy calculation - must filter out padding tokens
        total_non_padding_tokens = non_padding_mask.sum().float()
        correct_non_padding_tokens = ((predictions == target) & non_padding_mask).sum().float()
        token_accuracy = correct_non_padding_tokens / total_non_padding_tokens

        # Sample accuracy - a sample is correct if ALL tokens match (including padding)
        sample_correct = torch.all(predictions == target, dim=(1, 2)).float()  # Shape: [B]
        sample_accuracy = sample_correct.mean()

        return token_accuracy, sample_accuracy

    def data_accuracy(self, predictions: Dict[str, Tensor], target: Tensor, input_target: Tensor, attributes: List[Dict], phase: str):
        """
        Compute and log accuracy metrics broken down by dataset attribute.
        
        Args:
            predictions: Dictionary with keys 'input', 'output', 'predicted' containing prediction tensors
            target: Target tensor to compare against for output/predicted
            input_target: Target tensor to compare against for input
            attributes: List of attribute dictionaries with dataset information
            phase: 'train' or 'val' indicating which phase we're in
        """
        # Extract unique datasets from attributes
        datasets = {attr['dataset'] for attr in attributes}
        
        # Calculate accuracies for each dataset
        for dataset in datasets:
            # Create mask for this dataset
            dataset_mask = torch.tensor([attr['dataset'] == dataset for attr in attributes], 
                                       device=target.device)
            
            # Count samples for this dataset (for proper batch size logging)
            dataset_sample_count = dataset_mask.sum().item()
            
            # For each prediction type (input, output, predicted)
            for pred_type, pred_tensor in predictions.items():
                # Filter predictions by dataset mask
                dataset_preds = pred_tensor[dataset_mask]
                
                # Use the appropriate target based on prediction type
                if pred_type == 'input':
                    dataset_targets = input_target[dataset_mask]
                else:
                    dataset_targets = target[dataset_mask]
                    
                # Compute accuracies for this dataset and prediction type
                token_acc, sample_acc = self.compute_accuracies(dataset_preds, dataset_targets)
                
                # Create metrics dict for logging
                metrics = {
                    f'token_accuracy_{pred_type}({dataset})': token_acc,
                    f'sample_accuracy_{pred_type}({dataset})': sample_acc,
                }
                
                # Log metrics for this dataset
                self.log_metrics(metrics, phase, dataset_sample_count)

    def forward(self, input_grids: Tensor, output_grids: Tensor, program_ids: Tensor,
                beta_reconstruction: Tensor = torch.tensor(1.0), 
                beta_prediction: Tensor = torch.tensor(1.0),
                beta_trajectory: Tensor = torch.tensor(1.0),
                beta_alignment: Tensor = torch.tensor(0.0)):
        
        # Forward pass with provided scheduled parameters.
        predictions, loss_dict = self.model.forward(input_grids, output_grids, program_ids)
        input_token_accuracy, input_sample_accuracy = self.compute_accuracies(predictions["input"], input_grids)
        output_token_accuracy, output_sample_accuracy = self.compute_accuracies(predictions["output"], output_grids)
        prediction_token_accuracy, prediction_sample_accuracy = self.compute_accuracies(predictions["predicted"], output_grids) 
        
        # Compute total loss in a way that maintains the computational graph.
        # Scale the KLD losses by the number of latents (similar to Dalle-E paper).
        raw_losses = {
            'loss(Reconstruction)': loss_dict['reconstruction_loss'],
            'loss(Prediction)': loss_dict['prediction_loss'],
            'loss(Trajectory)': loss_dict['trajectory_loss'],
            'loss(Alignment)': loss_dict['alignment_loss']
        }
        
        # Compute weighted losses for total loss.
        weighted_losses = {
            'loss(Reconstruction)': raw_losses['loss(Reconstruction)'] * beta_reconstruction,
            'loss(Prediction)': raw_losses['loss(Prediction)'] * beta_prediction,
            'loss(Trajectory)': raw_losses['loss(Trajectory)'] * beta_trajectory,
            'loss(Alignment)': raw_losses['loss(Alignment)'] * beta_alignment
        }
        
        total_loss = sum(weighted_losses.values())
        
        # Return tensor values; logging callbacks can detach/convert them as needed.
        return predictions, {
            'loss': total_loss,
            **raw_losses,
            'token_accuracy(Input)': input_token_accuracy,
            'sample_accuracy(Input)': input_sample_accuracy,
            'token_accuracy(Output)': output_token_accuracy,
            'sample_accuracy(Output)': output_sample_accuracy,
            'token_accuracy(Prediction)': prediction_token_accuracy,
            'sample_accuracy(Prediction)': prediction_sample_accuracy,
        }

    def training_step(self, batch, batch_idx):
        # torch.compiler.cudagraph_mark_step_begin()
        input_grids, output_grids, program_ids, attributes = batch

        beta_reconstruction = torch.tensor(self.experiment_config.beta_reconstruction, device=input_grids.device)
        beta_prediction = torch.tensor(self.experiment_config.beta_prediction, device=input_grids.device)
        beta_trajectory = torch.tensor(self.experiment_config.beta_trajectory, device=input_grids.device)
        beta_alignment = torch.tensor(self.experiment_config.beta_alignment, device=input_grids.device)

        predictions, output_dict = self(
            input_grids=input_grids,
            output_grids=output_grids,
            program_ids=program_ids,
            beta_reconstruction=beta_reconstruction,
            beta_prediction=beta_prediction,
            beta_trajectory=beta_trajectory,
            beta_alignment=beta_alignment
        )

        output_dict['beta(Reconstruction)'] = beta_reconstruction
        output_dict['beta(Prediction)'] = beta_prediction
        output_dict['beta(Trajectory)'] = beta_trajectory
        output_dict['beta(Alignment)'] = beta_alignment

        self.log_metrics(output_dict, 'train', batch[0].size(0))
        
        # Calculate and log dataset-specific accuracies
        self.data_accuracy(predictions, output_grids, input_grids, attributes, 'train')

        return output_dict

    def validation_step(self, batch, batch_idx):
        # torch.compiler.cudagraph_mark_step_begin()
        input_grids, output_grids, program_ids, attributes = batch

        # For validation, scheduled values should be passed as provided (e.g., max_pct_mask can be set to 0 for no masking).
        beta_reconstruction = torch.tensor(self.experiment_config.beta_reconstruction, device=input_grids.device)
        beta_prediction = torch.tensor(self.experiment_config.beta_prediction, device=input_grids.device)
        beta_trajectory = torch.tensor(self.experiment_config.beta_trajectory, device=input_grids.device)
        beta_alignment = torch.tensor(self.experiment_config.beta_alignment, device=input_grids.device)

        predictions,output_dict = self(
            input_grids=input_grids,
            output_grids=output_grids,
            program_ids=program_ids,
            beta_reconstruction=beta_reconstruction,
            beta_prediction=beta_prediction,
            beta_trajectory=beta_trajectory,
            beta_alignment=beta_alignment
        )

        self.log_metrics(output_dict, 'val', batch[0].size(0))
        # Calculate and log dataset-specific accuracies
        self.data_accuracy(predictions, output_grids, input_grids, attributes, 'val')
        
        return output_dict

    def configure_optimizers(self):
        # Get parameters specifically from the program embedding module
        program_embedding_param_ids = set(id(p) for p in self.model.program_embedding.parameters())

        # Separate other parameters into decay and no-decay groups
        decay_params_other = []
        nodecay_params_other = []
        program_embedding_params = []

        for name, param in self.model.named_parameters(): # Iterate over model parameters
            if not param.requires_grad:
                continue

            param_id = id(param)
            if param_id in program_embedding_param_ids:
                program_embedding_params.append(param)
                # print(f"Found program embedding param: {name}") # Optional: for debugging
            else:
                # Apply weight decay to matrices (conv kernels, linear weights)
                if param.dim() >= 2:
                    decay_params_other.append(param)
                    # print(f"Found decay param: {name}") # Optional: for debugging
                # No weight decay for biases, norms, 1D params
                else:
                    nodecay_params_other.append(param)
                    # print(f"Found no-decay param: {name}") # Optional: for debugging

        # Define the main learning rate and the embedding learning rate
        main_lr = self.learning_rate
        embedding_lr = main_lr * self.experiment_config.program_embedding_lr_multiplier

        optim_groups = [
            {'params': decay_params_other, 'lr': main_lr, 'weight_decay': self.experiment_config.weight_decay},
            {'params': nodecay_params_other, 'lr': main_lr, 'weight_decay': 0.0},
            # Group for program embeddings with specific (lower) LR and no weight decay
            {'params': program_embedding_params, 'lr': embedding_lr, 'weight_decay': 0.0}
        ]


        # Verify parameter counts (optional, for debugging)
        num_prog_emb = sum(p.numel() for p in program_embedding_params)
        num_decay = sum(p.numel() for p in decay_params_other)
        num_nodecay = sum(p.numel() for p in nodecay_params_other)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total Params: {total_params}, ProgEmb: {num_prog_emb}, Decay: {num_decay}, NoDecay: {num_nodecay}")
        assert total_params == num_prog_emb + num_decay + num_nodecay, "Parameter misallocation in optimizer groups!"

        optimizer = AdamW(
            optim_groups,
            lr=main_lr, # Default LR (overridden by group LRs)
            betas=(self.experiment_config.adamw_betas_1, self.experiment_config.adamw_betas_2),
            eps=1e-8,
            fused=True if torch.cuda.is_available() else False
        )

        # Noam scheduler with linear warmup and cosine decay using lr-specific warmup
        def lr_lambda(step):
            warmup_steps = self.experiment_config.warmup_steps_lr
            min_lr_factor = self.experiment_config.lr_min / self.experiment_config.learning_rate

            if step < warmup_steps:
                base_factor = float(step) / float(max(1, warmup_steps))
            elif step < self.experiment_config.max_steps: # Use max_steps directly if decay_steps_lr is None or large
                progress = float(step - warmup_steps) / float(max(1, self.experiment_config.max_steps - warmup_steps))
                base_factor = min_lr_factor + 0.5 * (1.0 - min_lr_factor) * (1.0 + np.cos(np.pi * progress))
            else:
                base_factor = min_lr_factor

            return base_factor

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }


    def log_metrics(self, metrics: Dict[str, float], phase: str, batch_size: int):

        assert phase in ['train', 'val']

        # Process each metric in the outputs dictionary
        for key, value in metrics.items():
            # Skip WandB Images since we handle them separately
            # Handle the main loss separately.
            if key == 'loss':
                metric_name = f'Loss/{key}_{phase}'  # Default format.
            # Handle metrics with categories - loss(CE), loss(MI), etc.
            elif '(' in key and ')' in key:
                category = key.split('(')[-1].split(')')[0]  # Extract category.
                metric_name = f'{category}/{key.split("(")[0]}_{phase}'  # Format as "category/metric_phase"
            # Handle any remaining metrics.
            elif key.strip():
                metric_name = f'{key.capitalize()}/{key}_{phase}'
            
            sync_dist = False if phase == 'train' else True
            on_step = True if phase == 'train' else False
            on_epoch = True if phase == 'val' else False
            self.log(metric_name, value, on_step=on_step, on_epoch=on_epoch, batch_size=batch_size, logger=True, sync_dist=sync_dist)


def train(
    experiment_config: ProSIPExperimentConfig,
    run_name: str,
    project_name: str,
    checkpoint_dir: Path,
    debug_mode: bool = False,
    wandb_logging: bool = True,
    val_check_interval: int | None = None,
    resume_from: str | None = None,
    acceleration: AccelerationConfig | None = None,
    lr_find: bool = False,
    grad_log_interval: int = 100,
    visualization_interval: int = 100,
    num_grids_to_visualize: int = 4
) -> None:
    """Train a DVAE model with the given configuration."""
    

    # Create default acceleration config if none provided
    if acceleration is None:
        acceleration = AccelerationConfig()
    
    # Set all random seeds
    seed = experiment_config.seed
    pl.seed_everything(seed, workers=True)
    
    # Apply acceleration settings
    acceleration.apply_settings()

    # Disable validation if val_check_interval is negative
    validation_disabled = val_check_interval is not None and val_check_interval < 0
   
    # Create dataloaders - use values from experiment_config
    train_loader, val_loader, tokenizer = create_dataloaders(
        group_by_program=experiment_config.group_by_program,
        batch_size=experiment_config.batch_size,
        data_multiplier=experiment_config.data_multiplier,
        ds_type=experiment_config.dataset,
        padding_idx=experiment_config.model_config.padding_idx,
        num_train_examples=experiment_config.limit_training_samples,
        num_val_examples=experiment_config.limit_validation_samples,
        max_grid_height=experiment_config.model_config.grid_height,
        max_grid_width=experiment_config.model_config.grid_width,
    )

    experiment_config.model_config.program_vocab = len(tokenizer)
    experiment_config.set_em_start_step(len(train_loader))

    print(experiment_config)
    print("EM start step: ", experiment_config.em_start_step)

    if validation_disabled:
        print("Validation disabled. Checkpoints will not be saved.")
        print("Training batches won't be permuted either.")
        val_loader = None
        val_check_interval = None

    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=project_name,
        name=run_name,
        id=run_name,
        version=run_name,
        log_model=False,
        save_dir=checkpoint_dir,
        reinit=True,
        mode="disabled" if not wandb_logging else "online",
        config=experiment_config.to_dict()
    )

    lr_monitor = LearningRateMonitor(logging_interval="step", log_momentum=True, log_weight_decay=True)   # or "epoch"


    # Only add checkpoint callbacks if validation is enabled
    callbacks = [ CustomRichProgressBar(), lr_monitor]

    # if debug_mode:
    #     gradient_check_callback = GradientCheckCallback()
    #     callbacks.append(gradient_check_callback)
    
    if not validation_disabled:
        callbacks.extend([
            ModelCheckpointWithWandbSync(
                wandb_model_suffix="best",
                monitor='Prediction/loss_val',
                save_top_k=3,
                mode='min',
                auto_insert_metric_name=False,
                filename='best-step{step:07d}-Loss{Prediction/loss_val:.4f}',
                wandb_verbose=debug_mode
            ),
            ModelCheckpointWithWandbSync(
                wandb_model_suffix="backup",
                monitor='step',
                mode='max',
                save_top_k=2,
                every_n_train_steps=20 if debug_mode else val_check_interval,
                auto_insert_metric_name=False,
                filename='backup-step{step:07d}-Loss{Prediction/loss_val:.4f}',
                wandb_verbose=debug_mode
            )
        ])


    # profiler = pl.profilers.PyTorchProfiler()

    trainer = pl.Trainer(
        # profiler=profiler,
        default_root_dir=tempfile.gettempdir() if lr_find else None,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        accelerator=acceleration.device,
        precision=acceleration.precision,
        devices='auto',
        logger=wandb_logger,
        gradient_clip_val=experiment_config.gradient_clip_val,
        accumulate_grad_batches=experiment_config.accumulate_grad_batches,
        callbacks=callbacks,
        max_epochs=-1,
        max_steps=experiment_config.max_steps,
        limit_train_batches=100 if lr_find else None,
        limit_val_batches=0 if lr_find or validation_disabled else (10 if debug_mode else None),
        val_check_interval=None if lr_find or validation_disabled else val_check_interval,
        enable_model_summary=not lr_find
        # detect_anomaly=True if debug_mode else False
    )

    with trainer.init_module():
        # Initialize the model
        model = ProSIPTrainingModule(
            experiment_config,
            tokenizer,
            compile_model=acceleration.compile_model
        )

        # Load weights if a model source is specified
        if experiment_config.model_src:
            load_model_weights(model, experiment_config.model_src, project_name, checkpoint_dir)
        # wandb_logger.watch(model, log='parameters', log_graph=True, log_freq=100)

    if lr_find:
        tuner = Tuner(trainer)
        # For lr_find, only use train_loader
        lr_finder = tuner.lr_find(
            model,
            train_loader,
            val_dataloaders=None,  # Disable validation during lr_find
            update_attr=False
        )
        
        print("\nLearning rate finder results:")
        print(f"Suggested learning rate: {lr_finder.suggestion()}")
        
        fig = lr_finder.plot(suggest=True)
        output_file = os.path.join(os.getcwd(), f"lr_finder_{run_name}.png")
        fig.savefig(output_file)
        plt.close(fig)  # Close the figure to free memory
        return

    # Parameter counting moved to on_train_start method

    trainer.fit(
        model, 
        train_loader, 
        val_loader, 
        ckpt_path=resume_from
    )

if __name__ == "__main__":

    train_loader, val_loader, tokenizer = create_dataloaders(batch_size=64, 
                                                  data_multiplier=1, 
                                                  ds_type=DatasetType.ALL, 
                                                  padding_idx=15,
                                                  max_grid_height=32, 
                                                  max_grid_width=32,
                                                  num_workers=0,
                                                  num_val_examples=1000)
    batch = next(iter(train_loader))

    input_grids, output_grids, program_ids, attributes = batch

    print(input_grids.shape)
    print(output_grids.shape)
    print(program_ids.shape)
    # print(attributes)

    prosip_config = ProSIPConfig(program_vocab=len(tokenizer))

    model = ProSIPModel(prosip_config)
    # model = torch.torch.compile(
    #             model,
    #             fullgraph=True,
    #             mode="reduce-overhead",
    #             backend="inductor"
    #         )

    for i in range(10):
        print(f"Iteration {i}")
        output = model(input_grids, output_grids, program_ids)


    # experiment_config = ProSIPExperimentConfig(batch_size=4)
    # run_name = "prosip-test"
    # project_name = "prosip-test-project"
    # checkpoint_dir = Path("checkpoints")
    # debug_mode = False
    # wandb_logging = True
    # val_check_interval = 100

    # print(experiment_config)

    # acceleration = AccelerationConfig(
    #     compile_model=False
    # )

    # train(experiment_config, 
    #       run_name, 
    #       project_name, 
    #       checkpoint_dir, 
    #       debug_mode, 
    #       wandb_logging, 
    #       val_check_interval, 
    #       resume_from=None, 
    #       acceleration=acceleration, 
    #       lr_find=False, 
    #       grad_log_interval=100, 
    #       visualization_interval=100, 
    #       num_grids_to_visualize=4)