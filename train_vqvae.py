from functools import partial
import os
from typing import Dict, Any, Optional
import numpy as np
from dataclasses import dataclass, field
import tempfile
import torch
import time
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner.tuning import Tuner
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
from pips.grid_dataset import DatasetType, GridDataset, worker_init_fn
from pips.utils import generate_friendly_name
from pips.vq_vae import VQVAEConfig, VQVAE
from pips.misc.artifact import Artifact
from pips.misc.custom_progress_bar import CustomRichProgressBar
from pips.misc.schedule import Schedule  # Add this import
from pips.misc.checkpoint_with_wandb_sync import ModelCheckpointWithWandbSync
from pathlib import Path
from torch.serialization import add_safe_globals  # Add this import at the top
import wandb
import sys
import matplotlib.pyplot as plt
from pips.misc.acceleration_config import AccelerationConfig
from pips.misc.gradient_check_callback import GradientCheckCallback
import zlib
import random
from rich import print
import math
import yaml
import typer

import matplotlib

# This ensures that figures are rendered off‑screen and typically faster.
matplotlib.use('Agg')

# Initialize Typer app
app = typer.Typer(
    name="pips",
    help="Perception Informed Program Synthesis CLI",
    add_completion=False,
    pretty_exceptions_show_locals=False
)

class LoggingCallback(pl.Callback):
    
    def __init__(self, visualization_interval=50, save_to_disk=False, visualization_dir=None, grad_log_interval=100, num_grids_to_visualize=4):
        self.train_batch_start_time = None
        self.val_batch_start_time = None
        self.visualization_interval = visualization_interval  # Visualize every N steps
        self.save_to_disk = save_to_disk
        self.visualization_dir = Path(visualization_dir) if visualization_dir is not None else Path("visualizations")
        self.grad_log_interval = grad_log_interval  # How often to log gradient norms
        self.num_grids_to_visualize = num_grids_to_visualize  # Number of grids to visualize
        if save_to_disk:
            self.visualization_dir.mkdir(exist_ok=True, parents=True)
        self.val_batch_to_visualize = None
        # Initialize last visualization step so that logging happens immediately at start.
        self.last_logged_visualization = -visualization_interval
        

    def get_loss_string(self, outputs: Dict[str, torch.Tensor]) -> str:
        return ' | '.join([f"{l}: {v:.2e}" for l, v in outputs.items() if 'loss' in l or 'accuracy(TOKENS)' in l])
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=None):
        # Record the start time of the training batch.
        # If on GPU, record a CUDA event; otherwise, use time.monotonic().
        if torch.cuda.is_available():
            self.train_batch_start_time = torch.cuda.Event(enable_timing=True)
            self.train_batch_start_time.record()
        else:
            self.train_batch_start_time = time.monotonic()

    def _calculate_tokens_per_sec(self, start, batch):
        if start is not None:
            if isinstance(start, torch.cuda.Event):
                # Use a CUDA event for timing.
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                end_event.synchronize()  # Ensure all CUDA kernels have finished executing.
                # Get elapsed time in milliseconds (this waits only on the two events, not the entire GPU).
                elapsed_time_ms = start.elapsed_time(end_event)
                elapsed_time = elapsed_time_ms / 1000.0
            else:
                elapsed_time = time.monotonic() - start
            x, _ = batch
            num_tokens = x.size(0) * x.size(1)  # batch_size * tokens_per_batch
            tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0
            time_per_batch_ms = elapsed_time * 1000  # Convert to milliseconds
            return tokens_per_sec, time_per_batch_ms
        return 0, 0

    def visualize_reconstructions(self, pl_module, x, logits, phase):
        """Create visualization of input grids and their reconstructions."""
        if pl_module.global_rank != 0:
            return

        # Skip if not using WandB and not saving to disk.
        should_log_wandb = isinstance(pl_module.logger, WandbLogger)
        if not (should_log_wandb or self.save_to_disk):
            return

        if x is None or logits is None:
            return
        # Take a subset of the batch for visualization (e.g., 4 samples)
        n_samples = min(self.num_grids_to_visualize, x.size(0))
        x_subset = x[:n_samples]
        logits_subset = logits[:n_samples]
        
        # Get reconstructions from logits.
        reconstructions = logits_subset.argmax(dim=-1)
        
        # Reshape inputs and reconstructions to 2D grids.
        # Get grid dimensions from model config.
        height = pl_module.model_config.max_grid_height
        width = pl_module.model_config.max_grid_width
        
        # Reshape tensors to [batch, height, width].
        x_reshaped = x_subset.reshape(n_samples, height, width)
        recon_reshaped = reconstructions.reshape(n_samples, height, width)
        
        # Convert tensors to numpy arrays.
        x_np = x_reshaped.cpu().numpy()
        recon_np = recon_reshaped.cpu().numpy()
        
        # Define specific colors for values 0-9 and 15.
        color_map = {
            0: '#000000',  # black
            1: '#0074D9',  # blue
            2: '#FF4136',  # red
            3: '#2ECC40',  # green
            4: '#FFDC00',  # yellow
            5: '#AAAAAA',  # grey
            6: '#F012BE',  # fuschia
            7: '#FF851B',  # orange
            8: '#7FDBFF',  # teal
            9: '#870C25',  # brown
            15: '#FFFFFF'  # white
        }
        
        # Default color for other values (obnoxious pink).
        default_color = '#FF00FF'
        
        # Get all unique values from both input and reconstruction.
        all_values = np.unique(np.concatenate([x_np.flatten(), recon_np.flatten()]))
        
        # Create a custom colormap.
        import matplotlib.colors as mcolors
        colors = []
        for val in range(max(all_values.max() + 1, 16)):  # Ensure we have at least 16 colors (for value 15)
            if val in color_map:
                colors.append(color_map[val])
            else:
                colors.append(default_color)
        
        custom_cmap = mcolors.ListedColormap(colors)
        
        # Create the figure with proper space for a colorbar.
        fig = plt.figure(figsize=(n_samples * 3 + 1, 6), dpi=80)
        
        # Create a gridspec layout with space for the colorbar.
        gs = fig.add_gridspec(2, n_samples + 1, width_ratios=[1] * n_samples + [0.1])
        
        # Create axes for the plots.
        axes = [[fig.add_subplot(gs[i, j]) for j in range(n_samples)] for i in range(2)]
        
        # Plot inputs on the top row.
        for i in range(n_samples):
            im = axes[0][i].imshow(x_np[i], cmap=custom_cmap, vmin=0, vmax=len(colors) - 1)
            axes[0][i].set_title(f"Input {i+1}")
            axes[0][i].axis('off')
            
        # Plot reconstructions on the bottom row.
        for i in range(n_samples):
            axes[1][i].imshow(recon_np[i], cmap=custom_cmap, vmin=0, vmax=len(colors) - 1)
            axes[1][i].set_title(f"Reconstruction {i+1}")
            axes[1][i].axis('off')
            
        # Add a colorbar as a legend using the gridspec.
        cbar_ax = fig.add_subplot(gs[:, -1])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Grid Values')
        
        # Add ticks for the values that appear in the data.
        present_values = sorted(np.unique(np.concatenate([x_np.flatten(), recon_np.flatten()])))
        cbar.set_ticks(present_values)
        cbar.set_ticklabels([str(int(v)) for v in present_values])
        
        # Use tight_layout without a rect parameter.
        plt.tight_layout()
        
        # Log to wandb if available.
        if should_log_wandb:
            pl_module.logger.experiment.log({
                f'Reconstructions/{phase}': wandb.Image(fig)
            })
        
        # Save to disk if requested.
        if self.save_to_disk:
            filename = f"{phase}_step_{pl_module.global_step:07d}.png"
            save_path = self.visualization_dir / filename
            fig.savefig(save_path)
            print(f"Saved visualization to {save_path}")
        
        plt.close(fig)  # Close the figure to free memory

    def _create_metric_suffix(self, outputs, current_step):
        """
        Creates a standardized suffix for metrics with consistent formatting.
        
        Args:
            outputs: Dictionary containing model outputs
            current_step: Current global step
            
        Returns:
            String suffix to append to metric keys
        """
        suffix_parts = []
        
        # Add tau if available
        if 'tau' in outputs:
            suffix_parts.append(f"tau_{outputs['tau'].item():.4f}")
            
        # Add global step
        suffix_parts.append(f"step_{current_step}")
        
        # Add gumbel noise scale if available
        if 'gumbel_noise_scale' in outputs:
            suffix_parts.append(f"gns_{outputs['gumbel_noise_scale'].item():.4f}")
            
        # Combine parts with underscores and add a leading underscore
        return "_" + "_".join(suffix_parts) if suffix_parts else ""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        x = outputs.pop('input')
        logits = outputs.pop('logits')

        # Instead of using an exact modulo, check if it's time to log.
        current_step = pl_module.global_step
        should_visualize = (current_step - self.last_logged_visualization) >= self.visualization_interval

        # Visualize reconstructions if the time interval has been reached.
        if should_visualize:
            self.visualize_reconstructions(pl_module, x, logits, 'train')
            self.last_logged_visualization = current_step

        # Calculate tokens per second for the training batch.
        if self.train_batch_start_time is not None:
            tokens_per_sec, time_per_batch_ms = self._calculate_tokens_per_sec(self.train_batch_start_time, batch)
            print(f"\n[Train] {self.get_loss_string(outputs)} | T/s: {tokens_per_sec:.2f} | Δ(ms): {time_per_batch_ms:.1f}ms")
            outputs['tokens_per_sec'] = tokens_per_sec
            outputs['Δ_ms'] = time_per_batch_ms
        
        # Log the current learning rate.
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        pl_module.log('params/learning_rate', current_lr, on_step=True, on_epoch=False)
        
        # Log loss metrics using the helper method.
        self._log_metrics(pl_module, 'train', outputs, batch[0].size(0), on_step=True, on_epoch=False)

    def on_validation_epoch_start(self, trainer, pl_module):
        # Randomly select a batch index to visualize during this validation epoch.

        val_dataloader = None
        if trainer.val_dataloaders is not None:
            if isinstance(trainer.val_dataloaders, list):
                val_dataloader = trainer.val_dataloaders[0] if len(trainer.val_dataloaders) > 0 else None
            else:
                # In some PyTorch Lightning versions, val_dataloaders might be a single dataloader
                val_dataloader = trainer.val_dataloaders
        
        if val_dataloader is not None:
            self.val_batch_to_visualize = np.random.randint(0, len(val_dataloader))
        else:
            # No validation dataloader available.
            self.val_batch_to_visualize = None

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=None):
        # Record the start time of the validation batch.
        if torch.cuda.is_available():
            self.val_batch_start_time = torch.cuda.Event(enable_timing=True)
            self.val_batch_start_time.record()
        else:
            self.val_batch_start_time = time.monotonic()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        x = outputs.pop('input')
        logits = outputs.pop('logits')

        # Visualize reconstructions only for the randomly selected batch.
        if batch_idx == self.val_batch_to_visualize:
            self.visualize_reconstructions(pl_module, x, logits, 'val')
        

        # Calculate tokens per second for the validation batch.
        if self.val_batch_start_time is not None:
            tokens_per_sec, time_per_batch_ms = self._calculate_tokens_per_sec(self.val_batch_start_time, batch)
            print(f"\n[Eval]  {self.get_loss_string(outputs)} | T/s: {tokens_per_sec:.2f} | Δ(ms): {time_per_batch_ms:.1f}ms")
            outputs['tokens_per_sec'] = tokens_per_sec
            outputs['Δ_ms'] = time_per_batch_ms
        
        # Log loss metrics using the helper method.
        self._log_metrics(pl_module, 'val', outputs, batch[0].size(0), on_step=False, on_epoch=True)

    def _log_metrics(self, pl_module: pl.LightningModule, phase: str, outputs: Dict[str, torch.Tensor], batch_size: int, on_step: bool, on_epoch: bool):
        """Helper method to log loss metrics."""
        for key, value in outputs.items():
            # Handle the figure separately.
            if key.startswith('CodebookUsage/latent_distribution'):
                if isinstance(pl_module.logger, WandbLogger) and pl_module.global_rank == 0:
                    log_key = f'{key}_{phase}'
                    pl_module.logger.experiment.log({
                        log_key: wandb.Image(value)
                    })
                plt.close(value)  # Close the figure to free memory
                continue

            # Handle the main loss separately.
            if key == 'loss':
                metric_name = f'TotalLoss/{key}_{phase}'  # Default format.
            # Handle metrics with categories - loss(CE), loss(MI), etc.
            elif '(' in key and ')' in key:
                category = key.split('(')[-1].split(')')[0]  # Extract category.
                metric_name = f'{category}/{key.split("(")[0]}_{phase}'  # Format as "category/metric_phase"
            # Handle special parameters like 'hard', 'tau', 'beta', etc.
            elif key in ['mask_pct', 'max_mask_pct']:
                metric_name = f'params/{key}_{phase}'  # Group parameters under 'params/'.
            elif key in ['tokens_per_sec', 'Δ_ms']:
                metric_name = f'Throughput/{key}_{phase}'
            elif 'Codebook' in key:
                metric_name = f'{key}_{phase}'
            # Handle any remaining metrics.
            elif key.strip():
                metric_name = f'{key.capitalize()}/{key}_{phase}'
            
            sync_dist = False if phase == 'train' else True
            pl_module.log(metric_name, value, on_step=on_step, on_epoch=on_epoch, batch_size=batch_size, logger=True, sync_dist=sync_dist)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Log gradient norms at specified intervals."""
        # Compute the 2-norm for each layer.
        # If using mixed precision, the gradients are already unscaled here.
        if pl_module.global_rank == 0 and trainer.global_step % self.grad_log_interval == 0:
            norms = grad_norm(pl_module.model, norm_type=2)
            pl_module.log_dict(norms)

@dataclass
class ExperimentConfig:
    """Configuration for training hyperparameters and schedules"""
    # Model configuration
    model_config: VQVAEConfig = field(default_factory=VQVAEConfig)
    
    # General training parameters
    seed: int | None = None  # None means random seed
    batch_size: int = 64
    max_steps: int = 1_000_000
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    
    # Schedule types for all parameters
    mask_schedule_type: str = 'cosine'
    
    # Learning rate parameters
    learning_rate: float = 1e-4  # Consistent with Dalle-E paper
    lr_min: float = 1e-6  # Minimum learning rate to be reached after decay
    warmup_steps_lr: int = 10_000
    decay_steps_lr: int | None = None
    weight_decay: float = 1e-4  # Consistent with Dalle-E paper
    
    # Commitment loss parameters
    beta_commitment: float = 1.0
    
    # Mask percentage parameters
    mask_pct_start: float = 0.0
    max_mask_pct: float = 0.0
    transition_steps_mask_pct: int = 50_000
    warmup_steps_mask_pct: int = 0
    
    # Dataset parameters
    train_ds: DatasetType = DatasetType.TRAIN
    val_ds: DatasetType = DatasetType.VAL
    limit_training_samples: int | None = None  # Limit the number of training samples. None means use all samples.
    permute_train: bool = True  # Whether to permute the training data
    
    # Other parameters
    model_src: str | None = None

    def __post_init__(self):
        if self.accumulate_grad_batches < 1:
            raise ValueError("accumulate_grad_batches must be >= 1")
            
        # Generate random seed if none provided
        if self.seed is None:
            self.seed = np.random.randint(0, 2**32 - 1)

        # Cap warmup steps at max_steps and ensure warmup + transition <= max_steps
      
        # For mask percentage parameters
        self.warmup_steps_mask_pct = min(self.warmup_steps_mask_pct, self.max_steps)
        remaining_steps = self.max_steps - self.warmup_steps_mask_pct
        self.transition_steps_mask_pct = min(self.transition_steps_mask_pct, remaining_steps)
        
        # For learning rate (special case - using original name)
        self.warmup_steps_lr = min(self.warmup_steps_lr, self.max_steps)
        if self.decay_steps_lr is None:
            self.decay_steps_lr = self.max_steps - self.warmup_steps_lr

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
    def from_dict(cls, config_dict: dict) -> 'ExperimentConfig':
        """Create config from a dictionary."""
        # Create a copy to avoid modifying the input
        config = config_dict.copy()
        
        # Handle nested VQVAEConfig
        if isinstance(config.get('model_config'), dict):
            config['model_config'] = VQVAEConfig.from_dict(config['model_config'])
        
        # Handle DatasetType fields
        if 'train_ds' in config:
            config['train_ds'] = DatasetType[config['train_ds']]
        if 'val_ds' in config:
            config['val_ds'] = DatasetType[config['val_ds']]
            
        return cls(**config)

    @staticmethod
    def from_checkpoint(checkpoint_path: str) -> 'ExperimentConfig':
        """Load ExperimentConfig from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            ExperimentConfig: Configuration loaded from checkpoint
            
        Raises:
            ValueError: If checkpoint doesn't contain valid config
        """
        # Add our custom classes to safe globals
        add_safe_globals([ExperimentConfig, VQVAEConfig])
        
        # First try with weights_only=True for security
        try:
            ckpt = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        except Exception as e:
            print(f"Warning: Failed to load with weights_only=True ({str(e)}), falling back to weights_only=False")
            try:
                ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
            except Exception as e:
                raise ValueError(f"Failed to load checkpoint with both methods: {e}")
        
        try:
            config = ckpt['hyper_parameters']['experiment_config']
            if not isinstance(config, ExperimentConfig):
                raise ValueError("Checkpoint contains invalid config type")
            return config
            
        except KeyError as e:
            raise ValueError(f"Checkpoint doesn't contain valid config: {e}")

class VQVAETrainingModule(pl.LightningModule):
    def __init__(self, experiment_config: ExperimentConfig, compile_model: bool = False):
        super(VQVAETrainingModule, self).__init__()
        self.experiment_config = experiment_config
        self.padding_idx = experiment_config.model_config.padding_idx
        self.model_config = experiment_config.model_config
        self.model = None
        self.learning_rate = experiment_config.learning_rate
        self.compile_model = compile_model
        self.save_hyperparameters()

    def configure_model(self):
        """
        Compile the model after device placement.
        This gets called in on_fit_start so that the model is already on GPU.
        """
        if self.model is not None:
            return

        self.model = VQVAE(self.model_config)
        
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

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Handle loading checkpoints with different state dict keys."""
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # If we're using compile and the checkpoint wasn't compiled
            if self.compile_model and not any("_orig_mod" in key for key in state_dict.keys()):
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("model."):
                        new_key = key.replace("model.", "model._orig_mod.")
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                checkpoint["state_dict"] = new_state_dict
            # If we're not using compile but the checkpoint was compiled
            elif not self.compile_model and any("_orig_mod" in key for key in state_dict.keys()):
                new_state_dict = {}
                for key, value in state_dict.items():
                    if "_orig_mod" in key:
                        new_key = key.replace("model._orig_mod.", "model.")
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                checkpoint["state_dict"] = new_state_dict

    def get_scheduled_values(self, step: int, device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
        """Returns all scheduled values for the current step as tensors on the specified device.
        
        Args:
            step (int): Current training step.
            device (torch.device): The device on which to allocate the scheduled value tensors. Defaults to CPU.
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing scheduled values (tau, beta(CE), beta(MI), beta(TC),
            beta(DWKL), beta(KL), max_pct(MASK)) as tensors.
        """
        cfg = self.experiment_config
        
        # Max mask percentage schedule
        max_mask_pct_schedule = Schedule.get_schedule(
            initial_value=cfg.mask_pct_start,
            target_value=cfg.max_mask_pct,
            transition_steps=cfg.transition_steps_mask_pct,
            warmup_steps=cfg.warmup_steps_mask_pct,
            schedule_type=cfg.mask_schedule_type
        )
        
        return {
            'max_pct(MASK)': torch.tensor(max_mask_pct_schedule(step), device=device, dtype=torch.float32),
        }

    def forward(self, x: Tensor,
                beta_commitment: Tensor = torch.tensor(1.0),
                mask_pct: Tensor = torch.tensor(0.0)):
        
        # Forward pass with provided scheduled parameters.
        logits, losses = self.model.forward(x, mask_percentage=mask_pct)
        
        # Calculate token accuracy excluding padding tokens.
        non_padding_mask = (x != self.padding_idx)
        masked_correct_tokens = logits.argmax(dim=-1) == x
        masked_correct_tokens = masked_correct_tokens * non_padding_mask
        token_accuracy = masked_correct_tokens.sum() / non_padding_mask.sum()
        
        # Calculate sample accuracy (all tokens must be predicted correctly).
        sample_correct = (logits.argmax(dim=-1) == x).all(dim=1).float()
        sample_accuracy = sample_correct.mean()
        
        
        # Compute total loss in a way that maintains the computational graph.
        # Scale the KLD losses by the number of latents (similar to Dalle-E paper).
        raw_losses = {
            'loss(CE)': losses['ce_loss'],
            'loss(VQ)': losses['vq_loss'],
            'loss(Commitment)': losses['commitment_loss'],
        }
        
        # Compute weighted losses for total loss.
        weighted_losses = {
            'loss(CE)': raw_losses['loss(CE)'],
            'loss(VQ)': raw_losses['loss(VQ)'],
            'loss(Commitment)': raw_losses['loss(Commitment)'] * beta_commitment
        }
        
        total_loss = sum(weighted_losses.values())
        
        # Return tensor values; logging callbacks can detach/convert them as needed.
        return {
            'loss': total_loss,
            **raw_losses,
            'accuracy(TOKENS)': token_accuracy,
            'accuracy(SAMPLES)': sample_accuracy,
            'logits': logits,  # Add logits to the output dictionary.
            'input': x,        # Add input to the output dictionary.
        }

    def training_step(self, batch, batch_idx):
        torch.compiler.cudagraph_mark_step_begin()
        x, _ = batch

        # Extract scheduled values and convert them to tensors on the current device.
        scheduled = self.get_scheduled_values(self.global_step, device=x.device)

        beta_commitment = torch.tensor(self.experiment_config.beta_commitment, device=x.device)
        # Sample mask percentage for this batch; set max_pct_mask to 0 in validation for no masking.
        mask_pct = torch.empty(1, device=x.device).uniform_(0.0, scheduled['max_pct(MASK)'])[0]

        output_dict = self(
             x,
             beta_commitment=beta_commitment,
             mask_pct=mask_pct
        )

        output_dict['percent(MASK)'] = mask_pct
        output_dict['beta(Commitment)'] = beta_commitment
        output_dict.update(scheduled)
        
        return output_dict

    def validation_step(self, batch, batch_idx):
        torch.compiler.cudagraph_mark_step_begin()
        x, _ = batch

        # For validation, scheduled values should be passed as provided (e.g., max_pct_mask can be set to 0 for no masking).
        scheduled = self.get_scheduled_values(self.global_step, device=x.device)
        beta_commitment = torch.tensor(self.experiment_config.beta_commitment, device=x.device)
        mask_pct = torch.tensor(0.0, device=x.device) # No masking in validation

        output_dict = self(
             x,
             beta_commitment=beta_commitment,
             mask_pct=mask_pct
        )

        output_dict.update(scheduled)
        output_dict['beta(Commitment)'] = beta_commitment
        output_dict.pop('max_pct(MASK)') # Remove max_pct(MASK) from output_dict

        return output_dict

    def configure_optimizers(self):
        # Get all parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Split parameters based on dimensionality
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]

        # This excludes biases and BatchNorm or LayerNorm
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': self.experiment_config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(
            optim_groups,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=True if torch.cuda.is_available() else False
        )
        
        # Noam scheduler with linear warmup and cosine decay using lr-specific warmup
        def lr_lambda(step):
            warmup_steps = self.experiment_config.warmup_steps_lr
            min_lr_factor = self.experiment_config.lr_min / self.experiment_config.learning_rate  # Use config value
            
            if step < warmup_steps:
                # Linear warmup
                return float(step) / float(max(1, warmup_steps))
            elif step < self.experiment_config.decay_steps_lr + warmup_steps:
                # Cosine decay from 1.0 to min_lr_factor
                progress = float(step - warmup_steps) / float(max(1, self.experiment_config.max_steps - warmup_steps))
                return min_lr_factor + 0.5 * (1.0 - min_lr_factor) * (1.0 + np.cos(np.pi * progress))
            else:
                return min_lr_factor
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

def create_dataloaders(
    batch_size: int,
    padding_idx: int,
    max_grid_height: int,
    max_grid_width: int,
    permute_train: bool,
    limit_training_samples: int | None,
    min_training_samples: int | None,
    train_ds: DatasetType,
    val_ds: DatasetType,
    num_measure_samples: int = 10000,  # Number of samples to measure average information in bits
):
    """Create train and validation dataloaders and measure average grid information bits.

    Args:
        batch_size: Number of samples per batch.
        padding_idx: Padding index to be used in the grids.
        max_grid_height: Maximum grid height.
        max_grid_width: Maximum grid width.
        permute_train: If True, permute the training data.
        limit_training_samples: Maximum number of training samples to use. None means use all samples.
        min_training_samples: Minimum number of training samples to use. None means use all samples.
        train_ds: Training dataset type.
        val_ds: Validation dataset type.
        num_measure_samples: Number of grid samples to use for measuring average compressed bits.
    """
    print("permute_train:", permute_train)

    # Create training dataloader
    collate_fn_train = partial(
        GridDataset.collate_fn_project,
        pad_value=padding_idx,
        permute=permute_train,  # Use the permute_train parameter
        max_height=max_grid_height,
        max_width=max_grid_width
    )

    train_dataset = GridDataset(dataset_type=train_ds, max_samples=limit_training_samples, min_samples=min_training_samples)

    # Create validation dataset
    collate_fn_val = partial(
        GridDataset.collate_fn_project,
        pad_value=padding_idx,
        permute=False,
        max_height=max_grid_height,
        max_width=max_grid_width
    )
    val_dataset = GridDataset(dataset_type=val_ds)

    # Measure average compressed bits directly from the datasets (not using the dataloader)
    train_stats = measure_bits_stats_from_dataset(train_dataset, num_samples=num_measure_samples)
    val_stats = measure_bits_stats_from_dataset(val_dataset, num_samples=num_measure_samples)
    max_two_std_info_bits = max(train_stats['upper_bound'], val_stats['upper_bound'])


    # Print stats
    print("Train Grid Info Bits:", train_stats)
    print("Val Grid Info Bits:", val_stats)
    print("Max two std info bits:", max_two_std_info_bits)

    color_perm_bits = math.log2(math.factorial(10))  # 10 colors (0-9) permutation entropy
    array_transform_bits = math.log2(8)  # 8 possible array transformations
    min_bits_required = max_two_std_info_bits + color_perm_bits + array_transform_bits

    print("Color permutation bits:", color_perm_bits)
    print("Array transform bits:", array_transform_bits)
    print("Total augmentation bits:", color_perm_bits + array_transform_bits)
    print("Minimum bits required:", min_bits_required)

    # The reset is necessary so that the dataset is not cached in the workers.
    train_dataset.unload()
    val_dataset.unload()


    # Proceed to create DataLoader objects for training and validation
    num_workers = min(8, os.cpu_count() or 1)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_train,
        shuffle=True, 
        num_workers=num_workers,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_val,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        drop_last=True  # Drop last batch to avoid errors on compilation
    )

    print("Number of batches in training set: ", len(train_loader))
    print("Number of batches in validation set: ", len(val_loader))

    return train_loader, val_loader, min_bits_required

def measure_bits_stats_from_dataset(dataset: GridDataset, num_samples: int) -> dict:
    """
    Measures the compressed bits for grids in the dataset by selecting random samples,
    then computes statistics: overall mean, median, standard deviation, and a trimmed mean
    (mean computed on values within 2 standard deviations of the overall mean).

    Args:
        dataset: A GridDataset where each item has an 'array' attribute representing the grid.
        num_samples: The number of grid samples to use for the measurement.
    
    Returns:
        A dictionary with the following keys:
            'mean'         : Mean of all sample bits.
            'median'       : Median of all sample bits.
            'std'          : Standard deviation of all sample bits.
            'lower_bound'  : Lower bound of the range within 2 standard deviations of the overall mean.
            'upper_bound'  : Upper bound of the range within 2 standard deviations of the overall mean.
    """
    import numpy as np

    sample_bits = []
    total_samples = min(num_samples, len(dataset))
    # Generate a list of random indices from the dataset
    indices = random.sample(range(len(dataset)), total_samples)
    
    for i in indices:
        grid = dataset[i].array
        # If grid is a PyTorch tensor, convert it to a NumPy array.
        grid_array = grid.numpy() if hasattr(grid, "numpy") else grid
        bits = compress_grid(grid_array)
        sample_bits.append(bits)
    
    sample_bits_arr = np.array(sample_bits)
    overall_mean = np.mean(sample_bits_arr)
    overall_median = np.median(sample_bits_arr)
    overall_std = np.std(sample_bits_arr)
    
    # Filter values that fall within 2 standard deviations of the overall mean
    lower_bound = max(0, overall_mean - 2 * overall_std)
    upper_bound = overall_mean + 2 * overall_std    

    return {
        'mean': float(overall_mean),
        'median': float(overall_median),
        'std': float(overall_std),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound)
    } 

def compress_grid(grid):
    """
    Compresses a grid using zlib and returns its size in bits.
    
    Args:
        grid: A grid object with a tobytes() method.
        
    Returns:
        The size of the compressed grid in bits.
    """
    grid_bytes = grid.tobytes()
    compressed = zlib.compress(grid_bytes)
    return len(compressed) * 8  # bits

def load_model_weights(
    model: pl.LightningModule,
    model_src: str,
    project_name: str,
    checkpoint_dir: Path
) -> None:
    """Load model weights from a remote artifact.
    
    Args:
        model: The model to load weights into
        model_src: Artifact string in format [project/]run_name/{best|backup}[/{alias|step}] where:
            - alias can be 'best', 'best-N', 'latest', 'step-NNNNNNN'
            - step is a positive integer that will be converted to 'step-NNNNNNN' format
        project_name: Default project name if not specified in model_src
        checkpoint_dir: Directory to store downloaded checkpoints
        
    Raises:
        ValueError: If artifact cannot be found or loaded
        SystemExit: If no alias is specified (after displaying available checkpoints)
    """
    try:
        # Parse model source string first
        source_project, run_name, category, alias = Artifact.parse_artifact_string(
            model_src,
            default_project=project_name
        )
        
        # Initialize artifact manager with correct project and run name
        artifact_manager = Artifact(
            entity=wandb.api.default_entity,
            project_name=source_project,
            run_name=run_name
        )

        # Get local checkpoint path
        checkpoint_path = artifact_manager.get_local_checkpoint(
            category=category,
            alias=alias,
            checkpoint_dir=Path(checkpoint_dir)
        )
        
        # Load just the model weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.configure_model()  # Need to load the model first
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        print(f"Loaded model weights from {checkpoint_path}")
        
    except ValueError as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)

def train(
    experiment_config: ExperimentConfig,
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
    train_loader, val_loader, min_bits_required = create_dataloaders(
        batch_size=experiment_config.batch_size,
        padding_idx=experiment_config.model_config.padding_idx,
        max_grid_height=experiment_config.model_config.max_grid_height,
        max_grid_width=experiment_config.model_config.max_grid_width,
        permute_train=experiment_config.permute_train,
        limit_training_samples=experiment_config.limit_training_samples,
        min_training_samples=val_check_interval * experiment_config.batch_size if not validation_disabled else None,
        train_ds=experiment_config.train_ds,
        val_ds=experiment_config.val_ds
    )

    latent_bits = experiment_config.model_config.compute_latent_bits()
    print("Model Latent Code bits:", latent_bits)
    print("Model Latent Code bits / Minimum bits required:", latent_bits / min_bits_required)

    if latent_bits < min_bits_required:
        # Add a warning
        print("WARNING: Latent bits are less than minimum bits required. This means the model cannot use all the information in the data.")
        print("Try increasing the number of latent codes or the codebook size.")


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

    visualization_dir = Path(checkpoint_dir) / project_name / run_name / 'visualizations'

    # Define callbacks
    logging_callback = LoggingCallback(
        visualization_interval=visualization_interval,
        save_to_disk=False,
        visualization_dir=visualization_dir,
        grad_log_interval=grad_log_interval,  # Pass the gradient logging interval
        num_grids_to_visualize=num_grids_to_visualize  # Pass the number of grids to visualize
    )
    custom_progress_bar = CustomRichProgressBar()

    # Only add checkpoint callbacks if validation is enabled
    callbacks = [logging_callback, custom_progress_bar]

    # if debug_mode:
    #     gradient_check_callback = GradientCheckCallback()
    #     callbacks.append(gradient_check_callback)
    
    if not validation_disabled:
        callbacks.extend([
            ModelCheckpointWithWandbSync(
                wandb_model_suffix="best",
                monitor='CE/loss_val',
                save_top_k=3,
                mode='min',
                auto_insert_metric_name=False,
                filename='best-step{step:07d}-ce{CE/loss_val:.4f}-mi{MI/loss_val:.4f}-tc{TC/loss_val:.4f}-dwkl{DWKL/loss_val:.4f}-kl{KL/loss_val:.4f}',
                wandb_verbose=debug_mode
            ),
            ModelCheckpointWithWandbSync(
                wandb_model_suffix="backup",
                monitor='step',
                mode='max',
                save_top_k=2,
                every_n_train_steps=20 if debug_mode else val_check_interval,
                auto_insert_metric_name=False,
                filename='backup-step{step:07d}-ce{CE/loss_val:.4f}-mi{MI/loss_val:.4f}-tc{TC/loss_val:.4f}-dwkl{DWKL/loss_val:.4f}-kl{KL/loss_val:.4f}',
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
        enable_model_summary=not lr_find,
        # detect_anomaly=True if debug_mode else False
    )

    with trainer.init_module():
        # Initialize the model
        model = VQVAETrainingModule(
            experiment_config,
            compile_model=acceleration.compile_model
        )

        # Load weights if a model source is specified
        if experiment_config.model_src:
            load_model_weights(model, experiment_config.model_src, project_name, checkpoint_dir)
        wandb_logger.watch(model, log='all', log_graph=True, log_freq=100)

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

    trainer.fit(
        model, 
        train_loader, 
        val_loader, 
        ckpt_path=resume_from
    )

@app.command('export-config')
def export_default_train_args(output_path: Path = typer.Argument(..., help="Path to save the config YAML file")):
    """Export default arguments of the train function and ExperimentConfig to a YAML file.
    
    Args:
        output_path: Path where the YAML file will be saved
        
    Returns:
        Dictionary containing all default arguments
        
    Example:
        >>> export_default_train_args("train_defaults.yaml")
    """

    # Create a dictionary of default values
    defaults = {}
    
    # Get default ExperimentConfig
    default_config = ExperimentConfig()
    config_dict = default_config.to_dict()

    acceleration_config = AccelerationConfig()
    acceleration_dict = acceleration_config.to_dict()
    
    # Add experiment config as a nested dictionary
    defaults['experiment_config'] = config_dict
    defaults['acceleration_config'] = acceleration_dict
    
    # Add 'run_name', 'project_name', 'checkpoint_dir' under project_config
    defaults['project_config'] = {
        'run_name': generate_friendly_name(),
        'project_name': 'train-vq-vae',
        'checkpoint_dir': 'runs',
        'val_check_interval': 1000,
        'viz_interval': 1000,
    }

    # Write to YAML file
    with open(output_path, 'w') as f:
        yaml.safe_dump(defaults, f, default_flow_style=False, sort_keys=False)
    
    print(f"Default arguments exported to: {output_path}")
    return defaults


@app.command('train')
def new_train(
    config_path: Path = typer.Argument(..., help="Path to the config YAML file"),
    debug_mode: bool = typer.Option(False, "--debug", "-D", help="Enable debug mode"),
    lr_find: bool = typer.Option(False, "--lr-find", "-L", help="Enable learning rate finder"),
):
    """Train a model using configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML config file
        resume_from: Optional checkpoint path to resume training from
    """
    # Load the config file
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Extract experiment config and create ExperimentConfig instance
    experiment_config_dict = config_dict.pop('experiment_config')
    experiment_config = ExperimentConfig.from_dict(experiment_config_dict)

    # Extract acceleration config and create AccelerationConfig instance
    acceleration_config_dict = config_dict.pop('acceleration_config')
    acceleration_config = AccelerationConfig.from_dict(acceleration_config_dict)
    
    project_config = config_dict.pop('project_config')
    project_name = project_config['project_name']

    if debug_mode:
        project_name = f"{project_name}-debug"

    checkpoint_dir = Path(project_config['checkpoint_dir'])
    
    print(experiment_config.to_dict())

    # Call the train function with unpacked arguments   
    train(
        experiment_config=experiment_config,
        run_name=project_config['run_name'],
        project_name=project_name,
        checkpoint_dir=checkpoint_dir,
        debug_mode=debug_mode,
        val_check_interval=project_config['val_check_interval'],
        acceleration=acceleration_config,
        lr_find=lr_find,
        wandb_logging=True,
        grad_log_interval=1000,
        visualization_interval=project_config['viz_interval'],
        num_grids_to_visualize=4,
        resume_from=None
    )

if __name__ == '__main__':
    app()