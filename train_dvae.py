from functools import partial
import os
from typing import Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
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
from pips.dvae import GridDVAEConfig, GridDVAE
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

class LoggingCallback(pl.Callback):
    
    def __init__(self, visualization_interval=50, save_to_disk=False, visualization_dir=None, grad_log_interval=100, num_grids_to_visualize=4, ema_decay: float = 0.95):
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
        
        # Initialize EMAs for codebook usage tracking
        self.train_code_distribution_ema = None
        self.val_code_distribution_ema = None
        self.ema_decay = ema_decay  # EMA decay factor (higher = slower adaptation)

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

    def compute_entropy(self, log_alpha: Tensor, current_ema: Tensor = None, eps: float = 1e-8, add_codebook_usage: bool = True, ema_decay: float = 0.95):
        """
        Compute entropy metrics and update code distribution EMA.
        
        Args:
            log_alpha: Log alpha values from model output
            current_ema: Current EMA of code distribution (None for first call)
            eps: Small value for numerical stability
            add_codebook_usage: Whether to create and return a figure
            ema_decay: EMA decay factor (higher = slower adaptation)
            
        Returns:
            Tuple of (output_dict, updated_ema)
        """
        output_dict = {}
        # Calculate per-code perplexity
        normalized_log_alpha = F.log_softmax(log_alpha, dim=-1)
        probs = normalized_log_alpha.exp()

        per_code_entropy = -(probs * normalized_log_alpha).sum(dim=-1)  # [B, N]
        per_code_perplexity = torch.exp(per_code_entropy)  # [B, N]

        avg_perplexity = per_code_perplexity.mean()
        output_dict['CodebookUsage/perplexity'] = avg_perplexity.detach()

        # Average across batch dimension
        avg_per_code_perplexity = per_code_perplexity.mean(dim=0)  # [N]

        # Add per-code perplexity metrics
        for i, perp in enumerate(avg_per_code_perplexity):
            output_dict[f'Codebook/perplexity_code_{i}'] = perp.detach()
        
        # Calculate distribution for each code position (for current batch)
        current_code_distribution = probs.mean(dim=0).detach()  # [N, C]
        
        # Update EMA of code distribution
        if current_ema is None:
            # Initialize EMA with current distribution on first call
            updated_ema = current_code_distribution
        else:
            # Update EMA with current distribution
            updated_ema = ema_decay * current_ema + (1 - ema_decay) * current_code_distribution
        
        if add_codebook_usage and updated_ema is not None:
            # Use the EMA distribution for visualization
            # Convert to float32 before converting to numpy to avoid BFloat16 error
            code_dist_data = updated_ema.float().cpu().numpy()
            
            # Create the figure
            fig, ax = plt.subplots(figsize=(20, 15))
            im = ax.imshow(code_dist_data, aspect='auto', cmap='viridis')
            plt.colorbar(im)
            
            # Add labels
            ax.set_xlabel('Codebook Index')
            ax.set_ylabel('Position')
            ax.set_title('Code Usage Distribution (EMA)')
            
            # Add the figure to output_dict
            output_dict['Codebook/figure'] = fig

        return output_dict, updated_ema

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
        fig = plt.figure(figsize=(n_samples * 3 + 1, 6))
        
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

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        log_alpha = outputs.pop('log_alpha')
        x = outputs.pop('input')
        logits = outputs.pop('logits')

        # Instead of using an exact modulo, check if it's time to log.
        current_step = pl_module.global_step
        should_visualize = (current_step - self.last_logged_visualization) >= self.visualization_interval

        # Pass the current train EMA to compute_entropy and get the updated EMA back
        entropy_dict, self.train_code_distribution_ema = self.compute_entropy(
            log_alpha, 
            current_ema=self.train_code_distribution_ema,
            add_codebook_usage=should_visualize,
            ema_decay=self.ema_decay
        )
        outputs.update(entropy_dict)

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
        self.codebook_usage_figure_logged = False
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
        log_alpha = outputs.pop('log_alpha')
        x = outputs.pop('input')
        logits = outputs.pop('logits')

        # Only log the codebook usage figure once per epoch.
        # Pass the current val EMA to compute_entropy and get the updated EMA back
        entropy_dict, self.val_code_distribution_ema = self.compute_entropy(
            log_alpha, 
            current_ema=self.val_code_distribution_ema,
            add_codebook_usage=not self.codebook_usage_figure_logged,
            ema_decay=self.ema_decay
        )
        self.codebook_usage_figure_logged = True

        outputs.update(entropy_dict)

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
            if key == 'Codebook/figure':
                if isinstance(pl_module.logger, WandbLogger) and pl_module.global_rank == 0:  # Make sure we're using WandbLogger
                    # Only log every 20 global steps and only from rank 0.
                    pl_module.logger.experiment.log({
                        f'CodebookUsage/Usage_{phase}': wandb.Image(value)
                    })
                plt.close(value)  # Close the figure to free memory.
                continue

            # Handle the main loss separately.
            if key == 'loss':
                metric_name = f'TotalLoss/{key}_{phase}'  # Default format.
            # Handle metrics with categories - loss(CE), loss(MI), etc.
            elif '(' in key and ')' in key:
                category = key.split('(')[-1].split(')')[0]  # Extract category.
                metric_name = f'{category}/{key.split("(")[0]}_{phase}'  # Format as "category/metric_phase"
            # Handle special parameters like 'hard', 'tau', 'beta', etc.
            elif key in ['tau', 'mask_pct', 'max_mask_pct']:
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
    model_config: GridDVAEConfig
    
    # Add seed parameter
    seed: int | None = None  # None means random seed
    
    # Initial values (renamed from initial_*)
    tau_start: float = 3.5  # this is difference from Dalle-E paper which starts with 1.0. This is to match vanilla softmax.
    tau: float = 0.0625 # 1/16 as per Dalle-E paper
    beta_ce_start: float = 1.0  # Default to 1.0 to maintain original behavior
    beta_ce: float = 1.0  # Default to 1.0 to maintain original behavior
    beta_mi_start: float = 0.0
    beta_tc_start: float = 0.0
    beta_dwkl_start: float = 0.0
    beta_kl_start: float = 0.0
    mask_pct_start: float = 0.0
    
    # Final values
    beta_mi: float = 0.0
    beta_tc: float = 6.0
    beta_dwkl: float = 0.0
    beta_kl: float = 2.0
    max_mask_pct: float = 0.5

    
    # Schedule types
    tau_schedule_type: str = 'cosine'
    beta_schedule_type: str = 'cosine'
    mask_schedule_type: str = 'cosine'
    
    # Replace single warmup_steps with separate warmups
    warmup_steps_lr: int = 10_000
    decay_steps_lr: int | None = None
    warmup_steps_tau: int = 150_000
    warmup_steps_beta: int = 10_000
    warmup_steps_mask_pct: int = 50_000

    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-4 # Consistent with Dalle-E paper
    lr_min: float = 1e-6 # Minimum learning rate to be reached after decay
    weight_decay: float = 1e-4 # Consistent with Dalle-E paper
    max_steps: int = 1_000_000
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1

    # Dataset parameters
    train_ds: DatasetType = DatasetType.TRAIN
    val_ds: DatasetType = DatasetType.VAL
    limit_training_samples: int | None = None  # Limit the number of training samples. None means use all samples.
    permute_train: bool = True  # Whether to permute the training data

    # Add max_mask_pct parameter
    model_src: str | None = None
    tc_relu: bool = False

    def __post_init__(self):
        if self.accumulate_grad_batches < 1:
            raise ValueError("accumulate_grad_batches must be >= 1")
            
        # Generate random seed if none provided
        if self.seed is None:
            self.seed = np.random.randint(0, 2**32 - 1)

        ## Make all warmup steps <= max_steps
        self.warmup_steps_tau = min(self.warmup_steps_tau, self.max_steps)
        self.warmup_steps_beta = min(self.warmup_steps_beta, self.max_steps)
        self.warmup_steps_mask_pct = min(self.warmup_steps_mask_pct, self.max_steps)
        self.warmup_steps_lr = min(self.warmup_steps_lr, self.max_steps)

        if self.decay_steps_lr is None:
            self.decay_steps_lr = self.max_steps - self.warmup_steps_lr
  

    def to_dict(self) -> dict:
        """Convert config to a dictionary."""
        config_dict = {
            field: getattr(self, field) 
            for field in self.__dataclass_fields__
        }
        # Handle nested GridDVAEConfig
        config_dict['model_config'] = self.model_config.to_dict()
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ExperimentConfig':
        """Create config from a dictionary."""
        # Handle nested GridDVAEConfig
        if isinstance(config_dict.get('model_config'), dict):
            config_dict = config_dict.copy()
            config_dict['model_config'] = GridDVAEConfig.from_dict(config_dict['model_config'])
        return cls(**config_dict)

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
        add_safe_globals([ExperimentConfig, GridDVAEConfig])
        
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

class DVAETrainingModule(pl.LightningModule):
    def __init__(self, experiment_config: ExperimentConfig, compile_model: bool = False):
        super(DVAETrainingModule, self).__init__()
        self.experiment_config = experiment_config
        self.padding_idx = experiment_config.model_config.padding_idx
        self.model_config = experiment_config.model_config
        self.model = None
        self.learning_rate = experiment_config.learning_rate
        self.compile_model = compile_model
        self.save_hyperparameters()
        # Initialize q_z_marg with correct size but all zeros
        self.register_buffer('q_z_marg', torch.zeros(self.model_config.n_codes, self.model_config.codebook_size), persistent=True)
    
    def configure_model(self):
        """
        Compile the model after device placement.
        This gets called in on_fit_start so that the model is already on GPU.
        """
        if self.model is not None:
            return

        self.model = GridDVAE(self.model_config)
        
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

        # Temperature schedule with its own warmup.
        tau_schedule = Schedule.get_schedule(
            initial_value=cfg.tau_start,
            target_value=cfg.tau,
            warmup_steps=cfg.warmup_steps_tau,
            schedule_type=cfg.tau_schedule_type
        )
        
        # Add beta_ce schedule with shared beta warmup.
        beta_ce_schedule = Schedule.get_schedule(
            initial_value=cfg.beta_ce_start,
            target_value=cfg.beta_ce,
            warmup_steps=cfg.warmup_steps_beta,
            schedule_type=cfg.beta_schedule_type
        )
        
        # Beta schedules with shared beta warmup.
        beta_mi_schedule = Schedule.get_schedule(
            initial_value=cfg.beta_mi_start,
            target_value=cfg.beta_mi,
            warmup_steps=cfg.warmup_steps_beta,
            schedule_type=cfg.beta_schedule_type
        )
        
        beta_tc_schedule = Schedule.get_schedule(
            initial_value=cfg.beta_tc_start,
            target_value=cfg.beta_tc,
            warmup_steps=cfg.warmup_steps_beta,
            schedule_type=cfg.beta_schedule_type
        )
        
        beta_dwkl_schedule = Schedule.get_schedule(
            initial_value=cfg.beta_dwkl_start,
            target_value=cfg.beta_dwkl,
            warmup_steps=cfg.warmup_steps_beta,
            schedule_type=cfg.beta_schedule_type
        )
        
        beta_kl_schedule = Schedule.get_schedule(
            initial_value=cfg.beta_kl_start,
            target_value=cfg.beta_kl,
            warmup_steps=cfg.warmup_steps_beta,
            schedule_type=cfg.beta_schedule_type
        )
        
        # Add max mask percentage schedule (using beta warmup).
        max_mask_pct_schedule = Schedule.get_schedule(
            initial_value=cfg.mask_pct_start,
            target_value=cfg.max_mask_pct,
            warmup_steps=cfg.warmup_steps_mask_pct,
            schedule_type=cfg.mask_schedule_type
        )
        
        return {
            'tau': torch.tensor(tau_schedule(step), device=device, dtype=torch.float32),
            'beta(CE)': torch.tensor(beta_ce_schedule(step), device=device, dtype=torch.float32),
            'beta(MI)': torch.tensor(beta_mi_schedule(step), device=device, dtype=torch.float32),
            'beta(TC)': torch.tensor(beta_tc_schedule(step), device=device, dtype=torch.float32),
            'beta(DWKL)': torch.tensor(beta_dwkl_schedule(step), device=device, dtype=torch.float32),
            'beta(KL)': torch.tensor(beta_kl_schedule(step), device=device, dtype=torch.float32),
            'max_pct(MASK)': torch.tensor(max_mask_pct_schedule(step), device=device, dtype=torch.float32)
        }

    def forward(self, x: Tensor, q_z_marg: Optional[Tensor] = None, tau: Tensor = torch.tensor(1.0), beta_ce: Tensor = torch.tensor(1.0), beta_mi: Tensor = torch.tensor(0.0), beta_dwkl: Tensor = torch.tensor(0.0), beta_tc: Tensor = torch.tensor(0.0), beta_kl: Tensor = torch.tensor(1.0), mask_pct: Tensor = torch.tensor(0.0), tc_relu: bool = False):
        # All scheduled parameters are required as tensor arguments.
        
        # Forward pass with provided scheduled parameters.
        logits, log_alpha, losses, updated_q_z_marg = self.model.forward(
            x, 
            q_z_marg=q_z_marg,
            mask_percentage=mask_pct, 
            tau=tau
        )
        
        # Calculate token accuracy excluding padding tokens.
        non_padding_mask = (x != self.padding_idx)
        masked_correct_tokens = logits.argmax(dim=-1) == x
        masked_correct_tokens = masked_correct_tokens * non_padding_mask
        token_accuracy = masked_correct_tokens.sum() / non_padding_mask.sum()
        
        # Calculate sample accuracy (all tokens must be predicted correctly).
        sample_correct = (logits.argmax(dim=-1) == x).all(dim=1).float()
        sample_accuracy = sample_correct.mean()
        
        # Helper function to conditionally apply ReLU.
        def maybe_relu(x):
            return F.relu(x) if tc_relu else x
        
        # Compute total loss in a way that maintains the computational graph.
        # Scale the KLD losses by the number of latents (similar to Dalle-E paper).
        raw_losses = {
            'loss(CE)': losses['ce_loss'],
            'loss(MI)': losses['mi_loss'],
            'loss(DWKL)': losses['dwkl_loss'],
            'loss(TC)': maybe_relu(losses['tc_loss']),
            'loss(KL)': losses['kl_loss']
        }
        
        # Compute weighted losses for total loss.
        weighted_losses = {
            'loss(CE)': raw_losses['loss(CE)'] * beta_ce,
            'loss(MI)': raw_losses['loss(MI)'] * beta_mi,
            'loss(DWKL)': raw_losses['loss(DWKL)'] * beta_dwkl,
            'loss(TC)': raw_losses['loss(TC)'] * beta_tc,
            'loss(KL)': raw_losses['loss(KL)'] * beta_kl
        }
        
        total_loss = sum(weighted_losses.values())
        
        # Return tensor values; logging callbacks can detach/convert them as needed.
        return {
            'loss': total_loss,
            **raw_losses,
            'accuracy(TOKENS)': token_accuracy,
            'accuracy(SAMPLES)': sample_accuracy,
            'log_alpha': log_alpha,
            'logits': logits,  # Add logits to the output dictionary.
            'input': x,        # Add input to the output dictionary.
        }, updated_q_z_marg

    def training_step(self, batch, batch_idx):
        torch.compiler.cudagraph_mark_step_begin()
        x, _ = batch
        # Clone q_z_marg before passing it to forward to avoid CUDA graph issues
        q_z_marg_clone = self.q_z_marg.clone() if self.q_z_marg is not None else None
            
        # Check if q_z_marg should be treated as None
        effective_q_z_marg = None if q_z_marg_clone.sum() == 0 else q_z_marg_clone

        # Extract scheduled values and convert them to tensors on the current device.
        scheduled = self.get_scheduled_values(self.global_step, device=x.device)

        # Sample mask percentage for this batch; set max_pct_mask to 0 in validation for no masking.
        mask_pct = torch.empty(1, device=x.device).uniform_(0.0, scheduled['max_pct(MASK)'])[0]

        output_dict, updated_q_z_marg = self(
             x,
             q_z_marg=effective_q_z_marg,
             tc_relu=self.experiment_config.tc_relu,
             tau=scheduled['tau'],
             beta_ce=scheduled['beta(CE)'],
             beta_mi=scheduled['beta(MI)'],
             beta_dwkl=scheduled['beta(DWKL)'],
             beta_tc=scheduled['beta(TC)'],
             beta_kl=scheduled['beta(KL)'],
             mask_pct=mask_pct
        )

        output_dict['percent(MASK)'] = mask_pct
        output_dict.update(scheduled)
        
        # Update the global q_z_marg estimate using copy_ instead of assignment
        if updated_q_z_marg is not None:
            self.q_z_marg.copy_(updated_q_z_marg.detach())
        return output_dict

    def validation_step(self, batch, batch_idx):
        torch.compiler.cudagraph_mark_step_begin()
        x, _ = batch
        q_z_marg_clone = self.q_z_marg.clone() if self.q_z_marg is not None else None

        # Check if q_z_marg should be treated as None
        effective_q_z_marg = None if q_z_marg_clone.sum() == 0 else q_z_marg_clone

        # For validation, scheduled values should be passed as provided (e.g., max_pct_mask can be set to 0 for no masking).
        scheduled = self.get_scheduled_values(self.global_step, device=x.device)
        
        mask_pct = torch.tensor(0.0, device=x.device) # No masking in validation

        output_dict, _ = self(
             x,
             q_z_marg=effective_q_z_marg,
             tc_relu=self.experiment_config.tc_relu,
             tau=scheduled['tau'],
             beta_ce=scheduled['beta(CE)'],
             beta_mi=scheduled['beta(MI)'],
             beta_dwkl=scheduled['beta(DWKL)'],
             beta_tc=scheduled['beta(TC)'],
             beta_kl=scheduled['beta(KL)'],
             mask_pct=mask_pct
        )

        output_dict.update(scheduled)
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

    train_dataset = GridDataset(dataset_type=train_ds, max_samples=limit_training_samples)

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

    print("Max two std info bits:", max_two_std_info_bits)
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
        'mean': float(overall_mean) ,
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
            checkpoint_dir=checkpoint_dir
        )
        
        # Load just the model weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
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
    lr_find: bool = False,
    acceleration: AccelerationConfig | None = None,
    save_visualizations: bool = False,
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
        save_to_disk=save_visualizations,
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
        model = DVAETrainingModule(
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


if __name__ == '__main__':
    print("Please use the CLI interface to train models:")
    print("python cli.py dvae train --help")
    sys.exit(1) 