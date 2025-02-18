from functools import partial
import os
from typing import Dict, Any
import numpy as np
from dataclasses import dataclass
import tempfile

import warnings
import torch
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner.tuning import Tuner
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from pips.grid_dataset import GridDataset, worker_init_fn
from pips.dvae import GridDVAEConfig, GridDVAE
from pips.misc.artifact import Artifact
from pips.utils import generate_friendly_name
from pips.misc.custom_progress_bar import CustomRichProgressBar
from pips.misc.schedule import Schedule  # Add this import
from pips.misc.checkpoint_with_wandb_sync import ModelCheckpointWithWandbSync
from pathlib import Path
from torch.serialization import add_safe_globals  # Add this import at the top
import wandb
import sys
import matplotlib.pyplot as plt
from pips.misc.acceleration_config import AccelerationConfig


class LoggingCallback(pl.Callback):
    
    def __init__(self):
        self.train_batch_start_time = None
        self.val_batch_start_time = None

    def get_loss_string(self, outputs: Dict[str, torch.Tensor]) -> str:
        return ' | '.join([f"{l}: {v:.2e}" for l, v in outputs.items() if 'loss' in l])
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=None):
        # Record the start time of the training batch
        self.train_batch_start_time = time.monotonic()

    def _calculate_tokens_per_sec(self, start_time, batch):
        if start_time is not None:
            elapsed_time = time.monotonic() - start_time
            x, _ = batch
            num_tokens = x.size(0) * x.size(1)  # batch_size * tokens_per_batch
            tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0
            time_per_batch_ms = elapsed_time * 1000  # Convert to milliseconds
            return tokens_per_sec, time_per_batch_ms
        return 0, 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for GPU operations to finish

        # Calculate tokens per second for the training batch
        if self.train_batch_start_time is not None:
            tokens_per_sec, time_per_batch_ms = self._calculate_tokens_per_sec(self.train_batch_start_time, batch)
            print(f"\n[Train] {self.get_loss_string(outputs)} | T/s: {tokens_per_sec:.2f} | Δ(ms): {time_per_batch_ms:.1f}ms")
            outputs['tokens_per_sec'] = tokens_per_sec
            outputs['Δ_ms'] = time_per_batch_ms
        
        # Log the current learning rate
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        pl_module.log('params/learning_rate', current_lr, on_step=True, on_epoch=False)
        
        # Log loss metrics using the helper method
        self._log_metrics(pl_module, 'train', outputs, batch[0].size(0), on_step=True, on_epoch=False)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=None):
        # Record the start time of the validation batch
        self.val_batch_start_time = time.monotonic()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for GPU operations to finish
        # Calculate tokens per second for the validation batch
        if self.val_batch_start_time is not None:
            tokens_per_sec, time_per_batch_ms = self._calculate_tokens_per_sec(self.val_batch_start_time, batch)
            print(f"\n[Eval]  {self.get_loss_string(outputs)} | T/s: {tokens_per_sec:.2f} | Δ(ms): {time_per_batch_ms:.1f}ms")
            outputs['tokens_per_sec'] = tokens_per_sec
            outputs['Δ_ms'] = time_per_batch_ms
        
        # Log loss metrics using the helper method
        self._log_metrics(pl_module, 'val', outputs, batch[0].size(0), on_step=False, on_epoch=True)

    def _log_metrics(self, pl_module: pl.LightningModule, phase: str, outputs: Dict[str, torch.Tensor], batch_size: int, on_step: bool, on_epoch: bool):
        """Helper method to log loss metrics."""
        for key, value in outputs.items():
            # Handle the main loss separately
            if key == 'loss':
                metric_name = f'{phase}/{key}'  # Default format
                pl_module.log(metric_name, value, on_step=on_step, on_epoch=on_epoch, batch_size=batch_size, logger=False)
                continue

            # Handle metrics with categories - loss(CE), loss(MI), etc.
            if '(' in key and ')' in key:
                category = key.split('(')[-1].split(')')[0]  # Extract category
                metric_name = f'{category}/{key.split("(")[0]}_{phase}'  # Format as "category/metric_phase"
            # Handle special parameters like 'hard', 'tau', 'beta', etc.
            elif key in ['hard', 'tau', 'beta', 'mask_pct', 'max_mask_pct']:
                metric_name = f'params/{key}_{phase}'  # Group parameters under 'params/'
            elif key in ['tokens_per_sec', 'Δ_ms']:
                metric_name = f'Throughput/{key}_{phase}'
            elif key in ['avg_entropy', 'avg_perplexity']:
                metric_name = f'Codebook/{key}_{phase}'
            # Handle any remaining metrics
            elif key.strip():
                metric_name = f'{key.capitalize()}/{key}_{phase}'
            
            sync_dist = False if phase == 'train' else True
            pl_module.log(metric_name, float(value), on_step=on_step, on_epoch=on_epoch, batch_size=batch_size, logger=True, sync_dist=sync_dist)

@dataclass
class ExperimentConfig:
    """Configuration for training hyperparameters and schedules"""
    # Model configuration
    model_config: GridDVAEConfig
    
    # Add seed parameter
    seed: int | None = None  # None means random seed
    
    # Sampling parameters
    hard_from: int | None = None  # None: after warmup, 0: always hard, >0: after specific step
    reinMax: bool = True

    # Initial values (renamed from initial_*)
    tau_start: float = 1.0
    tau: float = 0.0625 # 1/16 as per Dalle-E paper
    beta_mi_start: float = 0.0
    beta_tc_start: float = 0.0
    beta_dwkl_start: float = 0.0
    beta_kl_start: float = 0.0
    
    # Final values (renamed from target_*)
    beta_mi: float = 0.0
    beta_tc: float = 6.0
    beta_dwkl: float = 0.0
    beta_kl: float = 2.0
    
    # Schedule types
    tau_schedule_type: str = 'cosine_decay'
    beta_schedule_type: str = 'cosine_anneal'
    
    # Replace single warmup_steps with separate warmups
    warmup_steps_lr: int = 10_000
    warmup_steps_tau: int = 150_000
    warmup_steps_beta: int = 10_000
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-4 # Consistent with Dalle-E paper
    weight_decay: float = 1e-4 # Consistent with Dalle-E paper
    max_steps: int = 1_000_000
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1

    # Add max_mask_pct parameter
    max_mask_pct: float = 0.5
    mask_schedule_type: str = 'cosine_anneal'

    padding_idx: int | None = None

    model_src: str | None = None

    def __post_init__(self):
        if self.hard_from is None:
            self.hard_from = self.warmup_steps_lr # Use LR warmup for hard schedule
        elif self.hard_from < 0:
            raise ValueError("hard_from must be None, 0, or a positive integer")
        
        if self.accumulate_grad_batches < 1:
            raise ValueError("accumulate_grad_batches must be >= 1")
            
        # Automatically set padding_idx if not provided
        if self.padding_idx is None:
            self.padding_idx = self.model_config.n_vocab - 1
            
        # Generate random seed if none provided
        if self.seed is None:
            self.seed = np.random.randint(0, 2**32 - 1)

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

    def get_scheduled_values(self, step: int) -> Dict[str, float]:
        """Returns all scheduled values for the current step."""
        cfg = self.experiment_config
        
        # Hard sampling threshold schedule (0.0 = soft, 1.0 = hard)
        hard_schedule = Schedule.get_schedule(
            initial_value=False,
            target_value=True,
            warmup_steps=cfg.hard_from,
            schedule_type='threshold'
        )
        
        # Temperature schedule with its own warmup
        tau_schedule = Schedule.get_schedule(
            initial_value=cfg.tau_start,
            target_value=cfg.tau,
            warmup_steps=cfg.warmup_steps_tau,
            schedule_type=cfg.tau_schedule_type
        )
        
        # Beta schedules with shared beta warmup
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
        
        # Add max mask percentage schedule (using beta warmup)
        max_mask_pct_schedule = Schedule.get_schedule(
            initial_value=0.0,
            target_value=cfg.max_mask_pct,
            warmup_steps=cfg.warmup_steps_beta,
            schedule_type=cfg.mask_schedule_type
        )
        
        return {
            'hard': hard_schedule(step),
            'tau': tau_schedule(step),
            'beta(MI)': beta_mi_schedule(step),
            'beta(TC)': beta_tc_schedule(step),
            'beta(DWKL)': beta_dwkl_schedule(step),
            'beta(KL)': beta_kl_schedule(step),
            'max_pct(MASK)': max_mask_pct_schedule(step),
        }

    def forward(self, x, q_z_marg=None, train=True):

        # Get current values for all scheduled parameters
        scheduled_values = self.get_scheduled_values(self.global_step)
        hard = scheduled_values['hard']
        reinMax = self.experiment_config.reinMax and hard

        # Sample mask percentage for this batch
        mask_pct = 0.0  # No masking during validation
        if train:
            max_mask_pct = scheduled_values['max_pct(MASK)']
            mask_pct = torch.empty(1, device=x.device).uniform_(0.0, max_mask_pct)[0]
        
        # Forward pass with current scheduled values and provided q_z_marg
        logits, losses, updated_q_z_marg = self.model.forward(
            x, 
            q_z_marg=q_z_marg,
            mask_percentage=mask_pct, 
            hard=hard, 
            reinMax=reinMax,
            tau=scheduled_values['tau']
        )

        # Calculate token accuracy
        predictions = logits.argmax(dim=-1)
        correct_tokens = (predictions == x).float()
        token_accuracy = correct_tokens.mean()

        # Calculate token accuracy excluding padding tokens
        padding_idx = self.experiment_config.padding_idx
        non_padding_mask = (x != padding_idx)
        acc_no_pad = (correct_tokens * non_padding_mask).sum() / non_padding_mask.sum()

        # Calculate sample accuracy
        sample_correct = correct_tokens.all(dim=1).float()
        sample_accuracy = sample_correct.mean()

        # The problem is that reconstruction loss (output tokens) is computed in a different space than the KLD losses (latent codes)
        # Per sample, CE is summed over number of output tokens (1024)
        # Per sample, KLD is computed over number of latent codes (say 16)
        # In practice, we typically compute the ce_loss per token, while keeping the KLD losss using batchmean 
        # (Ref: https://github.com/lucidrains/DALLE-pytorch/blob/58c1e1a4fef10725a79bd45cdb5581c03e3e59e7/dalle_pytorch/dalle_pytorch.py#L261)
        ce_loss = losses['ce_loss'] / x.size(1) # Normalize by number of tokens

        # Compute total loss in a way that maintains the computational graph.
        # Scale the KLD losses by the number of latents (similar to Dalle-E paper)
        raw_losses = {
            'loss(CE)': ce_loss,
            'loss(MI)': losses['mi_loss']/self.model_config.n_codes,
            'loss(DWKL)': losses['dwkl_loss']/self.model_config.n_codes,
            'loss(TC)': losses['tc_loss']/self.model_config.n_codes,
            'loss(KL)': losses['kl_loss']/self.model_config.n_codes
        }
        
        # Compute weighted losses for total loss
        weighted_losses = {
            'loss(CE)': raw_losses['loss(CE)'],
            'loss(MI)': raw_losses['loss(MI)'] * scheduled_values['beta(MI)'],
            'loss(DWKL)': raw_losses['loss(DWKL)'] * scheduled_values['beta(DWKL)'],
            'loss(TC)': raw_losses['loss(TC)'] * scheduled_values['beta(TC)'],
            'loss(KL)': raw_losses['loss(KL)'] * scheduled_values['beta(KL)']
        }
        
        total_loss = sum(weighted_losses.values())

        return {
            'loss': total_loss,
            **{k: v.detach() for k, v in raw_losses.items()},  # Log raw losses
            **{k: v for k, v in scheduled_values.items()},
            'percent(MASK)': mask_pct,
            'accuracy(TOKENS)': token_accuracy.detach(),
            'accuracy_no_pad(TOKENS)': acc_no_pad.detach(),
            'accuracy(SAMPLES)': sample_accuracy.detach(),
            'avg_entropy': losses['avg_entropy'].detach(),
            'avg_perplexity': losses['avg_perplexity'].detach()
        }, updated_q_z_marg

    def training_step(self, batch, batch_idx):
        torch.compiler.cudagraph_mark_step_begin()
        x, _ = batch
        # Clone q_z_marg before passing it to forward to avoid CUDA graph issues
        q_z_marg_clone = self.q_z_marg.clone() if self.q_z_marg is not None else None
            
        # Check if q_z_marg should be treated as None
        effective_q_z_marg = None if q_z_marg_clone.sum() == 0 else q_z_marg_clone

        output_dict, updated_q_z_marg = self(x, q_z_marg=effective_q_z_marg, train=True)

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

        # No update of q_z_marg in validation
        output_dict, _ = self(x, q_z_marg=effective_q_z_marg, train=False)
    
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
            min_lr_factor = 0.01 # 1/100 of the max LR (Consistent with Dalle-E paper)
            
            if step < warmup_steps:
                # Linear warmup
                return float(step) / float(max(1, warmup_steps))
            else:
                # Cosine decay from 1.0 to min_lr_factor
                progress = float(step - warmup_steps) / float(max(1, self.experiment_config.max_steps - warmup_steps))
                return min_lr_factor + 0.5 * (1.0 - min_lr_factor) * (1.0 + np.cos(np.pi * progress))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

def create_dataloaders(experiment_config: ExperimentConfig, debug_mode: bool = False):
    """Create train and validation dataloaders based on experiment configuration.
    
    Args:
        experiment_config: Configuration containing batch_size and padding_idx
        debug_mode: If True, uses reduced workers for debugging
    """
    project_size = (experiment_config.model_config.max_grid_height, 
                   experiment_config.model_config.max_grid_width)
    padding_idx = experiment_config.padding_idx
    batch_size = experiment_config.batch_size

    # Create training dataloader
    collate_fn_train = partial(GridDataset.collate_fn, pad_value=padding_idx, permute=True, project_size=project_size)
    train_dataset = GridDataset(train=True)

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

    # Create validation dataloader
    collate_fn_val = partial(GridDataset.collate_fn, pad_value=padding_idx, permute=False, project_size=project_size)
    val_dataset = GridDataset(train=False)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn_val,
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        drop_last=False
    )

    print("Number of batches in training set: ", len(train_loader))
    print("Number of batches in validation set: ", len(val_loader))

    return train_loader, val_loader


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
    debug_logging: bool = True,
    val_check_interval: int | None = None,
    resume_from: str | None = None,
    lr_find: bool = False,
    acceleration: AccelerationConfig | None = None,
    limit_train_batches: int | None = None,
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
   
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(experiment_config, debug_mode=debug_mode)
    
    # Disable validation if val_check_interval is negative
    validation_disabled = val_check_interval is not None and val_check_interval < 0
    if validation_disabled:
        print("Validation disabled. Checkpoints will not be saved.")
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
        mode="disabled" if debug_mode and not debug_logging else "online",
        config=experiment_config.to_dict()
    )

    # Define callbacks
    logging_callback = LoggingCallback()
    custom_progress_bar = CustomRichProgressBar()

    # Only add checkpoint callbacks if validation is enabled
    callbacks = [logging_callback, custom_progress_bar]
    
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

    trainer = pl.Trainer(
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
        max_steps=experiment_config.max_steps if not debug_mode else 1000,
        limit_train_batches=100 if lr_find else (50 if debug_mode else limit_train_batches),
        limit_val_batches=0 if lr_find or validation_disabled else (10 if debug_mode else None),
        val_check_interval=None if lr_find or validation_disabled else (20 if debug_mode else val_check_interval),
        enable_model_summary=not lr_find
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