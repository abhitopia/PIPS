import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch.optim import Adam
from pips.grid_dataset import GridDataset
from pips.dvae import GridDVAEConfig, GridDVAE
import torch
from functools import partial
from typing import Callable, Dict
import numpy as np
from dataclasses import dataclass
import warnings
import time
import wandb  # Ensure wandb is imported

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

import pytorch_lightning as pl
from pips.misc.custom_progress_bar import CustomRichProgressBar

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
            return num_tokens / elapsed_time if elapsed_time > 0 else 0
        return 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        # Calculate tokens per second for the training batch
        if self.train_batch_start_time is not None:
            tokens_per_sec = self._calculate_tokens_per_sec(self.train_batch_start_time, batch)
            print(f"[Train] {self.get_loss_string(outputs)} | T/s: {tokens_per_sec:.2f}")
            outputs['tokens_per_sec'] = tokens_per_sec
        
        # Log loss metrics using the helper method
        self._log_metrics(pl_module, 'train', outputs, batch[0].size(0), on_step=True, on_epoch=False)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=None):
        # Record the start time of the validation batch
        self.val_batch_start_time = time.monotonic()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        # Calculate tokens per second for the validation batch
        if self.val_batch_start_time is not None:
            tokens_per_sec = self._calculate_tokens_per_sec(self.val_batch_start_time, batch)
            print(f"[Eval]  {self.get_loss_string(outputs)} | T/s: {tokens_per_sec:.2f}")
            outputs['tokens_per_sec'] = tokens_per_sec
        
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
            elif key in ['hard', 'tau', 'beta', 'mask_pct']:
                metric_name = f'params/{key}_{phase}'  # Group parameters under 'params/'
            # Handle any remaining metrics
            elif key.strip():
                metric_name = f'{key.capitalize()}/{key}_{phase}'
            
            pl_module.log(metric_name, float(value), on_step=on_step, on_epoch=on_epoch, batch_size=batch_size, logger=True)

@dataclass
class ExperimentConfig:
    """Configuration for training hyperparameters and schedules"""

    # Sampling parameters
    hard_from: int | None = 0  # None: after warmup, 0: always hard, >0: after specific step
    reinMax: bool = True

    # Initial values
    initial_tau: float = 0.9
    min_tau: float = 0.1
    initial_beta_mi: float = 0.0
    initial_beta_tc: float = 0.0
    initial_beta_dwkl: float = 0.0
    initial_beta_kl: float = 0.0  # Add initial beta for KL
    
    # Target values
    target_beta_mi: float = 1.0
    target_beta_tc: float = 1.0
    target_beta_dwkl: float = 1.0
    target_beta_kl: float = 1.0   # Add target beta for KL
    
    # Schedule parameters
    warmup_steps: int = 10000
    tau_schedule_type: str = 'cosine_decay'
    beta_schedule_type: str = 'cosine_anneal'
    
    # Training parameters
    batch_size: int = 4
    learning_rate: float = 1e-3
    max_steps: int = 1000000

    # Add max_mask_pct parameter
    max_mask_pct: float = 0.5  # Maximum masking percentage to reach during training
    mask_schedule_type: str = 'linear'

    def __post_init__(self):
        if self.hard_from is None:
            self.hard_from = self.warmup_steps
        elif self.hard_from < 0:
            raise ValueError("hard_from must be None, 0, or a positive integer")

class Schedule:
    """Generic scheduler for parameter annealing or decay"""
    
    @staticmethod
    def get_schedule(
        initial_value: float, 
        target_value: float, 
        warmup_steps: int,
        schedule_type: str = 'linear'
    ) -> Callable[[int], float]:
        """
        Returns a schedule function that takes global_step as input and returns the current value.
        
        Args:
            initial_value: Starting value
            target_value: Final value
            warmup_steps: Number of steps to reach target value
            schedule_type: Type of schedule ('linear', 'exponential', 'cosine_decay', or 'cosine_anneal')
        """
        if schedule_type == 'linear':
            def schedule(step: int) -> float:
                if step >= warmup_steps:
                    return target_value
                return initial_value + (target_value - initial_value) * (step / warmup_steps)
                
        elif schedule_type == 'exponential':
            decay_rate = -np.log(target_value / initial_value) / warmup_steps
            def schedule(step: int) -> float:
                if step >= warmup_steps:
                    return target_value
                return initial_value * np.exp(-decay_rate * step)
                
        elif schedule_type == 'cosine_decay':
            def schedule(step: int) -> float:
                if step >= warmup_steps:
                    return target_value
                progress = step / warmup_steps
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                return target_value + (initial_value - target_value) * cosine_decay
                
        elif schedule_type == 'cosine_anneal':
            def schedule(step: int) -> float:
                if step >= warmup_steps:
                    return target_value
                progress = step / warmup_steps
                cosine_anneal = 0.5 * (1 - np.cos(np.pi * progress))
                return initial_value + (target_value - initial_value) * cosine_anneal
        elif schedule_type == 'threshold':
            def schedule(step: int) -> float:
                return target_value if step >= warmup_steps else initial_value
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
            
        return schedule


class DVAETrainingModule(pl.LightningModule):
    def __init__(self, model_config: GridDVAEConfig, experiment_config: ExperimentConfig):
        super(DVAETrainingModule, self).__init__()
        self.model_config = model_config
        self.experiment_config = experiment_config
        self.model = self.build_model()
        self.save_hyperparameters()

    def build_model(self):
        return GridDVAE(self.model_config)

    def get_scheduled_values(self, step: int) -> Dict[str, float]:
        """Returns all scheduled values for the current step."""
        cfg = self.experiment_config
        
        # Hard sampling threshold schedule (0.0 = soft, 1.0 = hard)
        hard_schedule = Schedule.get_schedule(
            initial_value=False,
            target_value=True,
            warmup_steps=cfg.hard_from,
            schedule_type='threshold'  # Immediate step from 0 to 1 at warmup_steps
        )
        
        # Temperature schedule
        tau_schedule = Schedule.get_schedule(
            initial_value=cfg.initial_tau,
            target_value=cfg.min_tau,
            warmup_steps=cfg.warmup_steps,
            schedule_type=cfg.tau_schedule_type
        )
        
        # Beta schedules
        beta_mi_schedule = Schedule.get_schedule(
            initial_value=cfg.initial_beta_mi,
            target_value=cfg.target_beta_mi,
            warmup_steps=cfg.warmup_steps,
            schedule_type=cfg.beta_schedule_type
        )
        
        beta_tc_schedule = Schedule.get_schedule(
            initial_value=cfg.initial_beta_tc,
            target_value=cfg.target_beta_tc,
            warmup_steps=cfg.warmup_steps,
            schedule_type=cfg.beta_schedule_type
        )
        
        beta_dwkl_schedule = Schedule.get_schedule(
            initial_value=cfg.initial_beta_dwkl,
            target_value=cfg.target_beta_dwkl,
            warmup_steps=cfg.warmup_steps,
            schedule_type=cfg.beta_schedule_type
        )
        
        # Add KL beta schedule
        beta_kl_schedule = Schedule.get_schedule(
            initial_value=cfg.initial_beta_kl,
            target_value=cfg.target_beta_kl,
            warmup_steps=cfg.warmup_steps,
            schedule_type=cfg.beta_schedule_type
        )
        
        # Add max mask percentage schedule
        max_mask_pct_schedule = Schedule.get_schedule(
            initial_value=0.0,
            target_value=cfg.max_mask_pct,
            warmup_steps=cfg.warmup_steps,
            schedule_type='cosine_anneal'
        )
        
        return {
            'hard': hard_schedule(step),  # Convert to boolean
            'tau': tau_schedule(step),
            'beta(MI)': beta_mi_schedule(step),
            'beta(TC)': beta_tc_schedule(step),
            'beta(DWKL)': beta_dwkl_schedule(step),
            'beta(KL)': beta_kl_schedule(step),  # Add KL beta
            'max_mask_pct': max_mask_pct_schedule(step),
        }

    def forward(self, x, train=True):
        reinMax = self.experiment_config.reinMax
        
        # Get current values for all scheduled parameters
        scheduled_values = self.get_scheduled_values(self.global_step)
        hard = scheduled_values['hard']

        # Sample mask percentage for this batch
        mask_pct = 0.0  # No masking during validation
        if train:
            max_mask_pct = scheduled_values['max_mask_pct']
            mask_pct = np.random.uniform(0.0, max_mask_pct)
        
        # Forward pass with current scheduled values
        _, reconstruction_loss, kld_losses = self.model.forward(
            x, 
            mask_percentage=mask_pct, 
            hard=hard, 
            reinMax=reinMax,
            tau=scheduled_values['tau']
        )

        # The problem is that reconstruction loss (output tokens) is computed in a different space than the KLD losses (latent codes)
        # Per sample, CE is summed over number of output tokens (1024)
        # Per sample, KLD is computed over number of latent codes (say 16)
        # In practice, we typically compute the ce_loss per token, while keeping the KLD losss using batchmean 
        # (Ref: https://github.com/lucidrains/DALLE-pytorch/blob/58c1e1a4fef10725a79bd45cdb5581c03e3e59e7/dalle_pytorch/dalle_pytorch.py#L261)
        reconstruction_loss = reconstruction_loss / x.size(1) # Normalize by number of tokens

        # Compute total loss in a way that maintains the computational graph
        loss_components = {
            'loss(CE)': reconstruction_loss,
            'loss(MI)': kld_losses['mi_loss'] * scheduled_values['beta(MI)'],
            'loss(DWKL)': kld_losses['dwkl_loss'] * scheduled_values['beta(DWKL)'],
            'loss(TC)': kld_losses['tc_loss'] * scheduled_values['beta(TC)'],
            'loss(KL)': kld_losses['kl_loss'] * scheduled_values['beta(KL)']  # Add KL loss component
        }
        
        total_loss = sum(loss_components.values())

        return {
            'loss': total_loss,
            **{k: v.detach() for k, v in loss_components.items()},
            **{k: v for k, v in scheduled_values.items()},  # Include scheduled values
            'mask_pct': mask_pct
        }

    def training_step(self, batch, batch_idx):
        x, _ = batch
        output_dict = self(x, train=True)
        
        return output_dict
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        output_dict = self(x, train=False)
    
        return output_dict

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.experiment_config.learning_rate)


def main():
    # Model configuration
    model_config = GridDVAEConfig(
        n_dim=256,
        n_head=8,
        n_layers=6,
        n_codes=16,
        codebook_size=512,
        rope_base=10000,
        dropout=0.0,
        n_pos=32 * 32,
        n_vocab=16
    )

    # Experiment configuration
    experiment_config = ExperimentConfig(
        initial_tau=0.9,
        min_tau=0.1,
        initial_beta_mi=0.0,
        initial_beta_tc=0.0,
        initial_beta_dwkl=0.0,
        initial_beta_kl=0.0,    # Add initial beta for KL
        target_beta_mi=1.0,
        target_beta_tc=1.0,
        target_beta_dwkl=1.0,
        target_beta_kl=1.0,     # Add target beta for KL
        warmup_steps=5000,
        batch_size=4,
        learning_rate=1e-3,
        max_steps=100000,
        max_mask_pct=0.5  # Set maximum masking percentage
    )

    # Create datasets and dataloaders
    pad_value = 10
    project_size = (32, 32)
    
    collate_fn_train = partial(GridDataset.collate_fn, pad_value=pad_value, permute=True, project_size=project_size)
    train_dataset = GridDataset(train=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=experiment_config.batch_size, 
        collate_fn=collate_fn_train,
        shuffle=True, 
        num_workers=0  # It must be 0 because loading the dataset is otherwise too slow
    )

    collate_fn_val = partial(GridDataset.collate_fn, pad_value=pad_value, permute=False, project_size=project_size)
    val_dataset = GridDataset(train=False)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=experiment_config.batch_size,
        collate_fn=collate_fn_val,
        shuffle=False, 
        num_workers=4,
        persistent_workers=True
    )

    # Initialize the model and trainer
    model = DVAETrainingModule(model_config, experiment_config)
    wandb_logger = WandbLogger(
        project='dvae-training',
        log_model='all',
        save_dir='wandb_logs',
    )

    # Add the custom logging callback
    logging_callback = LoggingCallback()
    custom_progress_bar = CustomRichProgressBar()

    trainer = pl.Trainer(
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger,
        gradient_clip_val=1.0,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val/loss'),  # Updated to match sanitized key
            logging_callback, 
            custom_progress_bar
        ],
        max_epochs=-1,
        max_steps=experiment_config.max_steps,
        limit_train_batches=1000,
        limit_val_batches=10,
        val_check_interval=5,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main() 