import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from pips.grid_dataset import GridDataset
from pips.dvae import GridDVAEConfig, GridDVAE
import torch
from functools import partial
import torch.nn.functional as F
from typing import Callable, Union, Dict
import numpy as np
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Configuration for training hyperparameters and schedules"""

    # Sampling parameters
    hard: bool = True
    reinMax: bool = True

    # Initial values
    initial_tau: float = 0.9
    min_tau: float = 0.1
    initial_beta_mi: float = 0.0
    initial_beta_tc: float = 0.0
    initial_beta_dwkl: float = 0.0
    
    # Target values
    target_beta_mi: float = 1.0
    target_beta_tc: float = 1.0
    target_beta_dwkl: float = 1.0
    
    # Schedule parameters
    warmup_steps: int = 5000
    tau_schedule_type: str = 'cosine_decay'
    beta_schedule_type: str = 'cosine_anneal'
    
    # Training parameters
    batch_size: int = 4
    learning_rate: float = 1e-3
    max_steps: int = 100000


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
        
        return {
            'tau': tau_schedule(step),
            'beta_mi': beta_mi_schedule(step),
            'beta_tc': beta_tc_schedule(step),
            'beta_dwkl': beta_dwkl_schedule(step)
        }

    def forward(self, x, train=True):
        mask_percentage = 0.0 if train else 0.0
        hard = self.experiment_config.hard
        reinMax = self.experiment_config.reinMax
        
        # Get current values for all scheduled parameters
        scheduled_values = self.get_scheduled_values(self.global_step)
        
        # Forward pass with current scheduled values
        _, reconstruction_loss, kld_losses = self.model.forward(
            x, 
            mask_percentage=mask_percentage, 
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
            'ce_loss': reconstruction_loss,
            'mi_loss': kld_losses['mi_loss'] * scheduled_values['beta_mi'],
            'dwkl_loss': kld_losses['dwkl_loss'] * scheduled_values['beta_dwkl'],
            'tc_loss': kld_losses['tc_loss'] * scheduled_values['beta_tc']
        }

     

        
        total_loss = sum(loss_components.values())

        return {
            'loss': total_loss,
            **{k: v.detach() for k, v in loss_components.items()},
            **{k: v for k, v in scheduled_values.items()}  # Include scheduled values
        }

    def training_step(self, batch, batch_idx):
        x, _ = batch
        output_dict = self(x)
        
        # Log all values
        for key, value in output_dict.items():
            if 'loss' in key:
                self.log(f'train/{key}', value, batch_size=x.size(0), prog_bar=True)
            
        return output_dict['loss']

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
        target_beta_mi=1.0,
        target_beta_tc=1.0,
        target_beta_dwkl=1.0,
        warmup_steps=5000,
        batch_size=4,
        learning_rate=1e-3,
        max_steps=100000
    )

    print(model_config)
    print(experiment_config)

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
        num_workers=0
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
    wandb_logger = WandbLogger(project='dvae-training', log_model=True)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger,
        gradient_clip_val=1.0,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor='val/loss')],
        max_steps=experiment_config.max_steps,
        limit_train_batches=10,
        limit_val_batches=10,
        val_check_interval=5,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main() 