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
from pips.ctd_autoencoder import CTDAutoEncoderConfig, CTDAutoEncoder
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

    def visualize_reconstructions(self, pl_module, x, logits, phase, indices=None):
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
        height = pl_module.model_config.grid_height
        width = pl_module.model_config.grid_width
        
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
        
        # Get codebook information if indices are provided
        codebook_size = pl_module.model_config.codebook_size if indices is not None else 0
        
        # Adjust figure height based on whether indices are provided and codebook size
        # For large codebooks, make the discrete code visualization much taller
        if indices is not None:
            # Calculate height with consideration for codebook size
            # Use a logarithmic scale to keep very large codebooks manageable
            discrete_height = min(16, 4 + 3 * np.log10(codebook_size))  # Increased height scaling
            grid_height = 6  # Fixed height for input and reconstruction grids
            fig_height = grid_height + discrete_height
            n_rows = 3
        else:
            fig_height = 6  # Standard height when no indices
            n_rows = 2
        
        # Create the figure with proper space for a colorbar.
        fig = plt.figure(figsize=(n_samples * 3 + 1, fig_height), dpi=80)
        
        # Create a gridspec layout with special height ratios to make discrete codes taller
        if n_rows == 3:
            # Make the discrete code row much taller than the other two rows
            height_ratios = [1, 1, discrete_height/2]
        else:
            height_ratios = [1, 1]
            
        gs = fig.add_gridspec(n_rows, n_samples + 1, 
                             width_ratios=[1] * n_samples + [0.1],
                             height_ratios=height_ratios)
        
        # Create axes for the plots.
        axes = [[fig.add_subplot(gs[i, j]) for j in range(n_samples)] for i in range(n_rows)]
        
        # Plot inputs on the top row.
        for i in range(n_samples):
            im = axes[0][i].imshow(x_np[i], cmap=custom_cmap, vmin=0, vmax=len(colors) - 1)
            axes[0][i].set_title(f"Input {i+1}")
            axes[0][i].axis('off')
            
        # Plot reconstructions on the middle row.
        for i in range(n_samples):
            axes[1][i].imshow(recon_np[i], cmap=custom_cmap, vmin=0, vmax=len(colors) - 1)
            axes[1][i].set_title(f"Reconstruction {i+1}")
            axes[1][i].axis('off')
            
        # Plot discrete codes if provided
        if indices is not None and n_rows > 2:
            indices_subset = indices[:n_samples] if indices.size(0) >= n_samples else indices
            
            # Process indices to create visualization
            for i in range(min(n_samples, indices_subset.size(0))):
                # Get indices for this sample
                sample_indices = indices_subset[i]  # Shape: [n_codes]
                
                # Create boolean matrix where each row corresponds to a position
                # and each column corresponds to a codebook entry
                n_codes = sample_indices.size(0)
                
                # Create a matrix of zeros, then set the used indices to 1
                # Transpose the matrix so codebook entries are on y-axis and positions on x-axis
                bool_matrix = np.zeros((codebook_size, n_codes))
                for pos, idx in enumerate(sample_indices.cpu().numpy()):
                    bool_matrix[idx, pos] = 1
                
                # Plot the boolean matrix using a high contrast colormap for better visibility
                im_discrete = axes[2][i].imshow(bool_matrix, cmap='hot', aspect='auto')
                axes[2][i].set_title(f"Discrete Codes {i+1}")
                
                # Add labels only for the first sample to avoid clutter
                if i == 0:
                    axes[2][i].set_xlabel("Position")
                    axes[2][i].set_ylabel("Codebook Index")
                
                # Add sparse ticks to avoid clutter - now x-axis is positions, y-axis is codebook
                max_ticks = 10
                
                # X-axis ticks (positions)
                if n_codes > max_ticks:
                    tick_step = max(1, n_codes // max_ticks)
                    xticks = np.arange(0, n_codes, tick_step)
                    axes[2][i].set_xticks(xticks)
                    axes[2][i].set_xticklabels([str(x) for x in xticks])
                else:
                    # If small number of positions, show all ticks
                    axes[2][i].set_xticks(np.arange(n_codes))
                    axes[2][i].set_xticklabels([str(x) for x in range(n_codes)])
                
                # Y-axis ticks (codebook indices)
                if codebook_size > max_ticks:
                    # For large codebooks, show more reference points
                    max_codebook_ticks = 20  # Increase number of ticks for large codebooks
                    tick_step = max(1, codebook_size // max_codebook_ticks)
                    yticks = np.arange(0, codebook_size, tick_step)
                    axes[2][i].set_yticks(yticks)
                    axes[2][i].set_yticklabels([str(y) for y in yticks])
                    
                    # Add minor ticks for a more detailed grid
                    minor_tick_step = max(1, tick_step // 2)
                    minor_yticks = np.arange(0, codebook_size, minor_tick_step)
                    axes[2][i].set_yticks(minor_yticks, minor=True)
                
                # Add a grid for better readability
                # Major grid lines aligned with ticks
                axes[2][i].grid(which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                
                # If we have minor ticks, add minor grid lines
                if codebook_size > max_ticks:
                    axes[2][i].grid(which='minor', color='gray', linestyle='-', linewidth=0.25, alpha=0.3)
                
                # Highlight the cells with 1s to make them more visible
                for pos in range(n_codes):
                    idx = sample_indices[pos].item()
                    axes[2][i].add_patch(plt.Rectangle((pos - 0.5, idx - 0.5), 1, 1, 
                                                      fill=False, edgecolor='red', linewidth=1.5))
                
                # Removed the numerical annotations to reduce clutter
            
        # Add a colorbar as a legend using the gridspec.
        cbar_ax = fig.add_subplot(gs[:2, -1])  # Colorbar spans only input and reconstruction rows
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Grid Values')
        
        # Add ticks for the values that appear in the data.
        present_values = sorted(np.unique(np.concatenate([x_np.flatten(), recon_np.flatten()])))
        cbar.set_ticks(present_values)
        cbar.set_ticklabels([str(int(v)) for v in present_values])
        
        # If indices are provided, add a colorbar for the discrete code visualization
        if indices is not None and n_rows > 2:
            cbar_discrete_ax = fig.add_subplot(gs[2:, -1])
            cbar_discrete = fig.colorbar(im_discrete, cax=cbar_discrete_ax)
            cbar_discrete.set_label('Code Usage')
            cbar_discrete.set_ticks([0, 1])
            cbar_discrete.set_ticklabels(['Unused', 'Used'])
        
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

    def calculate_codebook_usage(self, indices: Tensor, codebook_size: int) -> dict:
        """
        Calculate codebook usage statistics from quantization indices.
        
        Args:
            indices (Tensor): Quantization indices of shape [B, n_codes]
            codebook_size (int): Size of the codebook
        Returns:
            dict: Dictionary containing prefixed codebook usage statistics ready for logging
        """

        if indices is None:
            return {}
        
        assert indices.dim() == 2, "Indices must be a 2D tensor"
        flat_indices = indices.reshape(-1)
            
        # Count unique indices
        unique_indices, counts = torch.unique(flat_indices, return_counts=True)
        
        # Calculate percentage of codebook being used
        unique_count = unique_indices.size(0)
        usage_percent = unique_count / codebook_size
        
        # Calculate usage per position
        B, n_codes = indices.size()
        
        # Calculate entropy of the usage distribution (overall)
        normalized_counts = counts.float() / counts.sum()
        entropy = -torch.sum(normalized_counts * torch.log2(normalized_counts + 1e-10))
        max_entropy = torch.log2(torch.tensor(codebook_size, dtype=torch.float32))
        entropy_percent = entropy / max_entropy
        perplexity = 2 ** entropy
        
        # Initialize the results dictionary with prefixed keys
        results = {
            "CodebookUsage/codebook_usage:": usage_percent,
            "CodebookUsage/codebook_entropy": entropy_percent.item(),
            "CodebookUsage/codebook_perplexity": perplexity.item(),
        }
            
        return results
    
    def _create_visualization(self, data, title, xlabel, ylabel, threshold=None, stats_func=None, colors=None, title_suffix=""):
        """
        Helper method to create bar chart visualizations with consistent styling.
        
        Args:
            data (Tensor): Data to visualize as a bar chart
            title (str): Title of the chart
            xlabel (str): Label for x-axis
            ylabel (str): Label for y-axis
            threshold (float, optional): Value to draw as horizontal threshold line
            stats_func (callable, optional): Function to generate stats text
            colors (list, optional): Colors for each bar
            title_suffix (str): Additional text to append to the title
            
        Returns:
            matplotlib.figure.Figure: A figure containing the visualization
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a low-res figure
        fig, ax = plt.subplots(figsize=(8, 4), dpi=80)
        
        # Convert data to numpy array if it's a tensor
        if torch.is_tensor(data):
            # Convert to float32 first if it's BFloat16 or other unsupported dtype
            if data.dtype == torch.bfloat16:
                data = data.to(torch.float32)
            data = data.cpu().numpy()
        
        # Create x-axis positions
        positions = np.arange(len(data))
        
        # Create bar plot
        ax.bar(positions, data, width=1.0, alpha=0.7, color=colors)
        
        # Add threshold line if provided
        if threshold is not None:
            ax.axhline(y=threshold, color='r', linestyle='--', 
                       label=f'Threshold ({threshold})')
        
        # Add title and labels
        full_title = f"{title}{' ' + title_suffix if title_suffix else ''}"
        ax.set_title(full_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Add stats text if provided
        if stats_func is not None:
            stats_text = stats_func(data)
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add legend if we have a threshold or colors with actual labels
        if threshold is not None:
            ax.legend()
        
        # Make layout tight
        fig.tight_layout()
        
        return fig

    def visualize_cluster_sizes(self, vq_embedding, title_suffix=""):
        """
        Create a visualization of the cluster size distribution.
        
        Args:
            vq_embedding: The VQEmbedding module instance
            title_suffix (str): Additional text to append to the visualization title
            
        Returns:
            matplotlib.figure.Figure: A figure containing the visualization
        """
        import numpy as np
        
        def cluster_stats(data):
            """Generate cluster size statistics text"""
            return (
                f"Mean: {np.mean(data):.2f}\n"
                f"Max: {np.max(data):.2f}\n"
                f"Min: {np.min(data):.2f}\n"
                f"Unused: {(data < vq_embedding.unused_reset_threshold).sum()}/{len(data)}"
            )
        
        return self._create_visualization(
            data=vq_embedding.cluster_size,
            title="Codebook Usage Distribution",
            xlabel="Codebook Entry Index",
            ylabel="Cluster Size",
            threshold=vq_embedding.unused_reset_threshold,
            stats_func=cluster_stats,
            title_suffix=title_suffix
        )

    def visualize_update_magnitudes(self, vq_embedding, title_suffix=""):
        """
        Create a visualization of the update magnitude distribution.
        
        Args:
            vq_embedding: The VQEmbedding module instance
            title_suffix (str): Additional text to append to the visualization title
            
        Returns:
            matplotlib.figure.Figure: A figure containing the visualization
        """
        import numpy as np
        from matplotlib.patches import Patch
        
        # Color bars by whether the entry is below the reset threshold
        # Convert to float32 first if it's BFloat16
        cluster_size = vq_embedding.cluster_size
        if cluster_size.dtype == torch.bfloat16:
            cluster_size = cluster_size.to(torch.float32)
        is_below_threshold = cluster_size.cpu().numpy() < vq_embedding.unused_reset_threshold
        colors = ['red' if below else 'blue' for below in is_below_threshold]
        
        def magnitude_stats(data):
            """Generate update magnitude statistics text"""
            return (
                f"Mean: {np.mean(data):.5f}\n"
                f"Max: {np.max(data):.5f}\n"
                f"Min: {np.min(data):.5f}\n"
                f"Top entries updated: {np.argsort(data)[-5:][::-1]}"
            )
        
        fig = self._create_visualization(
            data=vq_embedding.update_magnitudes,
            title="Codebook Update Magnitudes",
            xlabel="Codebook Entry Index",
            ylabel="Update Magnitude (L2 Norm)",
            stats_func=magnitude_stats,
            colors=colors,
            title_suffix=title_suffix
        )
        
        # Add custom legend
        ax = fig.axes[0]
        legend_elements = [
            Patch(facecolor='blue', alpha=0.7, label='Active Entries'),
            Patch(facecolor='red', alpha=0.7, label='Below Reset Threshold')
        ]
        ax.legend(handles=legend_elements)
        
        return fig

    def visualize_codebook_state(self, pl_module, step=None, phase='train'):
        """
        Create visualizations for both cluster sizes and update magnitudes if model uses EMA codebook.
        Also handles logging to WandB if logger is available.
        
        Args:
            pl_module: The Lightning module with the model
            step (int, optional): Current training step to include in visualization titles
            phase (str): 'train' or 'val' to specify which phase we're in
        """
        # Only visualize codebook if using EMA and not skipping codebook
        if not (pl_module.model_config.use_ema and pl_module.model_config.codebook_size > 0):
            return
        
        # Only visualize during training phase, not validation
        if phase != 'train':
            return
        
        # Get the VQEmbedding instance directly
        vq_embedding = pl_module.model.codebook
        if vq_embedding is None:
            return
        
        suffix = f" (Step {step})" if step is not None else ""
        
        # Create visualizations
        cluster_fig = self.visualize_cluster_sizes(vq_embedding, title_suffix=suffix)
        update_fig = self.visualize_update_magnitudes(vq_embedding, title_suffix=suffix)
        
        # Save to disk if requested
        if self.save_to_disk:
            cluster_filename = f"cluster_sizes_{phase}_step_{step:07d}.png"
            update_filename = f"update_magnitudes_{phase}_step_{step:07d}.png"
            cluster_fig.savefig(self.visualization_dir / cluster_filename)
            update_fig.savefig(self.visualization_dir / update_filename)
        
        # Log directly to WandB if available
        if isinstance(pl_module.logger, WandbLogger):
            viz_data = {
                f'VQEmbedding/cluster_sizes_{phase}': wandb.Image(cluster_fig),
                f'VQEmbedding/update_magnitudes_{phase}': wandb.Image(update_fig)
            }
            # Remove the step parameter to let WandB use its internal step counter
            pl_module.logger.experiment.log(viz_data)  # Remove the step parameter
        
        # Close figures to avoid memory leaks
        plt.close(cluster_fig)
        plt.close(update_fig)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        x = outputs.pop('input')
        logits = outputs.pop('logits')
        
        # Extract and process codebook indices if they exist
        codebook_indices = outputs.pop('codebook_indices')
        
        # Get lightweight codebook metrics and add to outputs for logging
        codebook_metrics = self.log_codebook_metrics(pl_module)
        outputs.update(codebook_metrics)

        # Get pre-formatted codebook usage metrics
        codebook_usage_metrics = self.calculate_codebook_usage(codebook_indices, pl_module.model_config.codebook_size)
        outputs.update(codebook_usage_metrics)
        
        # Instead of using an exact modulo, check if it's time to log.
        current_step = pl_module.global_step
        should_visualize = (current_step - self.last_logged_visualization) >= self.visualization_interval

        # Visualize if the time interval has been reached.
        if should_visualize:
            # Visualize reconstructions and pass codebook indices
            self.visualize_reconstructions(pl_module, x, logits, 'train', indices=codebook_indices)
            
            # Create and log codebook visualizations (visualization and logging now handled inside this method)
            self.visualize_codebook_state(pl_module, step=current_step, phase='train')
            
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
        
        # Extract and process codebook indices if they exist
        codebook_indices = outputs.pop('codebook_indices')
        
        # Get lightweight codebook metrics and add to outputs for logging
        # codebook_metrics = self.log_codebook_metrics(pl_module)
        # outputs.update(codebook_metrics)
        
        # Get pre-formatted codebook usage metrics
        codebook_usage_metrics = self.calculate_codebook_usage(codebook_indices, pl_module.model_config.codebook_size)
        outputs.update(codebook_usage_metrics)

        # Visualize reconstructions only for the randomly selected batch.
        if batch_idx == self.val_batch_to_visualize:
            self.visualize_reconstructions(pl_module, x, logits, 'val', indices=codebook_indices)
        
        # Calculate tokens per second for the validation batch.
        if self.val_batch_start_time is not None:
            tokens_per_sec, time_per_batch_ms = self._calculate_tokens_per_sec(self.val_batch_start_time, batch)
            print(f"\n[Eval]  {self.get_loss_string(outputs)} | T/s: {tokens_per_sec:.2f} | Δ(ms): {time_per_batch_ms:.1f}ms")
            outputs['tokens_per_sec'] = tokens_per_sec
            outputs['Δ_ms'] = time_per_batch_ms
        
        # Log loss metrics using the helper method.
        self._log_metrics(pl_module, 'val', outputs, batch[0].size(0), on_step=False, on_epoch=True)

    def _log_metrics(self, pl_module: pl.LightningModule, phase: str, outputs: Dict[str, torch.Tensor], batch_size: int, on_step: bool, on_epoch: bool):
        """
        Helper method to log metrics to WandB.
        """
        # Process each metric in the outputs dictionary
        for key, value in outputs.items():
            # Skip WandB Images since we handle them separately
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

    def log_codebook_metrics(self, pl_module):
        """
        Calculate lightweight numeric metrics about codebook state for every batch.
        Args:
            pl_module: The Lightning module with the model            
        Returns:
            dict: Dictionary containing metrics for logging
        """
        # Only track metrics if using EMA and not skipping codebook
        if not (pl_module.model_config.use_ema and pl_module.model_config.codebook_size > 0):
            return {}
        
        # Get the VQEmbedding instance
        vq_embedding = pl_module.model.codebook
        if vq_embedding is None:
            return {}
        
        # Calculate metrics while keeping tensors on their original device
        with torch.no_grad():
            # Get cluster size metrics
            cluster_sizes = vq_embedding.cluster_size
            max_cluster_size = cluster_sizes.max().item()
            mean_cluster_size = cluster_sizes.mean().item()
            min_cluster_size = cluster_sizes.min().item()
            unused_count = (cluster_sizes < vq_embedding.unused_reset_threshold).sum().item()
            unused_percent = unused_count / len(cluster_sizes) * 100.0
            
            # Get update magnitude metrics (if available)
            update_magnitudes = vq_embedding.update_magnitudes
            mean_update = update_magnitudes.mean().item()
            min_update = update_magnitudes.min().item()
            max_update = update_magnitudes.max().item()
        
        # Create metrics dictionary
        return {
            f'CodebookMetrics/cluster_size_mean': mean_cluster_size,
            f'CodebookMetrics/cluster_size_min': min_cluster_size,
            f'CodebookMetrics/cluster_size_max': max_cluster_size,
            f'CodebookMetrics/unused_count': unused_count,
            f'CodebookMetrics/unused_percent': unused_percent,
            f'CodebookMetrics/update_magnitude_mean': mean_update,
            f'CodebookMetrics/update_magnitude_min': min_update,
            f'CodebookMetrics/update_magnitude_max': max_update,
        }

@dataclass
class ExperimentConfig:
    """Configuration for training hyperparameters and schedules"""
    # Model configuration
    model_config: CTDAutoEncoderConfig = field(default_factory=CTDAutoEncoderConfig)
    
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
    beta_ce: float = 1.0
    beta_vq: float = 1.0
    beta_commitment: float = 0.25
    
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
    kmeans_init_codebook: bool = False
    kmeans_init_max_datapoints: int = 5_00_000
    kmeans_init_batch_size: int = 100_000

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
            config['model_config'] = CTDAutoEncoderConfig.from_dict(config['model_config'])
        
        # Handle DatasetType fields
        if 'train_ds' in config:
            config['train_ds'] = DatasetType[config['train_ds']]
        if 'val_ds' in config:
            config['val_ds'] = DatasetType[config['val_ds']]
            
        return cls(**config)

    @staticmethod
    def from_checkpoint(checkpoint_path: str) -> 'ExperimentConfig':
        # Add our custom classes to safe globals
        add_safe_globals([ExperimentConfig, CTDAutoEncoderConfig])
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        config = ckpt['hyper_parameters']['experiment_config']
        assert isinstance(config, ExperimentConfig)
        return config

class CTDAutoEncoderTrainingModule(pl.LightningModule):
    def __init__(self, experiment_config: ExperimentConfig, compile_model: bool = False):
        super(CTDAutoEncoderTrainingModule, self).__init__()
        self.experiment_config = experiment_config
        self.model_config = experiment_config.model_config
        self.model = None
        self.pad_idx = experiment_config.model_config.pad_idx
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

        self.model = CTDAutoEncoder(self.model_config)
        
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
        """
        Called at the beginning of training after dataloaders are initialized.
        Initialize codebook with k-means clustering if enabled in config.
        """
        super().on_train_start()
        
        # Initialize codebook with k-means if enabled
        if self.experiment_config.kmeans_init_codebook:
            print(f"Initializing codebook with k-means clustering...")
            
            # This line seems to work in your code, so leaving it as is
            train_dataloader = self.trainer.train_dataloader
            
            # Get the original uncompiled model
            target_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
            
            # Initialize codebook with k-means on the original model
            target_model.initialize_codebook_with_kmeans(
                data_loader=train_dataloader,
                device=self.device,
                max_datapoints=self.experiment_config.kmeans_init_max_datapoints,
                batch_size=self.experiment_config.kmeans_init_batch_size
            )
            
            # If using compilation, recompile to ensure optimizations work with new weights
            if self.compile_model and hasattr(self.model, "_orig_mod"):
                # Reapply compilation to ensure optimizations work with new weights
                print("Recompiling model with initialized codebook...")
                self.model = torch.compile(
                    target_model,
                    fullgraph=True,
                    mode="reduce-overhead",
                    backend="inductor"
                )
            
            print("Codebook initialization complete!")

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Handle loading checkpoints with different state dict keys and handle mismatched codebook sizes."""
        if "state_dict" in checkpoint:
            original_state_dict = checkpoint["state_dict"].copy()
            state_dict = checkpoint["state_dict"]
            
            # 1. Determine if the checkpoint model is compiled 
            is_checkpoint_compiled = any("_orig_mod" in key for key in state_dict.keys())
            
            # 2. Determine if our current model is compiled by checking its state dict structure
            # First make sure the model is configured
            if self.model is None:
                self.configure_model()
            
            # Get our model's state dict to check if it's compiled and for key structure
            current_state_dict = self.state_dict()
            is_current_model_compiled = any("_orig_mod" in key for key in current_state_dict.keys())
            
            # 3. Check for codebook size mismatch directly
            codebook_mismatch = False
            codebook_size_checkpoint = None
            codebook_size_current = None
            
            for key, value in state_dict.items():
                if "codebook.embedding.weight" in key or "codebook.vq_embs.weight" in key:
                    codebook_size_checkpoint = value.size(0)
                    codebook_size_current = self.model_config.codebook_size
                    
                    if codebook_size_current != codebook_size_checkpoint:
                        codebook_mismatch = True
                        print(f"WARNING: Codebook size mismatch! Current: {codebook_size_current}, Checkpoint: {codebook_size_checkpoint}")
                        print(f"Loading with strict=False to skip codebook parameters")
                    break
            
            # 4. Critical fix - ensure checkpoint state dict perfectly matches current model structure
            # but without codebook parameters when there's a mismatch
            new_state_dict = {}
            
            # Get the keys that should be in the final state dict from the current model
            for target_key in current_state_dict.keys():
                # Skip any codebook parameters if there's a mismatch
                if codebook_mismatch and "codebook" in target_key:
                    continue
                
                # Find the corresponding key in the checkpoint state dict
                source_key = None
                if is_current_model_compiled and not is_checkpoint_compiled:
                    # Current is compiled, checkpoint isn't
                    possible_source_key = target_key.replace("model._orig_mod.", "model.")
                    if possible_source_key in state_dict:
                        source_key = possible_source_key
                elif not is_current_model_compiled and is_checkpoint_compiled:
                    # Current isn't compiled, checkpoint is
                    possible_source_key = target_key.replace("model.", "model._orig_mod.")
                    if possible_source_key in state_dict:
                        source_key = possible_source_key
                else:
                    # Both are the same (compiled or not)
                    if target_key in state_dict:
                        source_key = target_key
                
                # If we found a matching source key, copy the value with data type conversion if needed
                if source_key is not None:
                    value = state_dict[source_key]
                    
                    # Handle data type conversion if needed
                    if hasattr(value, 'dtype') and value.dtype != torch.float32 and value.dtype != torch.int64:
                        try:
                            value = value.to(torch.float32)
                        except Exception as e:
                            print(f"WARNING: Failed to convert {source_key} from {value.dtype} to float32: {e}")
                    
                    new_state_dict[target_key] = value
            
            # 5. Replace the checkpoint's state dict with our precision-crafted version
            checkpoint["state_dict"] = new_state_dict
            
            # 6. Log the final stats
            if codebook_mismatch:
                print(f"Successfully loaded non-codebook weights")
                # Initialize codebook with k-means if enabled to get a good starting point
                if self.experiment_config.kmeans_init_codebook:
                    print("Will initialize codebook using k-means when training starts")
        else:
            print(f"Loaded all model weights")

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
                beta_ce: Tensor = torch.tensor(1.0),
                beta_commitment: Tensor = torch.tensor(1.0),
                beta_vq: Tensor = torch.tensor(1.0),
                mask_pct: Tensor = torch.tensor(0.0)):
        
        # Forward pass with provided scheduled parameters.
        logits, meta_dict = self.model.forward(x, mask_percentage=mask_pct)
        
        # Calculate token accuracy excluding padding tokens.
        non_padding_mask = (x != self.pad_idx)  # Shape: [B, H, W]
        predictions = logits.argmax(dim=-1)  # Shape: [B, H, W]

        # Simple token accuracy calculation - must filter out padding tokens
        total_non_padding_tokens = non_padding_mask.sum().float()
        correct_non_padding_tokens = ((predictions == x) & non_padding_mask).sum().float()
        token_accuracy = correct_non_padding_tokens / total_non_padding_tokens

        # Sample accuracy - a sample is correct if ALL tokens match (including padding)
        sample_correct = torch.all(predictions == x, dim=(1, 2)).float()  # Shape: [B]
        sample_accuracy = sample_correct.mean()
        
        
        # Compute total loss in a way that maintains the computational graph.
        # Scale the KLD losses by the number of latents (similar to Dalle-E paper).
        raw_losses = {
            'loss(CE)': meta_dict['ce_loss'],
            'loss(VQ)': meta_dict.get('vq_loss', torch.tensor(0.0, device=x.device)),
            'loss(Commitment)': meta_dict.get('commitment_loss', torch.tensor(0.0, device=x.device)),
        }
        
        # Compute weighted losses for total loss.
        weighted_losses = {
            'loss(CE)': raw_losses['loss(CE)'] * beta_ce,
            'loss(VQ)': raw_losses['loss(VQ)'] * beta_vq,
            'loss(Commitment)': raw_losses['loss(Commitment)'] * beta_commitment,
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
            'codebook_indices': meta_dict.get('indices', None)
        }

    def training_step(self, batch, batch_idx):
        torch.compiler.cudagraph_mark_step_begin()
        x, _ = batch

        # Extract scheduled values and convert them to tensors on the current device.
        scheduled = self.get_scheduled_values(self.global_step, device=x.device)

        beta_commitment = torch.tensor(self.experiment_config.beta_commitment, device=x.device)
        beta_ce = torch.tensor(self.experiment_config.beta_ce, device=x.device)
        beta_vq = torch.tensor(self.experiment_config.beta_vq, device=x.device)
        # Sample mask percentage for this batch; set max_pct_mask to 0 in validation for no masking.
        mask_pct = torch.empty(1, device=x.device).uniform_(0.0, scheduled['max_pct(MASK)'])[0]

        output_dict = self(
             x,
             beta_commitment=beta_commitment,
             beta_ce=beta_ce,
             beta_vq=beta_vq,
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
        beta_ce = torch.tensor(self.experiment_config.beta_ce, device=x.device)
        beta_vq = torch.tensor(self.experiment_config.beta_vq, device=x.device)
        mask_pct = torch.tensor(0.0, device=x.device) # No masking in validation

        output_dict = self(
             x,
             beta_commitment=beta_commitment,
             beta_ce=beta_ce,
             beta_vq=beta_vq,
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
        max_width=max_grid_width,
        flatten=False
    )

    train_dataset = GridDataset(dataset_type=train_ds, max_samples=limit_training_samples, min_samples=min_training_samples)

    # Create validation dataset
    collate_fn_val = partial(
        GridDataset.collate_fn_project,
        pad_value=padding_idx,
        permute=False,
        max_height=max_grid_height,
        max_width=max_grid_width,
        flatten=False
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
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.configure_model()  # Need to load the model first

        model_sd = model.state_dict()
        ckpt_sd = checkpoint['state_dict']

        for k in model_sd:
            if k not in ckpt_sd:
                print(f"Skipping missing key {k}")
                ckpt_sd[k] = model_sd[k]

        print(f"Loading all parameters with strict=True")
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        print(f"Successfully loaded all model weights from {checkpoint_path}")
        
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
        padding_idx=experiment_config.model_config.pad_idx,
        max_grid_height=experiment_config.model_config.grid_height,
        max_grid_width=experiment_config.model_config.grid_width,
        permute_train=experiment_config.permute_train,
        limit_training_samples=experiment_config.limit_training_samples,
        min_training_samples=val_check_interval * experiment_config.batch_size if not validation_disabled else None,
        train_ds=experiment_config.train_ds,
        val_ds=experiment_config.val_ds
    )

    codebook_size = experiment_config.model_config.codebook_size
    latent_bits = 0 if codebook_size == 0 else experiment_config.model_config.n_codes * math.log2(codebook_size)

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
        enable_model_summary=not lr_find
        # detect_anomaly=True if debug_mode else False
    )

    with trainer.init_module():
        # Initialize the model
        model = CTDAutoEncoderTrainingModule(
            experiment_config,
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
    
    # Add experiment config as a nested dictionary
    defaults['experiment_config'] = config_dict
    
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
    compile_model: bool = typer.Option(True, "--no-compile", help="Disable model compilation", is_flag=True, flag_value=False),
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
    acceleration_config = AccelerationConfig(
        device='auto',
        precision='bf16-true',
        matmul_precision='high',
        compile_model=compile_model
    )
    
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