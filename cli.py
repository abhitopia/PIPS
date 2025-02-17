#!/usr/bin/env python3

import typer
from pathlib import Path
import logging
from train_dvae import ExperimentConfig, GridDVAEConfig, train, generate_friendly_name
from torch.serialization import add_safe_globals
import wandb
from pips.misc.artifact import Artifact
from pips.misc.acceleration_config import AccelerationConfig
import click


# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Typer app
app = typer.Typer(
    name="pips",
    help="Perception Informed Program Synthesis CLI",
    add_completion=False,
    pretty_exceptions_show_locals=False
)



# Create a sub-Typer app for dvae commands
dvae_app = typer.Typer(help="Train, evaluate, and manage DVAE models.")

# Register the dvae sub-app with the main app
app.add_typer(dvae_app, name="dvae")

# Add to safe globals before any checkpoint loading
add_safe_globals([ExperimentConfig, GridDVAEConfig])

def get_common_options():
    """Returns common options used across commands."""
    model_src_help = (
        "Format: [project/]run_name/{best|backup}[/{alias|step}] where:\n"
        "- alias can be 'best', 'best-N', 'latest', 'step-NNNNNNN'\n"
        "- step is a positive integer that will be converted to 'step-NNNNNNN' format"
    )
    
    return {
        # Model source options
        "model_src_option": typer.Option(None, "--model-src", "-m", help=model_src_help),
        "model_src_argument": typer.Argument(..., help=model_src_help),
        
        # Acceleration options
        "compile_model": typer.Option(True, "--no-compile", help="Disable model compilation if specified", is_flag=True, flag_value=False),
        "matmul_precision": typer.Option("high", "--matmul-precision", help="Matmul precision.", case_sensitive=False, show_choices=True, click_type=click.Choice(list(AccelerationConfig.VALID_MATMUL_PRECISIONS))),
        "precision": typer.Option("bf16-true", "--precision", help="Training precision. Note: will be overridden to '32-true' if device is cpu.", case_sensitive=False, show_choices=True, click_type=click.Choice(list(AccelerationConfig.VALID_PRECISIONS))),
        "device": typer.Option("auto", "--device", help="Device to use. 'auto' selects cuda if available, else cpu.", case_sensitive=False, show_choices=True, click_type=click.Choice(list(AccelerationConfig.VALID_DEVICES))),
        
        # Project tracking options
        "project_name": typer.Option("dvae-training", "--project", "-p", help="Project name for experiment tracking"),
        "checkpoint_dir": typer.Option(Path("./runs"), "--checkpoint-dir", "-d", help="Base directory for checkpoints"),
        # Debug option
        "debug": typer.Option(False, "--debug", "-D", help="Enable debug mode with reduced dataset and steps"),
        # Add val_check_interval to common options
        "val_check_interval": typer.Option(5000, "--val-check-interval", "--vci", help="Number of steps between validation checks"),

        "lr_find": typer.Option(False, "--lr-find", help="Run learning rate finder instead of training"),

    }

@dvae_app.command("train")
def new(
    # Project tracking
    project_name: str = get_common_options()["project_name"],
    run_name: str = typer.Option(None, "--name", "-n", help="Run name (generated if not specified)"),
    checkpoint_dir: Path = get_common_options()["checkpoint_dir"],
    
    # Model loading
    model_src: str = get_common_options()["model_src_option"],
    
    # Model architecture
    n_dim: int = typer.Option(256, "--n-dim", "-d", help="Dimension of model embeddings"),
    n_head: int = typer.Option(8, "--n-head", "-h", help="Number of attention heads"),
    n_layers: int = typer.Option(4, "--n-layers", "-l", help="Number of transformer layers"),
    n_codes: int = typer.Option(16, "--n-codes", "-c", help="Number of latent codes"),
    codebook_size: int = typer.Option(512, "--codebook-size", "--cs", help="Size of each codebook"),
    dropout: float = typer.Option(0.0, "--dropout", "--dp", help="Dropout rate"),
    
    # Sampling parameters
    hard_from: int = typer.Option(
        None, 
        "--hard-from", 
        "--hf",
        help="When to start hard sampling. None: after LR warmup, 0: always hard, >0: after specific step"
    ),
    
    # Training parameters
    batch_size: int = typer.Option(64, "--batch-size", "--bs", help="Training batch size"),
    learning_rate: float = typer.Option(1e-4, "--learning-rate", "--lr", help="Initial learning rate"),
    weight_decay: float = typer.Option(1e-4, "--weight-decay", "--wd", help="AdamW weight decay"),
    max_steps: int = typer.Option(1_000_000, "--max-steps", "--ms", help="Maximum training steps"),
    gradient_clip_val: float = typer.Option(1.0, "--gradient-clip-val", "--gc", help="Gradient clipping value"),
    accumulate_grad_batches: int = typer.Option(1, "--accumulate-grad-batches", "--acc", help="Number of batches to accumulate gradients"),
    
    # Split warmup steps into separate parameters
    warmup_steps_lr: int = typer.Option(10_000, "--warmup-steps-lr", "--wsl", help="Learning rate warmup steps"),
    warmup_steps_tau: int = typer.Option(150_000, "--warmup-steps-tau", "--wst", help="Temperature warmup steps"),
    warmup_steps_beta: int = typer.Option(10_000, "--warmup-steps-beta", "--wsb", help="Beta parameters warmup steps"),
    
    # Regularization parameters
    tau_start: float = typer.Option(1.0, "--tau-start", "--ts", help="Starting temperature for Gumbel-Softmax"),
    tau: float = typer.Option(0.0625, "--tau", "-t", help="Final temperature for Gumbel-Softmax"),
    max_mask_pct: float = typer.Option(0.5, "--max-mask-pct", "--msk", help="Maximum masking percentage during training. Warms up with beta schedule"),
    
    # Beta values for loss components
    beta_mi: float = typer.Option(0.0, "--beta-mi", "--bmi", help="Beta for mutual information loss"),
    beta_tc: float = typer.Option(6.0, "--beta-tc", "--btc", help="Beta for total correlation loss"),
    beta_dwkl: float = typer.Option(0.0, "--beta-dwkl", "--bdw", help="Beta for dimension-wise KL loss"),
    beta_kl: float = typer.Option(2.0, "--beta-kl", "--bkl", help="Beta for KL loss"),
    
    # Add seed parameter
    seed: int = typer.Option(
        None, 
        "--seed", 
        "-s", 
        help="Random seed for reproducibility. If not provided, a random seed will be generated."
    ),

    # Common options
    val_check_interval: int = get_common_options()["val_check_interval"],
    debug: bool = get_common_options()["debug"],
    compile_model: bool = get_common_options()["compile_model"],
    matmul_precision: str = get_common_options()["matmul_precision"],
    precision: str = get_common_options()["precision"],
    device: str = get_common_options()["device"],
    lr_find: bool = get_common_options()["lr_find"],
):
    """Train a new DVAE model with specified configuration."""
    
    # Append debug suffix to project name if in debug mode
    project_name = f"{project_name}-debug" if debug else project_name
    
    # Create acceleration config (will validate and resolve settings)
    acceleration = AccelerationConfig(
        device=device,
        precision=precision,
        matmul_precision=matmul_precision,
        compile_model=compile_model
    )
    
    # Create fresh config with CLI parameters
    model_config = GridDVAEConfig(
        n_dim=n_dim,
        n_head=n_head,
        n_layers=n_layers,
        n_codes=n_codes,
        codebook_size=codebook_size,
        rope_base=10000,
        dropout=dropout,
        n_vocab=16,  # Fixed for grid world
        max_grid_height=32,  # Fixed for grid world
        max_grid_width=32,  # Fixed for grid world
    )
    
    config = ExperimentConfig(
        model_config=model_config,
        model_src=model_src,
        hard_from=hard_from,
        tau_start=tau_start,
        tau=tau,
        beta_mi_start=0.0,
        beta_tc_start=0.0,
        beta_dwkl_start=0.0,
        beta_kl_start=0.0,
        beta_mi=beta_mi,
        beta_tc=beta_tc,
        beta_dwkl=beta_dwkl,
        beta_kl=beta_kl,
        warmup_steps_lr=warmup_steps_lr,
        warmup_steps_tau=warmup_steps_tau,
        warmup_steps_beta=warmup_steps_beta,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_steps=max_steps,
        max_mask_pct=max_mask_pct,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        seed=seed,
    )

    run_name = generate_friendly_name() if run_name is None else run_name

    # Start training with the resolved settings
    train(
        experiment_config=config,
        run_name=run_name,
        project_name=project_name,
        checkpoint_dir=checkpoint_dir,
        debug_mode=debug,
        lr_find=lr_find,
        acceleration=acceleration,
        val_check_interval=val_check_interval
    )


@dvae_app.command()
def resume(
    model_src: str = get_common_options()["model_src_argument"],
    project_name: str = get_common_options()["project_name"],
    checkpoint_dir: Path = get_common_options()["checkpoint_dir"],
    debug: bool = get_common_options()["debug"],
    compile_model: bool = get_common_options()["compile_model"],
    matmul_precision: str = get_common_options()["matmul_precision"],
    precision: str = get_common_options()["precision"],
    device: str = get_common_options()["device"],
    batch_size: int = typer.Option(None, "--batch-size", "--bs", help="Override training batch size"),
    learning_rate: float = typer.Option(None, "--learning-rate", "--lr", help="Override learning rate"),
    lr_find: bool = get_common_options()["lr_find"],
    val_check_interval: int = get_common_options()["val_check_interval"],
):
    """Resume training from a checkpoint."""
    
    # Append debug suffix to project name if in debug mode
    project_name = f"{project_name}-debug" if debug else project_name
    
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
    local_checkpoint_path = artifact_manager.get_local_checkpoint(
        category=category,
        alias=alias,
        checkpoint_dir=checkpoint_dir
    )

    # Load config and resume training
    config = ExperimentConfig.from_checkpoint(str(local_checkpoint_path))
    
    # Override batch size and learning rate if provided
    if batch_size is not None:
        config.batch_size = batch_size
    if learning_rate is not None:
        config.learning_rate = learning_rate

    # Create acceleration config (will validate and resolve settings)
    acceleration = AccelerationConfig(
        device=device,
        precision=precision,
        matmul_precision=matmul_precision,
        compile_model=compile_model
    )

    train(
        experiment_config=config,
        run_name=run_name,
        project_name=project_name,
        checkpoint_dir=checkpoint_dir,
        debug_mode=debug,
        resume_from=local_checkpoint_path,
        acceleration=acceleration,
        val_check_interval=val_check_interval,
        lr_find=lr_find
    )


if __name__ == "__main__":
    app() 