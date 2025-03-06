#!/usr/bin/env python3

import typer
from pathlib import Path
import logging
from train_dvae import ExperimentConfig, GridDVAEConfig, train
from torch.serialization import add_safe_globals
import wandb
from pips.misc.artifact import Artifact
from pips.utils import generate_friendly_name
from pips.misc.acceleration_config import AccelerationConfig
import click
from rich import print


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
        # Modify val_check_interval help text to include info about negative values
        "val_check_interval": typer.Option(
            5000, 
            "--val-check-interval", 
            "--vci", 
            help="Number of steps between validation checks. Set to negative value to disable validation."
        ),

        "lr_find": typer.Option(False, "--lr-find", help="Run learning rate finder instead of training"),

        # Add visualization interval option
        "viz_interval": typer.Option(
            100, 
            "--viz-interval", 
            "--vi",
            help="Number of steps between visualizations and gradient logging"
        ),

        "wandb_logging": typer.Option(True, "--no-wandb", help="Disable WandB logging", is_flag=True, flag_value=False),

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
    n_grid_layer: int = typer.Option(4, "--n-grid-layer", "-gl", help="Number of grid transformer layers"),
    n_latent_layer: int = typer.Option(4, "--n-latent-layer", "-ll", help="Number of latent transformer layers"),
    n_codes: int = typer.Option(16, "--n-codes", "-c", help="Number of latent codes"),
    codebook_size: int = typer.Option(512, "--codebook-size", "--cs", help="Size of each codebook"),
    dropout: float = typer.Option(0.0, "--dropout", "--dp", help="Dropout rate"),
    gamma: float = typer.Option(2.0, "--gamma", "--g", help="Focal loss gamma parameter (default: 2.0)"),
    pad_weight: float = typer.Option(0.01, "--pad-weight", "--pw", help="Weight for pad token loss (default: 0.01 = 1% of normal weight)"),
    use_exp_relaxed: bool = typer.Option(False, "--exp-relaxed", help="Use exponentially relaxed Gumbel-Softmax"),
    use_monte_carlo_kld: bool = typer.Option(False, "--monte-carlo-kld", help="Use Monte Carlo KLD estimation instead of approximate KLD"),
    
    # Training parameters
    batch_size: int = typer.Option(64, "--batch-size", "--bs", help="Training batch size"),
    learning_rate: float = typer.Option(1e-4, "--learning-rate", "--lr", help="Initial learning rate"),
    lr_min: float = typer.Option(None, "--lr-min", help="Minimum learning rate (defaults to 1% of learning rate if not specified)"),
    weight_decay: float = typer.Option(1e-4, "--weight-decay", "--wd", help="AdamW weight decay"),
    max_steps: int = typer.Option(1_000_000, "--max-steps", "--ms", help="Maximum training steps"),
    gradient_clip_val: float = typer.Option(1.0, "--gradient-clip-val", "--gc", help="Gradient clipping value"),
    accumulate_grad_batches: int = typer.Option(1, "--accumulate-grad-batches", "--acc", help="Number of batches to accumulate gradients"),
    
    # Learning rate and other warmup / decay steps
    warmup_steps_lr: int = typer.Option(10_000, "--warmup-steps-lr", "--wsl", help="Learning rate warmup steps"),
    decay_steps_lr: int = typer.Option(None, "--decay-steps-lr", "--dsl", help="Learning rate decay steps, if not specified, will be set to max_steps - warmup_steps_lr"),
    warmup_steps_tau: int = typer.Option(150_000, "--warmup-steps-tau", "--wst", help="Temperature warmup steps"),
    warmup_steps_beta: int = typer.Option(10_000, "--warmup-steps-beta", "--wsb", help="Beta parameters warmup steps"),
    warmup_steps_mask_pct: int = typer.Option(50_000, "--warmup-steps-mask-pct", "--wsm", help="Mask percentage warmup steps"),

    # Regularization parameters
    tau_start: float = typer.Option(3.5, "--tau-start", "--ts", help="Starting temperature for Gumbel-Softmax"),
    tau: float = typer.Option(0.0625, "--tau", "-t", help="Final temperature for Gumbel-Softmax"),
    max_mask_pct: float = typer.Option(0.0, "--max-mask-pct", "--msk", help="Maximum masking percentage during training"),
    
    # Beta values for loss components
    beta_ce: float = typer.Option(1.0, "--beta-ce", "--bce", help="Beta for cross-entropy loss. Stays constant."),
    beta_kl: float = typer.Option(1.0, "--beta-kl", "--bkl", help="Beta for KL loss"),
    beta_mi: float = typer.Option(0.0, "--beta-mi", "--bmi", help="Beta for mutual information loss"),
    beta_tc: float = typer.Option(0.0, "--beta-tc", "--btc", help="Beta for total correlation loss"),
    beta_dwkl: float = typer.Option(0.0, "--beta-dwkl", "--bdw", help="Beta for dimension-wise KL loss"),
    
    # Training data options
    seed: int = typer.Option(None, "--seed", "-s", help="Random seed for reproducibility. If not provided, a random seed will be generated."),
    limit_training_samples: int = typer.Option(None, "--limit-samples", "--lts", help="Limit the number of training samples. None means use all samples."),
    shuffle_train: bool = typer.Option(True, "--no-shuffle", help="Shuffle the training data. Default is True.", is_flag=True, flag_value=False),

    # Logging options
    viz_interval: int = get_common_options()["viz_interval"],
    val_check_interval: int = get_common_options()["val_check_interval"],
    debug: bool = get_common_options()["debug"],
    wandb_logging: bool = get_common_options()["wandb_logging"],

    # Acceleration options
    matmul_precision: str = get_common_options()["matmul_precision"],
    precision: str = get_common_options()["precision"],
    compile_model: bool = get_common_options()["compile_model"],
    device: str = get_common_options()["device"],

    # Misc options
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
        n_grid_layer=n_grid_layer,
        n_latent_layer=n_latent_layer,
        n_codes=n_codes,
        codebook_size=codebook_size,
        dropout=dropout,
        gamma=gamma,
        n_vocab=16,  # Fixed for grid world
        max_grid_height=32,  # Fixed for grid world
        max_grid_width=32,  # Fixed for grid world
        pad_weight=pad_weight,
        use_exp_relaxed=use_exp_relaxed,
        use_monte_carlo_kld=use_monte_carlo_kld,
    )
    
    config = ExperimentConfig(
        model_config=model_config,
        model_src=model_src,
        tau_start=tau_start,
        tau=tau,
        beta_ce_start=beta_ce,
        beta_mi_start=0.0,
        beta_tc_start=0.0,
        beta_dwkl_start=0.0,
        beta_kl_start=0.0,
        mask_pct_start=0.0,
        beta_ce=beta_ce,
        beta_mi=beta_mi,
        beta_tc=beta_tc,
        beta_dwkl=beta_dwkl,
        beta_kl=beta_kl,
        warmup_steps_lr=warmup_steps_lr,
        decay_steps_lr=decay_steps_lr,
        warmup_steps_tau=warmup_steps_tau,
        warmup_steps_beta=warmup_steps_beta,
        warmup_steps_mask_pct=warmup_steps_mask_pct,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lr_min=lr_min if lr_min is not None else learning_rate * 0.01,
        weight_decay=weight_decay,
        max_steps=max_steps,
        max_mask_pct=max_mask_pct,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        seed=seed,
    )

    run_name = generate_friendly_name() if run_name is None else run_name

    print(config.to_dict())
    # Start training with the resolved settings
    train(
        experiment_config=config,
        run_name=run_name,
        project_name=project_name,
        checkpoint_dir=checkpoint_dir,
        debug_mode=debug,
        lr_find=lr_find,
        acceleration=acceleration,
        val_check_interval=val_check_interval,
        limit_training_samples=limit_training_samples,
        shuffle_train=shuffle_train,
        visualization_interval=viz_interval,
        grad_log_interval=viz_interval,
        wandb_logging=wandb_logging
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
    viz_interval: int = get_common_options()["viz_interval"],
    wandb_logging: bool = get_common_options()["wandb_logging"],
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
        lr_find=lr_find,
        visualization_interval=viz_interval,
        grad_log_interval=viz_interval,
        wandb_logging=wandb_logging
    )


if __name__ == "__main__":
    app() 