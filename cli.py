#!/usr/bin/env python3

import typer
from pathlib import Path
import logging
from train_dvae import ExperimentConfig, GridDVAEConfig, train, generate_friendly_name
from torch.serialization import add_safe_globals
import wandb
from pips.misc.artifact import Artifact


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

@dvae_app.command("train")
def new(
    # Project tracking
    project_name: str = typer.Option("dvae-training", "--project", "-p", help="Project name for experiment tracking"),
    run_name: str = typer.Option(None, "--name", "-n", help="Run name (generated if not specified)"),
    checkpoint_dir: Path = typer.Option(
        Path("./runs"), 
        "--checkpoint-dir", 
        "-d", 
        help="Base directory for checkpoints"
    ),
    
    # Model loading
    model_src: str = typer.Option(
        None, 
        "--model-src", 
        "-m",
        help="Format: [project/]run_name/{best|backup}[/{alias|step}] where:\n"
             "- alias can be 'best', 'best-N', 'latest', 'step-NNNNNNN'\n"
             "- step is a positive integer that will be converted to 'step-NNNNNNN' format"
    ),
    
    # Model architecture
    n_dim: int = typer.Option(256, "--n-dim", "-d", help="Dimension of model embeddings"),
    n_head: int = typer.Option(8, "--n-head", "-h", help="Number of attention heads"),
    n_layers: int = typer.Option(6, "--n-layers", "-l", help="Number of transformer layers"),
    n_codes: int = typer.Option(16, "--n-codes", "-c", help="Number of latent codes"),
    codebook_size: int = typer.Option(512, "--codebook-size", "--cs", help="Size of each codebook"),
    dropout: float = typer.Option(0.0, "--dropout", "--dp", help="Dropout rate"),
    
    # Training parameters
    batch_size: int = typer.Option(4, "--batch-size", "--bs", help="Training batch size"),
    learning_rate: float = typer.Option(1e-3, "--learning-rate", "--lr", help="Initial learning rate"),
    weight_decay: float = typer.Option(0.01, "--weight-decay", "--wd", help="AdamW weight decay"),
    max_steps: int = typer.Option(100000, "--max-steps", "--ms", help="Maximum training steps"),
    warmup_steps: int = typer.Option(5000, "--warmup-steps", "--ws", help="Learning rate warmup steps"),
    gradient_clip_val: float = typer.Option(1.0, "--gradient-clip-val", "--gc", help="Gradient clipping value"),
    accumulate_grad_batches: int = typer.Option(1, "--accumulate-grad-batches", "--acc", help="Number of batches for gradient accumulation"),
    
    # Regularization parameters
    initial_tau: float = typer.Option(0.9, "--initial-tau", "--tau", help="Initial temperature for Gumbel-Softmax"),
    final_tau: float = typer.Option(0.1, "--final-tau", "--ft", help="Final temperature for Gumbel-Softmax"),
    max_mask_pct: float = typer.Option(0.5, "--max-mask-pct", "--msk", help="Maximum masking percentage during training"),
    
    # Beta values for loss components
    target_beta_mi: float = typer.Option(1.0, "--target-beta-mi", "--bmi", help="Target beta for mutual information loss"),
    target_beta_tc: float = typer.Option(1.0, "--target-beta-tc", "--btc", help="Target beta for total correlation loss"),
    target_beta_dwkl: float = typer.Option(1.0, "--target-beta-dwkl", "--bdw", help="Target beta for dimension-wise KL loss"),
    target_beta_kl: float = typer.Option(1.0, "--target-beta-kl", "--bkl", help="Target beta for KL loss"),
    
    # Training mode
    debug: bool = typer.Option(False, "--debug", "-D", help="Enable debug mode with reduced dataset and steps"),
    
    # Add seed parameter
    seed: int = typer.Option(
        None, 
        "--seed", 
        "-s", 
        help="Random seed for reproducibility. If not provided, a random seed will be generated."
    ),
    lr_find: bool = typer.Option(False, "--lr-find", help="Run learning rate finder instead of training"),
    compile_model: bool = typer.Option(
        True,
        "--compile-model/--no-compile-model",
        help="Compile model using torch.compile if using GPU. Defaults to True; use --no-compile-model to disable."
    ),
):
    """Train a new DVAE model with specified configuration."""
    
    # Add debug suffix to project name if in debug mode
    project_name = f"{project_name}-debug" if debug else project_name
    
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
        model_src=model_src,  # Add model source to config
        initial_tau=initial_tau,
        min_tau=final_tau,
        initial_beta_mi=0.0,
        initial_beta_tc=0.0,
        initial_beta_dwkl=0.0,
        initial_beta_kl=0.0,
        target_beta_mi=target_beta_mi,
        target_beta_tc=target_beta_tc,
        target_beta_dwkl=target_beta_dwkl,
        target_beta_kl=target_beta_kl,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_steps=max_steps,
        max_mask_pct=max_mask_pct,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        seed=seed,  # Add seed to config
    )

    run_name = generate_friendly_name() if run_name is None else run_name

    # Start training with project name and checkpoint directory
    train(
        experiment_config=config,
        run_name=run_name,
        project_name=project_name,
        checkpoint_dir=checkpoint_dir,
        debug_mode=debug,
        lr_find=lr_find,
        compile_model=compile_model,
    )


@dvae_app.command()
def resume(
    model_src: str = typer.Argument(
        ..., 
        help="Format: [project/]run_name/{best|backup}[/{alias|step}] where:\n"
             "- alias can be 'best', 'best-N', 'latest', 'step-NNNNNNN'\n"
             "- step is a positive integer that will be converted to 'step-NNNNNNN' format"
    ),
    project_name: str = typer.Option("dvae-training", "--project", "-p", help="Project name for experiment tracking"),
    checkpoint_dir: Path = typer.Option(Path("./runs"), "--checkpoint-dir", "-d", help="Base directory for checkpoints"),
    debug: bool = typer.Option(False, "--debug", "-D", help="Enable debug mode with reduced dataset and steps"),
    compile_model: bool = typer.Option(
        True,
        "--compile-model/--no-compile-model",
        help="Compile model using torch.compile if using GPU. Defaults to True; use --no-compile-model to disable."
    ),
):
    """Resume training from a checkpoint."""
    
    # Add debug suffix to project name if in debug mode
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
    train(
        experiment_config=config,
        run_name=run_name,
        project_name=project_name,
        checkpoint_dir=checkpoint_dir,
        debug_mode=debug,
        resume_from=local_checkpoint_path,
        compile_model=compile_model,
    )


if __name__ == "__main__":
    app() 