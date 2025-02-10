import typer
from pathlib import Path
import logging
from train_dvae import ExperimentConfig, GridDVAEConfig, train, generate_friendly_name
from torch.serialization import add_safe_globals
import wandb
from pips.misc.artifact import Artifact


# Initialize logger
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
    debug_logging: bool = typer.Option(False, "--debug-logging", "-L", help="Enable logging in debug mode"),
    checkpoint_path: str = typer.Option(None, "--checkpoint-path", "--ckpt", help="Path to checkpoint for resuming training"),
):
    """Train a new DVAE model with specified configuration."""

    if checkpoint_path:
        # Resume training from checkpoint
        config = ExperimentConfig.from_checkpoint(checkpoint_path)
        assert run_name is not None, "--run_name must be None when checkpoint path is specified"
        run_name = Path(checkpoint_path).parent.parent.name
    else:
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
        )
        run_name = generate_friendly_name() if run_name is None else run_name

    # Start training with project name and checkpoint directory
    train(
        experiment_config=config,
        run_name=run_name,
        project_name=f"{project_name}-debug" if debug else project_name,
        checkpoint_dir=checkpoint_dir,
        debug_mode=debug,
        debug_logging=debug_logging,
    )


@dvae_app.command()
def resume(
    run_name: str = typer.Argument(..., help="Run name to resume from"),
    project_name: str = typer.Option("dvae-training", "--project", "-p", help="Project name for experiment tracking"),
    step: int = typer.Option(None, "--step", "-s", help="Step number to resume from"),
    alias: str = typer.Option(None, "--alias", "-a", help="Alias to resume from (e.g. 'best', 'best-2', 'step-0001000')"),
    backup: bool = typer.Option(False, "--backup", "-B", help="Use backup checkpoints instead of best checkpoints"),
    checkpoint_dir: Path = typer.Option(
        Path("./runs"), 
        "--checkpoint-dir", 
        "-d", 
        help="Base directory for checkpoints"
    ),
    debug: bool = typer.Option(False, "--debug", "-D", help="Enable debug mode with reduced dataset and steps"),
    debug_logging: bool = typer.Option(False, "--debug-logging", "-L", help="Enable logging in debug mode"),
):
    """Resume training from a checkpoint."""
    
    # Initialize artifact manager
    artifact_manager = Artifact(
        entity=wandb.api.default_entity,
        project_name=project_name,
        run_name=run_name
    )

    # Get artifacts for the specified category
    category = "backup" if backup else "best"
    artifacts = artifact_manager.get_artifacts(category)
    if not artifacts:
        raise ValueError(f"No artifacts found for run '{run_name}' in category '{category}'")

    # If no specific checkpoint requested, list available ones and exit
    if step is None and alias is None:
        artifact_manager.display_checkpoints_table(artifacts)
        return

    # Find and ensure local checkpoint exists
    matching_artifact = artifact_manager.find_matching_artifact(artifacts, step, alias)
    local_checkpoint_path = artifact_manager.ensure_local_checkpoint(matching_artifact, checkpoint_dir)

    # Load config and resume training
    config = ExperimentConfig.from_checkpoint(str(local_checkpoint_path))
    train(
        experiment_config=config,
        run_name=run_name,
        project_name=f"{project_name}-debug" if debug else project_name,
        checkpoint_dir=checkpoint_dir,
        debug_mode=debug,
        debug_logging=debug_logging,
    )


@dvae_app.command()
def lr_find():
    """Run learning rate finder to determine optimal learning rate."""
    pass


@dvae_app.command()
def evaluate():
    """Evaluate a trained DVAE model."""
    pass


if __name__ == "__main__":
    app() 