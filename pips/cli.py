import typer
import hydra
from omegaconf import DictConfig
from pathlib import Path
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize Typer app
app = typer.Typer(
    name="pips",
    help="Perception Informed Program Synthesis CLI",
    add_completion=False,
)

@app.command()
def train(
    config_path: Path = typer.Option(
        "config/config.yaml",
        help="Path to the config file",
        exists=True,
    ),
    overrides: list[str] = typer.Option(
        None,
        help="Hydra overrides (e.g. optimizer.lr=0.1)",
    ),
):
    """Train a model with the given configuration."""
    
    @hydra.main(version_base=None, config_path=str(config_path.parent), config_name=config_path.stem)
    def _train(cfg: DictConfig) -> None:
        logger.info(f"Training with config: {cfg}")
        # Add your training logic here
        pass

    # Convert overrides list to argv format for Hydra
    argv = [] if overrides is None else [f"{override}" for override in overrides]
    _train(argv)

@app.command()
def synthesize(
    input_grid: Path = typer.Argument(
        ...,
        help="Path to the input grid file",
        exists=True,
    ),
    config_path: Path = typer.Option(
        "config/config.yaml",
        help="Path to the config file",
        exists=True,
    ),
    overrides: list[str] = typer.Option(
        None,
        help="Hydra overrides",
    ),
):
    """Synthesize a program for the given input grid."""
    
    @hydra.main(version_base=None, config_path=str(config_path.parent), config_name=config_path.stem)
    def _synthesize(cfg: DictConfig) -> None:
        logger.info(f"Synthesizing with config: {cfg}")
        logger.info(f"Input grid: {input_grid}")
        # Add your synthesis logic here
        pass

    argv = [] if overrides is None else [f"{override}" for override in overrides]
    _synthesize(argv)

if __name__ == "__main__":
    app() 