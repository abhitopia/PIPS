#!/usr/bin/env python

from pips.utils import generate_friendly_name
from prosip.model import train, ProSIPExperimentConfig
from pathlib import Path

import matplotlib.pyplot as plt
from pips.misc.acceleration_config import AccelerationConfig
from rich import print
import yaml
import typer

# Initialize Typer app
app = typer.Typer(
    name="prosip",
    help="PROgram Synthesis Informed by Perception CLI",
    add_completion=True,
    pretty_exceptions_show_locals=False
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
    default_config = ProSIPExperimentConfig()
    config_dict = default_config.to_dict()
    
    # Add experiment config as a nested dictionary
    defaults['experiment_config'] = config_dict
    
    # Add 'run_name', 'project_name', 'checkpoint_dir' under project_config
    defaults['project_config'] = {
        'run_name': generate_friendly_name(),
        'project_name': 'prosip-test',
        'checkpoint_dir': 'prosip_runs',
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
    num_workers: int = typer.Option(8, "--num-workers", "-W", help="Number of workers for dataloader"),
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
    experiment_config = ProSIPExperimentConfig.from_dict(experiment_config_dict)

    # Extract acceleration config and create AccelerationConfig instance
    acceleration_config = AccelerationConfig(
        device='auto',
        precision='bf16-mixed',
        matmul_precision='high',
        compile_model=compile_model
    )
    
    project_config = config_dict.pop('project_config')
    project_name = project_config['project_name']

    if debug_mode:
        project_name = f"{project_name}-debug"

    checkpoint_dir = Path(project_config['checkpoint_dir'])
    
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
        num_workers=num_workers,
        resume_from=None
    )

if __name__ == '__main__':
    app()