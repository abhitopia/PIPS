#!/usr/bin/env python3

import typer
from pathlib import Path
import logging
from train_dvae import ExperimentConfig, GridDVAEConfig, train
from torch.serialization import add_safe_globals
import wandb
from pips.misc.artifact import Artifact
from pips.grid_dataset import DatasetType
from pips.utils import generate_friendly_name
from pips.misc.acceleration_config import AccelerationConfig
import click
from rich import print
from enum import Enum
import yaml
import inspect
from typing import Dict, Any, Optional
import os


# Helper functions for YAML configuration
def save_config_to_yaml(config: Dict[str, Any], file_path: Path) -> None:
    """Save configuration dictionary to a YAML file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Configuration saved to: {file_path}")

def load_config_from_yaml(file_path: Path) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_default_param_values(func) -> Dict[str, Any]:
    """Extract default parameter values from a function."""
    signature = inspect.signature(func)
    defaults = {}
    
    for param_name, param in signature.parameters.items():
        # Skip *args, **kwargs, and parameters without defaults
        if param.default is not inspect.Parameter.empty and param_name != 'self':
            # Handle Typer Option objects
            if isinstance(param.default, typer.models.OptionInfo):
                defaults[param_name] = param.default.default
            else:
                defaults[param_name] = param.default
    
    return defaults

def get_current_config_from_func(func, **kwargs) -> Dict[str, Any]:
    """
    Get all configuration parameters from a function, including those with None defaults.
    
    Args:
        func: The function whose parameters we want to extract
        **kwargs: Any override values for specific parameters
        
    Returns:
        A dictionary of parameter names to their values
    """
    # Get the signature of the function
    signature = inspect.signature(func)
    
    # Extract all parameters (even those with None defaults)
    config = {}
    for param_name, param in signature.parameters.items():
        # Skip self parameter if present (for methods)
        if param_name == 'self':
            continue
            
        # Skip parameters without defaults (required parameters)
        if param.default is inspect.Parameter.empty:
            continue
            
        # Handle Typer Option objects
        if isinstance(param.default, typer.models.OptionInfo):
            # Store the default value, even if it's None
            config[param_name] = param.default.default
        else:
            # Store regular default values
            config[param_name] = param.default
    
    # Update with any provided override values
    for k, v in kwargs.items():
        if k in config:
            config[k] = v
    
    # Convert Enum values to strings and Path objects to strings
    for k, v in list(config.items()):
        if isinstance(v, Enum):
            config[k] = v.value
        elif isinstance(v, Path):
            config[k] = str(v)
    
    # Verify that we have all parameters
    func_params = set(p.name for p in signature.parameters.values() 
                     if p.name != 'self' and p.default is not inspect.Parameter.empty)
    config_params = set(config.keys())
    
    missing_params = func_params - config_params
    if missing_params:
        logger.warning(f"Missing parameters in configuration for {func.__name__}: {missing_params}")
    
    return config

def apply_config_from_yaml(yaml_config: Dict[str, Any], local_vars: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply configuration from a YAML file to local variables.
    
    Args:
        yaml_config: The loaded YAML configuration
        local_vars: Dictionary of local variables (usually from locals())
        
    Returns:
        Updated dictionary of local variables
    """
    # Create a copy of local_vars to update
    updated_vars = local_vars.copy()
    
    # Define mappings from nested keys to flat parameter names
    # This handles the known sections in our YAML structure
    section_mapping = {
        "model": "",  # model.n_dim -> n_dim
        "training": "",  # training.batch_size -> batch_size
        "schedules": {
            "types": "",  # schedules.types.tau_schedule_type -> tau_schedule_type
            "lr": "",  # schedules.lr.learning_rate -> learning_rate
            "tau": "",  # schedules.tau.tau -> tau
            "mask": "",  # schedules.mask.max_mask_pct -> max_mask_pct
            "residual": "",  # schedules.residual.residual_scaling -> residual_scaling
            "gumbel": "",  # schedules.gumbel.gumbel_noise_scale -> gumbel_noise_scale
        },
        "loss_weights": {
            "ce": "",  # loss_weights.ce.beta_ce -> beta_ce
            "kl": "",  # loss_weights.kl.beta_kl -> beta_kl
            "mi": "",  # loss_weights.mi.beta_mi -> beta_mi
            "tc": "",  # loss_weights.tc.beta_tc -> beta_tc
            "dwkl": "",  # loss_weights.dwkl.beta_dwkl -> beta_dwkl
            "diversity": {
                "entropy": "",  # loss_weights.diversity.entropy.beta_diversity_entropy -> beta_diversity_entropy
                "sample": "",  # loss_weights.diversity.sample.beta_diversity_sample -> beta_diversity_sample
                "position": "", # loss_weights.diversity.position.beta_diversity_position -> beta_diversity_position
                "usage": "",  # loss_weights.diversity.usage.beta_diversity_usage -> beta_diversity_usage
            }
        },
        "logging": "",  # logging.viz_interval -> viz_interval
        "acceleration": "",  # acceleration.device -> device
        "dataset": "",  # dataset.train_ds -> train_ds
        "project": "",  # project.project_name -> project_name
    }
    
    # Function to recursively extract values from nested config using our mapping
    def extract_values(config, mapping, prefix=""):
        result = {}
        for key, value in config.items():
            if key in mapping:
                # This is a section we have a mapping for
                if isinstance(mapping[key], dict):
                    # This is a nested section
                    if isinstance(value, dict):
                        result.update(extract_values(value, mapping[key], prefix))
                    else:
                        print(f"Warning: Expected a dict for {key}, got {type(value)}")
                else:
                    # This is a section that maps directly to param names
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            # Map nested parameter to a flat parameter name
                            param_name = subkey
                            result[param_name] = subvalue
                    else:
                        print(f"Warning: Expected a dict for {key}, got {type(value)}")
            else:
                # This is a direct parameter at this level
                param_name = key
                result[param_name] = value
        return result
    
    # Extract flattened values using our mapping
    flattened_values = extract_values(yaml_config, section_mapping)
    
    # Debug output
    print(f"Loaded {len(flattened_values)} parameters from YAML config")
    
    # Apply the flattened values to our local vars
    applied_count = 0
    for param_name, value in flattened_values.items():
        if param_name in updated_vars:
            # Special handling for run_name - only override if not explicitly set on CLI
            if param_name == "run_name" and updated_vars["run_name"] is not None:
                continue
                
            # Skip if it's a flag that was explicitly set via CLI
            if isinstance(value, bool):
                bool_flags = ["skip_codebook", "normalise_kq", "tc_relu", "debug", 
                             "compile_model", "lr_find", "wandb_logging", 
                             "use_exp_relaxed", "use_monte_carlo_kld", "permute_train"]
                
                if param_name in bool_flags:
                    # Check if this was explicitly set on CLI
                    original_value = updated_vars[param_name]
                    default_value = get_default_param_values(new).get(param_name)
                    
                    # If the original value is not the default, it was set on CLI
                    if original_value != default_value:
                        continue
            
            # Update the variable
            updated_vars[param_name] = value
            applied_count += 1
        else:
            print(f"Warning: Parameter '{param_name}' in config file is not a valid parameter for the command")
    
    print(f"Applied {applied_count} parameters from YAML config")
    
    return updated_vars

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

# Add this class definition somewhere before the CLI function
class InitMode(str, Enum):
    NORMAL = "normal"
    XAVIER = "xavier"

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
        "matmul_precision": typer.Option("high", "--matmul-precision", "--mp", help="Matmul precision.", case_sensitive=False, show_choices=True, click_type=click.Choice(list(AccelerationConfig.VALID_MATMUL_PRECISIONS))),
        "precision": typer.Option("bf16-true", "--precision", "--prec", help="Training precision. Note: will be overridden to '32-true' if device is cpu.", case_sensitive=False, show_choices=True, click_type=click.Choice(list(AccelerationConfig.VALID_PRECISIONS))),
        "device": typer.Option("auto", "--device", "--dev", help="Device to use. 'auto' selects cuda if available, else cpu.", case_sensitive=False, show_choices=True, click_type=click.Choice(list(AccelerationConfig.VALID_DEVICES))),
        
        # Project tracking options
        "project_name": typer.Option("dvae-training", "--project", "-p", help="Project name for experiment tracking"),
        "checkpoint_dir": typer.Option(Path("./runs"), "--checkpoint-dir", "-cd", help="Base directory for checkpoints"),
        # Debug option
        "debug": typer.Option(False, "--debug", "-D", help="Enable debug mode with reduced dataset and steps"),
        # Modify val_check_interval help text to include info about negative values
        "val_check_interval": typer.Option(
            1000, 
            "--val-check-interval", 
            "--vci", 
            help="Number of steps between validation checks. Set to negative value to disable validation."
        ),

        "lr_find": typer.Option(False, "--lr-find", "--lrf", help="Run learning rate finder instead of training"),

        # Add visualization interval option
        "viz_interval": typer.Option(
            1000, 
            "--viz-interval", 
            "--vi",
            help="Number of steps between visualizations and gradient logging"
        ),

        "wandb_logging": typer.Option(True, "--no-wandb", "--nw", help="Disable WandB logging", is_flag=True, flag_value=False),

    }

# Add export-config command
@dvae_app.command("export-config")
def export_config(
    output_path: Path = typer.Argument(..., help="Path to save the config YAML file"),
    # Include any parameters you want to customize for the export
    model_src: Optional[str] = get_common_options()["model_src_option"],
    run_name: Optional[str] = typer.Option(None, "--name", "-n", help="Run name to include in the config"),
):
    """Export the default configuration to a YAML file that can be later used with --config option."""
    # Get default configuration from the new command
    config = get_current_config_from_func(new)
    
    # Update with any values provided to this command
    if model_src is not None:
        config["model_src"] = model_src
    if run_name is not None:
        config["run_name"] = run_name
        
    # Organize into sections for better readability
    organized_config = {
        "model": {k: v for k, v in config.items() if k in [
            "n_dim", "n_head", "n_grid_layer", "n_latent_layer", "n_codes", "codebook_size",
            "dropout", "gamma", "pad_weight", "use_exp_relaxed", "use_monte_carlo_kld",
            "init_mode", "skip_codebook", "normalise_kq", "use_pure_logits_for_loss", "codebook_ema_update"
        ]},
        "training": {k: v for k, v in config.items() if k in [
            "batch_size", "weight_decay", "max_steps",
            "gradient_clip_val", "accumulate_grad_batches", "seed", "permute_train",
            "limit_training_samples", "lr_find"
        ]},
        "schedules": {
            "types": {k: v for k, v in config.items() if k.endswith("_schedule_type")},
            "lr": {k: v for k, v in config.items() if k in [
                "learning_rate", "lr_min", "warmup_steps_lr", "decay_steps_lr"
            ]},
            "tau": {k: v for k, v in config.items() if k.startswith("tau_") or k == "tau" or k.startswith("warmup_steps_tau") or k.startswith("transition_steps_tau")},
            "mask": {k: v for k, v in config.items() if k.startswith("mask_pct") or k == "max_mask_pct" or k.startswith("warmup_steps_mask") or k.startswith("transition_steps_mask")},
            "residual": {k: v for k, v in config.items() if k.startswith("residual_scaling") or k.startswith("warmup_steps_residual") or k.startswith("transition_steps_residual")},
            "gumbel": {k: v for k, v in config.items() if k.startswith("gumbel_noise") or k.startswith("warmup_steps_gumbel") or k.startswith("transition_steps_gumbel")},
        },
        "loss_weights": {
            "ce": {k: v for k, v in config.items() if k.startswith("beta_ce") or k.startswith("warmup_steps_beta_ce") or k.startswith("transition_steps_beta_ce")},
            "kl": {k: v for k, v in config.items() if k.startswith("beta_kl") or k.startswith("warmup_steps_beta_kl") or k.startswith("transition_steps_beta_kl")},
            "mi": {k: v for k, v in config.items() if k.startswith("beta_mi") or k.startswith("warmup_steps_beta_mi") or k.startswith("transition_steps_beta_mi")},
            "tc": {k: v for k, v in config.items() if k.startswith("beta_tc") or k.startswith("warmup_steps_beta_tc") or k.startswith("transition_steps_beta_tc") or k == "tc_relu"},
            "dwkl": {k: v for k, v in config.items() if k.startswith("beta_dwkl") or k.startswith("warmup_steps_beta_dwkl") or k.startswith("transition_steps_beta_dwkl")},
            "diversity": {
                "entropy": {k: v for k, v in config.items() if k.startswith("beta_diversity_entropy") or k.startswith("warmup_steps_beta_diversity_entropy") or k.startswith("transition_steps_beta_diversity_entropy")},
                "sample": {k: v for k, v in config.items() if k.startswith("beta_diversity_sample") or k.startswith("warmup_steps_beta_diversity_sample") or k.startswith("transition_steps_beta_diversity_sample")},
                "position": {k: v for k, v in config.items() if k.startswith("beta_diversity_position") or k.startswith("warmup_steps_beta_diversity_position") or k.startswith("transition_steps_beta_diversity_position")},
                "usage": {k: v for k, v in config.items() if k.startswith("beta_diversity_usage") or k.startswith("warmup_steps_beta_diversity_usage") or k.startswith("transition_steps_beta_diversity_usage")},
            }
        },
        "logging": {k: v for k, v in config.items() if k in [
            "viz_interval", "val_check_interval", "debug", "wandb_logging"
        ]},
        "acceleration": {k: v for k, v in config.items() if k in [
            "matmul_precision", "precision", "compile_model", "device"
        ]},
        "dataset": {k: v for k, v in config.items() if k in [
            "train_ds", "val_ds"
        ]},
        "project": {k: v for k, v in config.items() if k in [
            "project_name", "checkpoint_dir", "model_src", "run_name"
        ]},
    }
    
    # Check if we're missing any parameters in our organized config
    all_organized_keys = set()
    
    def collect_keys(config_dict):
        keys = set()
        for k, v in config_dict.items():
            if isinstance(v, dict):
                keys.update(collect_keys(v))
            else:
                keys.add(k)
        return keys
    
    all_organized_keys = collect_keys(organized_config)
    all_config_keys = set(config.keys())
    
    missing_keys = all_config_keys - all_organized_keys
    if missing_keys:
        # Add a 'misc' section for any parameters not captured in other sections
        organized_config["misc"] = {k: config[k] for k in missing_keys}
        logger.info(f"Added {len(missing_keys)} parameters to 'misc' section: {', '.join(missing_keys)}")
    
    # Save to file
    save_config_to_yaml(organized_config, output_path)

@dvae_app.command("train")
def new(
    # Add config file option
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to YAML configuration file"),
    
    # Project tracking
    project_name: str = get_common_options()["project_name"],
    run_name: str = typer.Option(None, "--name", "-n", help="Run name (generated if not specified)"),
    checkpoint_dir: Path = get_common_options()["checkpoint_dir"],
    
    # Model loading
    model_src: str = get_common_options()["model_src_option"],
    
    # Model architecture
    n_dim: int = typer.Option(256, "--n-dim", "-nd", help="Dimension of model embeddings"),
    n_head: int = typer.Option(8, "--n-head", "-nh", help="Number of attention heads"),
    n_grid_layer: int = typer.Option(4, "--n-grid-layer", "-ngl", help="Number of grid transformer layers"),
    n_latent_layer: int = typer.Option(4, "--n-latent-layer", "-nll", help="Number of latent transformer layers"),
    n_codes: int = typer.Option(128, "--n-codes", "-nc", help="Number of latent codes"),
    codebook_size: int = typer.Option(512, "--codebook-size", "-cb", help="Size of each codebook"),
    dropout: float = typer.Option(0.0, "--dropout", "-do", help="Dropout rate"),
    gamma: float = typer.Option(2.0, "--gamma", "-ga", help="Focal loss gamma parameter (default: 2.0)"),
    pad_weight: float = typer.Option(0.01, "--pad-weight", "-pw", help="Weight for pad token loss (default: 0.01 = 1% of normal weight)"),
    use_exp_relaxed: bool = typer.Option(False, "--exp-relaxed", "-er", help="Use exponentially relaxed Gumbel-Softmax"),
    use_monte_carlo_kld: bool = typer.Option(False, "--monte-carlo-kld", "-mck", help="Use Monte Carlo KLD estimation instead of approximate KLD"),
    init_mode: InitMode = typer.Option(InitMode.NORMAL, "--init-mode", "-im", help="Initialization mode for model weights", case_sensitive=False),
    skip_codebook: bool = typer.Option(False, "--skip-codebook", "-sc", help="Skip the codebook", is_flag=True, flag_value=True),
    normalise_kq: bool = typer.Option(False, "--normalise-kq", "-nkq", help="Normalise the keys and queries", is_flag=True, flag_value=True),
    use_pure_logits_for_loss: bool = typer.Option(False, "--use-pure-logits-for-loss", "-uplf", help="Use pure logits for loss", is_flag=True, flag_value=True),
    codebook_ema_update: bool = typer.Option(False, "--codebook-ema", "-cema", help="Update the codebook using EMA", is_flag=True, flag_value=True),
    
    # Training parameters
    batch_size: int = typer.Option(64, "--batch-size", "-bs", help="Training batch size"),
    learning_rate: float = typer.Option(4e-5, "--learning-rate", "-lr", help="Initial learning rate"),
    lr_min: float = typer.Option(None, "--lr-min", "-lrm", help="Minimum learning rate (defaults to 1% of learning rate if not specified)"),
    weight_decay: float = typer.Option(1e-4, "--weight-decay", "-wd", help="AdamW weight decay"),
    max_steps: int = typer.Option(100_000, "--max-steps", "-ms", help="Maximum training steps"),
    gradient_clip_val: float = typer.Option(1.0, "--gradient-clip-val", "-gcv", help="Gradient clipping value"),
    accumulate_grad_batches: int = typer.Option(1, "--accumulate-grad-batches", "-agb", help="Number of batches to accumulate gradients"),
    
    # Schedule types
    tau_schedule_type: str = typer.Option('cosine', "--tau-schedule-type", "-taus", help="Schedule type for tau transition (linear, cosine, exponential, threshold)"),
    beta_schedule_type: str = typer.Option('cosine', "--beta-schedule-type", "-bets", help="Schedule type for beta transitions (linear, cosine, exponential, threshold)"),
    mask_schedule_type: str = typer.Option('cosine', "--mask-schedule-type", "-mskt", help="Schedule type for mask percentage transition (linear, cosine, exponential, threshold)"),
    residual_scaling_schedule_type: str = typer.Option('cosine', "--residual-schedule-type", "-rsst", help="Schedule type for residual scaling transition (linear, cosine, exponential, threshold)"),
    gumbel_noise_scale_schedule_type: str = typer.Option('cosine', "--gumbel-noise-scale-schedule-type", "-gnst", help="Schedule type for gumbel noise scale transition (linear, cosine, exponential, threshold)"),
        
    # Learning rate warmup / decay steps
    warmup_steps_lr: int = typer.Option(100, "--warmup-steps-lr", "-wlr", help="Learning rate warmup steps"),
    decay_steps_lr: int = typer.Option(None, "--decay-steps-lr", "-dlr", help="Learning rate decay steps, if not specified, will be set to max_steps - warmup_steps_lr"),
    
    # Schedule parameters - tau
    tau_start: float = typer.Option(1.0, "--tau-start", "-ts", help="Starting temperature for Gumbel-Softmax."),
    tau: float = typer.Option(1.0, "--tau", "-ta", help="Final temperature for Gumbel-Softmax."),
    transition_steps_tau: int = typer.Option(5_000, "--transition-steps-tau", "-tsta", help="Steps to transition tau from initial to target value"),
    warmup_steps_tau: int = typer.Option(0, "--warmup-steps-tau", "-wst", help="Steps to wait before starting tau transition"),
    
    # Schedule parameters - residual scaling
    residual_scaling_start: float = typer.Option(1.0, "--residual-scaling-start", "-rss", help="Residual scaling start value"),
    residual_scaling: float = typer.Option(0.0, "--residual-scaling", "-rs", help="Residual scaling target value"),
    transition_steps_residual_scaling: int = typer.Option(10_000, "--transition-steps-residual", "-trs", help="Steps to transition residual scaling from initial to target value"),
    warmup_steps_residual_scaling: int = typer.Option(0, "--warmup-steps-residual", "-wrs", help="Steps to wait before starting residual scaling transition"),

    # Schedule parameters - gumbel noise scale
    gumbel_noise_scale_start: float = typer.Option(0.0, "--gumbel-noise-scale-start", "-gnss", help="Starting gumbel noise scale"),
    gumbel_noise_scale: float = typer.Option(0.0, "--gumbel-noise-scale", "-gns", help="Gumbel noise scale target value"),
    transition_steps_gumbel_noise_scale: int = typer.Option(10_000, "--transition-steps-gumbel-noise-scale", "-tgns", help="Steps to transition gumbel noise scale from initial to target value"),
    warmup_steps_gumbel_noise_scale: int = typer.Option(0, "--warmup-steps-gumbel-noise-scale", "-wgns", help="Steps to wait before starting gumbel noise scale transition"),

    # Schedule parameters - mask percentage
    mask_pct_start: float = typer.Option(0.0, "--mask-pct-start", "-mps", help="Starting masking percentage"),
    max_mask_pct: float = typer.Option(0.0, "--max-mask-pct", "-mmp", help="Maximum masking percentage during training"),
    transition_steps_mask_pct: int = typer.Option(5_000, "--transition-steps-mask", "-tsm", help="Steps to transition mask percentage from initial to target value"),
    warmup_steps_mask_pct: int = typer.Option(0, "--warmup-steps-mask", "-wsm", help="Steps to wait before starting mask percentage transition"),

    # Beta values and schedules for Cross-Entropy
    beta_ce_start: float = typer.Option(1.0, "--beta-ce-start", "-bces", help="Starting beta for cross-entropy loss"),
    beta_ce: float = typer.Option(1.0, "--beta-ce", "-bce", help="Target beta for cross-entropy loss"),
    transition_steps_beta_ce: int = typer.Option(10_000, "--transition-steps-ce", "-tsce", help="Steps to transition beta CE from initial to target value"),
    warmup_steps_beta_ce: int = typer.Option(0, "--warmup-steps-ce", "-wsce", help="Steps to wait before starting beta CE transition"),
   
    # beta div entropy
    beta_diversity_entropy_start: float = typer.Option(0.0, "--beta-div-entropy-start", "-bdes", help="Starting beta for diversity entropy loss"),
    beta_diversity_entropy: float = typer.Option(0.0, "--beta-div-entropy", "-bde", help="Target beta for diversity entropy loss"),
    transition_steps_beta_diversity_entropy: int = typer.Option(10_000, "--transition-steps-div-entropy", "-tsde", help="Steps to transition beta diversity entropy from initial to target value"),
    warmup_steps_beta_diversity_entropy: int = typer.Option(0, "--warmup-steps-div-entropy", "-wsde", help="Steps to wait before starting beta diversity entropy transition"),

    # beta div sample
    beta_diversity_sample_start: float = typer.Option(0.0, "--beta-div-sample-start", "-bds", help="Starting beta for diversity sample loss"),
    beta_diversity_sample: float = typer.Option(0.0, "--beta-div-sample", "-bds", help="Target beta for diversity sample loss"),
    transition_steps_beta_diversity_sample: int = typer.Option(10_000, "--transition-steps-div-sample", "-tsds", help="Steps to transition beta diversity sample from initial to target value"),
    warmup_steps_beta_diversity_sample: int = typer.Option(0, "--warmup-steps-div-sample", "-wsds", help="Steps to wait before starting beta diversity sample transition"),

    # beta div position
    beta_diversity_position_start: float = typer.Option(0.0, "--beta-div-position-start", "-bdp", help="Starting beta for diversity position loss"),
    beta_diversity_position: float = typer.Option(0.0, "--beta-div-position", "-bdp", help="Target beta for diversity position loss"),
    transition_steps_beta_diversity_position: int = typer.Option(10_000, "--transition-steps-div-position", "-tsdp", help="Steps to transition beta diversity position from initial to target value"),
    warmup_steps_beta_diversity_position: int = typer.Option(0, "--warmup-steps-div-position", "-wsdp", help="Steps to wait before starting beta diversity position transition"),

    # beta div usage
    beta_diversity_usage_start: float = typer.Option(0.0, "--beta-div-usage-start", "-bdu", help="Starting beta for diversity usage loss"),
    beta_diversity_usage: float = typer.Option(0.0, "--beta-div-usage", "-bdu", help="Target beta for diversity usage loss"),
    transition_steps_beta_diversity_usage: int = typer.Option(10_000, "--transition-steps-div-usage", "-tsdu", help="Steps to transition beta diversity usage from initial to target value"),
    warmup_steps_beta_diversity_usage: int = typer.Option(0, "--warmup-steps-div-usage", "-wsdu", help="Steps to wait before starting beta diversity usage transition"),    

    # Beta values and schedules for KL Divergence
    beta_kl_start: float = typer.Option(0.0, "--beta-kl-start", "-bks", help="Starting beta for KL loss"),
    beta_kl: float = typer.Option(0.0, "--beta-kl", "-bk", help="Target beta for KL loss"),
    transition_steps_beta_kl: int = typer.Option(10_000, "--transition-steps-kl", "-tsk", help="Steps to transition beta KL from initial to target value"),
    warmup_steps_beta_kl: int = typer.Option(0, "--warmup-steps-kl", "-wsk", help="Steps to wait before starting beta KL transition"),
    
    # Beta values and schedules for Mutual Information
    beta_mi_start: float = typer.Option(0.0, "--beta-mi-start", "-bmis", help="Starting beta for mutual information loss"),
    beta_mi: float = typer.Option(0.0, "--beta-mi", "-bmi", help="Target beta for mutual information loss"),
    transition_steps_beta_mi: int = typer.Option(10_000, "--transition-steps-mi", "-tsmi", help="Steps to transition beta MI from initial to target value"),
    warmup_steps_beta_mi: int = typer.Option(0, "--warmup-steps-mi", "-wsmi", help="Steps to wait before starting beta MI transition"),
    
    # Beta values and schedules for Total Correlation
    beta_tc_start: float = typer.Option(0.0, "--beta-tc-start", "-btcs", help="Starting beta for total correlation loss"),
    beta_tc: float = typer.Option(0.0, "--beta-tc", "-btc", help="Target beta for total correlation loss"),
    transition_steps_beta_tc: int = typer.Option(10_000, "--transition-steps-tc", "-tstc", help="Steps to transition beta TC from initial to target value"),
    warmup_steps_beta_tc: int = typer.Option(0, "--warmup-steps-tc", "-wstc", help="Steps to wait before starting beta TC transition"),
    
    # Beta values and schedules for Dimension-wise KL
    beta_dwkl_start: float = typer.Option(0.0, "--beta-dwkl-start", "-bdwks", help="Starting beta for dimension-wise KL loss"),
    beta_dwkl: float = typer.Option(0.0, "--beta-dwkl", "-bdwk", help="Target beta for dimension-wise KL loss"),
    transition_steps_beta_dwkl: int = typer.Option(10_000, "--transition-steps-dwkl", "-tsdwk", help="Steps to transition beta DWKL from initial to target value"),
    warmup_steps_beta_dwkl: int = typer.Option(0, "--warmup-steps-dwkl", "-wsdwk", help="Steps to wait before starting beta DWKL transition"),
    
    # TC ReLU option
    tc_relu: bool = typer.Option(False, "--tc-relu", "-tcr", help="Apply ReLU to TC loss", is_flag=True, flag_value=True),
    
    # Training data options
    train_ds: DatasetType = typer.Option(DatasetType.TRAIN, "--train-ds", "-tds", help="Training dataset type. Default is TRAIN.", case_sensitive=False, show_choices=True),
    val_ds: DatasetType = typer.Option(DatasetType.VAL, "--val-ds", "-vds", help="Validation dataset type. Default is VAL.", case_sensitive=False, show_choices=True),
    seed: int = typer.Option(None, "--seed", "-sd", help="Random seed for reproducibility. If not provided, a random seed will be generated."),
    limit_training_samples: int = typer.Option(None, "--limit-samples", "-ls", help="Limit the number of training samples. None means use all samples."),
    permute_train: bool = typer.Option(True, "--no-permute", "-np", help="Permute the training data. Default is True.", is_flag=True, flag_value=False),

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
    
    # Get all parameters from locals, exclude built-in and temporary variables
    params = {k: v for k, v in locals().items() 
              if not k.startswith('_') and k != 'config_file' and k != 'self'}
    
    # If config file is provided, update parameters from it
    if config_file is not None:
        try:
            print(f"Loading configuration from {config_file}")
            yaml_config = load_config_from_yaml(config_file)
            
            # Apply configuration to parameters
            # First create a dictionary of locals to pass to apply_config_from_yaml
            locals_dict = locals()
            updated_locals = apply_config_from_yaml(yaml_config, locals_dict)
            
            # Update our params dictionary with values from updated_locals
            for key in params.keys():
                if key in updated_locals:
                    if params[key] != updated_locals[key]:
                        print(f"Updated param from config: {key} = {updated_locals[key]}")
                        params[key] = updated_locals[key]

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            import traceback
            traceback.print_exc()
            raise typer.Exit(code=1)
    
    # Create acceleration config (will validate and resolve settings)
    acceleration = AccelerationConfig(
        device=params["device"],
        precision=params["precision"],
        matmul_precision=params["matmul_precision"],
        compile_model=params["compile_model"]
    )

    # Generate run name if not provided
    if params["run_name"] is None:
        params["run_name"] = generate_friendly_name()

    # Create GridDVAEConfig
    model_config = GridDVAEConfig(
        n_dim=params["n_dim"],
        n_head=params["n_head"],
        n_grid_layer=params["n_grid_layer"],
        n_latent_layer=params["n_latent_layer"],
        n_codes=params["n_codes"],
        codebook_size=params["codebook_size"],
        dropout=params["dropout"],
        gamma=params["gamma"],
        n_vocab=16,  # Fixed for grid world
        max_grid_height=32,  # Fixed for grid world
        max_grid_width=32,  # Fixed for grid world
        pad_weight=params["pad_weight"],
        use_exp_relaxed=params["use_exp_relaxed"],
        use_monte_carlo_kld=params["use_monte_carlo_kld"],
        init_mode=params["init_mode"],
        skip_codebook=params["skip_codebook"],
        normalise_kq=params["normalise_kq"],
        use_pure_logits_for_loss=params["use_pure_logits_for_loss"],
        codebook_ema_update=params["codebook_ema_update"]
    )
    
    # Create ExperimentConfig
    experiment_config = ExperimentConfig(
        model_config=model_config,
        seed=params["seed"],

        # Tau
        tau_start=params["tau_start"],
        tau=params["tau"],
        transition_steps_tau=params["transition_steps_tau"],
        warmup_steps_tau=params["warmup_steps_tau"],

        # CE
        beta_ce_start=params["beta_ce_start"],
        beta_ce=params["beta_ce"],
        transition_steps_beta_ce=params["transition_steps_beta_ce"],
        warmup_steps_beta_ce=params["warmup_steps_beta_ce"],

        # Diversity Losses
        ## Entropy
        beta_diversity_entropy_start=params["beta_diversity_entropy_start"],
        beta_diversity_entropy=params["beta_diversity_entropy"],
        transition_steps_beta_diversity_entropy=params["transition_steps_beta_diversity_entropy"],
        warmup_steps_beta_diversity_entropy=params["warmup_steps_beta_diversity_entropy"],

        ## Sample
        beta_diversity_sample_start=params["beta_diversity_sample_start"],
        beta_diversity_sample=params["beta_diversity_sample"],
        transition_steps_beta_diversity_sample=params["transition_steps_beta_diversity_sample"],
        warmup_steps_beta_diversity_sample=params["warmup_steps_beta_diversity_sample"],
        
        ## Position
        beta_diversity_position_start=params["beta_diversity_position_start"],
        beta_diversity_position=params["beta_diversity_position"],
        transition_steps_beta_diversity_position=params["transition_steps_beta_diversity_position"],
        warmup_steps_beta_diversity_position=params["warmup_steps_beta_diversity_position"],

        ## Usage
        beta_diversity_usage_start=params["beta_diversity_usage_start"],
        beta_diversity_usage=params["beta_diversity_usage"],
        transition_steps_beta_diversity_usage=params["transition_steps_beta_diversity_usage"],
        warmup_steps_beta_diversity_usage=params["warmup_steps_beta_diversity_usage"],

        # KL Losses
        ## KL
        beta_kl_start=params["beta_kl_start"],
        beta_kl=params["beta_kl"],
        transition_steps_beta_kl=params["transition_steps_beta_kl"],
        warmup_steps_beta_kl=params["warmup_steps_beta_kl"],

        ## MI
        beta_mi_start=params["beta_mi_start"],
        beta_mi=params["beta_mi"],
        transition_steps_beta_mi=params["transition_steps_beta_mi"],
        warmup_steps_beta_mi=params["warmup_steps_beta_mi"],

        ## TC
        beta_tc_start=params["beta_tc_start"],
        beta_tc=params["beta_tc"],
        transition_steps_beta_tc=params["transition_steps_beta_tc"],
        warmup_steps_beta_tc=params["warmup_steps_beta_tc"],

        ## DWKL
        beta_dwkl_start=params["beta_dwkl_start"],
        beta_dwkl=params["beta_dwkl"],
        transition_steps_beta_dwkl=params["transition_steps_beta_dwkl"],
        warmup_steps_beta_dwkl=params["warmup_steps_beta_dwkl"],

        # Mask
        mask_pct_start=params["mask_pct_start"],
        max_mask_pct=params["max_mask_pct"],
        transition_steps_mask_pct=params["transition_steps_mask_pct"],
        warmup_steps_mask_pct=params["warmup_steps_mask_pct"],

        # Residual scaling
        residual_scaling_start=params["residual_scaling_start"],
        residual_scaling=params["residual_scaling"],
        transition_steps_residual_scaling=params["transition_steps_residual_scaling"],
        warmup_steps_residual_scaling=params["warmup_steps_residual_scaling"],

        # Gumbel noise scale
        gumbel_noise_scale_start=params["gumbel_noise_scale_start"],
        gumbel_noise_scale=params["gumbel_noise_scale"],
        transition_steps_gumbel_noise_scale=params["transition_steps_gumbel_noise_scale"],
        warmup_steps_gumbel_noise_scale=params["warmup_steps_gumbel_noise_scale"],
        
        # Schedule parameters
        tau_schedule_type=params["tau_schedule_type"],
        beta_schedule_type=params["beta_schedule_type"],
        mask_schedule_type=params["mask_schedule_type"],
        residual_scaling_schedule_type=params["residual_scaling_schedule_type"],
        gumbel_noise_scale_schedule_type=params["gumbel_noise_scale_schedule_type"],
        
        # Schedule timing parameters
        learning_rate=params["learning_rate"],
        lr_min=params["lr_min"] if params["lr_min"] is not None else params["learning_rate"] * 0.01,
        warmup_steps_lr=params["warmup_steps_lr"],
        decay_steps_lr=params["decay_steps_lr"],

        # Training parameters
        batch_size=params["batch_size"],
        weight_decay=params["weight_decay"],
        max_steps=params["max_steps"],
        gradient_clip_val=params["gradient_clip_val"],
        accumulate_grad_batches=params["accumulate_grad_batches"],
        
        # Dataset parameters
        train_ds=params["train_ds"],
        val_ds=params["val_ds"],
        limit_training_samples=params["limit_training_samples"],
        permute_train=params["permute_train"],
        
        # Other parameters
        model_src=params["model_src"],
        tc_relu=params["tc_relu"]
    )

    print(experiment_config.to_dict())
    # Start training with the resolved settings
    train(
        experiment_config=experiment_config,
        run_name=params["run_name"],
        project_name=params["project_name"],
        checkpoint_dir=params["checkpoint_dir"],
        debug_mode=params["debug"],
        lr_find=params["lr_find"],
        acceleration=acceleration,
        val_check_interval=params["val_check_interval"],
        visualization_interval=params["viz_interval"],
        grad_log_interval=params["viz_interval"],
        wandb_logging=params["wandb_logging"]
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
    batch_size: int = typer.Option(None, "--batch-size", "-bs", help="Override training batch size"),
    learning_rate: float = typer.Option(None, "--learning-rate", "-lr", help="Override learning rate"),
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