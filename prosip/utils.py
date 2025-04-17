from pathlib import Path
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import wandb
import pytorch_lightning as pl
from pips.misc.artifact import Artifact


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


def normalize_state_dict(current_state_dict, ckpt_state_dict):
    # 1. Determine if the checkpoint is compiled by checking its state dict structure
    is_checkpoint_compiled = any("_orig_mod" in key for key in ckpt_state_dict.keys())
    # 2. Determine if our current model is compiled by checking its state dict structure
    is_current_model_compiled = any("_orig_mod" in key for key in current_state_dict.keys())
    
    # 4. Critical fix - ensure checkpoint state dict perfectly matches current model structure
    # but without codebook parameters when there's a mismatch
    new_state_dict = {}
    
    # Get the keys that should be in the final state dict from the current model
    for target_key in current_state_dict.keys():           
        # Find the corresponding key in the checkpoint state dict
        source_key = None
        if is_current_model_compiled and not is_checkpoint_compiled:
            # Current is compiled, checkpoint isn't
            possible_source_key = target_key.replace("model._orig_mod.", "model.")
            if possible_source_key in ckpt_state_dict:
                source_key = possible_source_key
        elif not is_current_model_compiled and is_checkpoint_compiled:
            # Current isn't compiled, checkpoint is
            possible_source_key = target_key.replace("model.", "model._orig_mod.")
            if possible_source_key in ckpt_state_dict:
                source_key = possible_source_key
        else:
            # Both are the same (compiled or not)
            if target_key in ckpt_state_dict:
                source_key = target_key
        
        # If we found a matching source key, copy the value with data type conversion if needed
        if source_key is not None:
            value = ckpt_state_dict[source_key]
            
            # Handle data type conversion if needed
            if hasattr(value, 'dtype') and value.dtype != torch.float32 and value.dtype != torch.int64:
                try:
                    value = value.to(torch.float32)
                except Exception as e:
                    print(f"WARNING: Failed to convert {source_key} from {value.dtype} to float32: {e}")
            
            new_state_dict[target_key] = value

    return new_state_dict



def get_activation_fn(activation: str, return_module: bool = False):
    """Convert activation function name to the corresponding PyTorch implementation.
    
    Args:
        activation: String name of the activation function
        
    Returns:
        PyTorch activation function
        
    Raises:
        ValueError: If activation name is not recognized
    """
    activation = activation.lower()
    if activation == "relu":
        return F.relu if not return_module else nn.ReLU
    elif activation == "gelu":
        return F.gelu if not return_module else nn.GELU
    elif activation == "silu" or activation == "swish":
        return F.silu if not return_module else nn.SiLU
    elif activation == "tanh":
        return F.tanh if not return_module else nn.Tanh
    elif activation == "sigmoid":
        return F.sigmoid if not return_module else nn.Sigmoid
    else:
        raise ValueError(f"Unknown activation function: {activation}")
