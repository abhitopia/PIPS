import pytorch_lightning as pl
import torch

class GradientCheckCallback(pl.Callback):
    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Iterate over all parameters in the model
        for name, param in pl_module.named_parameters():
            # Only check parameters that have accumulated gradients
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"[GradientCheck] Detected NaN in gradients for parameter: {name}")
                elif torch.isinf(param.grad).any():
                    print(f"[GradientCheck] Detected Inf in gradients for parameter: {name}")
                # else:
                #     print(f"[GradientCheck] All good in gradients for parameter: {name}")
