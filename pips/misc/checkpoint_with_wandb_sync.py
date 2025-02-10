import os
import re
import logging
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from .artifact import Artifact

class ModelCheckpointWithWandbSync(ModelCheckpoint):
    """ModelCheckpoint that syncs only current local checkpoints as WandB artifacts.
    
    It tracks only the checkpoints this callback saved (via _save_checkpoint) and when syncing:
      - Deletes any remote artifact whose file no longer exists locally.
      - Uploads new artifacts only for checkpoints recorded by this callback.
      - When an artifact already exists, it updates its best-* alias if needed, but leaves the step alias unchanged.
    """
    def __init__(self, wandb_model_suffix="best", wandb_verbose=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_checkpoints = set()  # Track filenames saved by this callback.
        self._wandb_model_suffix = wandb_model_suffix
        self._artifact_manager = None
        self.wandb_verbose = wandb_verbose
        self.logger = logging.getLogger(__name__)

    def _get_artifact_manager(self, trainer):
        if self._artifact_manager is None:
            self._artifact_manager = Artifact(
                entity=trainer.logger.experiment.entity,
                project_name=trainer.logger.experiment.project,
                run_name=trainer.logger.experiment.id,
                verbose=self.wandb_verbose
            )
        return self._artifact_manager

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        filename = os.path.basename(filepath)
        self._saved_checkpoints.add(filename)
        self._sync_wandb_artifacts(trainer)

    def _sync_wandb_artifacts(self, trainer):
        checkpoint_dir = Path(self.dirpath)
        # All local checkpoint files (excluding symlinks like "last.ckpt")
        local_ckpts_all = {
            p.name: p for p in checkpoint_dir.glob("*.ckpt")
            if p.is_file() and not p.is_symlink()
        }
        # But only consider for upload those that were saved by this callback.
        local_ckpts_saved = {
            name: path for name, path in local_ckpts_all.items()
            if name in self._saved_checkpoints
        }
        
        try:
            artifact_manager = self._get_artifact_manager(trainer)
            artifacts = artifact_manager.get_artifacts(self._wandb_model_suffix)
            logged_artifacts = {
                artifact.metadata.get("filepath"): artifact 
                for artifact in artifacts 
                if artifact.metadata.get("filepath")
            }

            # Delete remote artifacts for checkpoint files that no longer exist locally.
            for fname, artifact in list(logged_artifacts.items()):
                if fname not in local_ckpts_all:
                    artifact_manager.delete_artifact(artifact)
                    logged_artifacts.pop(fname)

            # Prepare best_k_models using absolute paths.
            best_k_models_abs = {os.path.abspath(str(k)): v for k, v in self.best_k_models.items()}

            # Iterate over this callback's saved checkpoints in ascending order
            for fname, path_obj in sorted(
                local_ckpts_saved.items(),
                key=lambda item: int(re.search(r'step(\d+)', item[0]).group(1))
                        if re.search(r'step(\d+)', item[0]) else 0
            ):
                full_path = os.path.abspath(str(path_obj))
                # Compute best alias if this checkpoint is in best_k_models.
                best_alias = None
                if full_path in best_k_models_abs:
                    sorted_best = sorted(
                        best_k_models_abs.items(),
                        key=lambda x: x[1],
                        reverse=(self.mode == "max")
                    )
                    rank = next(i for i, (p, _) in enumerate(sorted_best) if p == full_path) + 1
                    best_alias = "best" if rank == 1 else f"best-{rank}"

                if fname in logged_artifacts:
                    # Update the artifact's best alias if needed
                    artifact = logged_artifacts[fname]
                    updated_aliases = [a for a in artifact.aliases if not a.startswith("best")]
                    if best_alias:
                        updated_aliases.append(best_alias)
                    artifact_manager.update_artifact_aliases(artifact, updated_aliases)
                else:
                    # Upload new artifact
                    aliases = []
                    if best_alias:
                        aliases.append(best_alias)
                    match = re.search(r'step(\d+)', fname)
                    if match:
                        aliases.append(f"step-{match.group(1)}")
                    
                    artifact_manager.create_and_log_artifact(
                        name=f"model-{trainer.logger.experiment.id}-{self._wandb_model_suffix}",
                        file_path=path_obj,
                        aliases=aliases,
                        metadata={"filepath": fname},
                        run=trainer.logger.experiment
                    )
        except Exception as e:
            self.logger.error(f"Error syncing wandb artifacts: {e}")
            raise