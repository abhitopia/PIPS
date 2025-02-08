import os
import re
import wandb
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint

class ModelCheckpointWithWandbSync(ModelCheckpoint):
    """ModelCheckpoint that syncs only current local checkpoints as WandB artifacts.
    
    It tracks only the checkpoints this callback saved (via _save_checkpoint) and when syncing:
      - Deletes any remote artifact whose file no longer exists locally.
      - Uploads new artifacts only for checkpoints recorded by this callback.
      - When an artifact already exists, it updates its best-* alias if needed, but leaves the step alias unchanged.
    """
    def __init__(self, wandb_model_suffix="best", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_checkpoints = set()  # Track filenames saved by this callback.
        self._wandb_model_suffix = wandb_model_suffix

    def _save_checkpoint(self, trainer, filepath):
        print(f"Called {self._wandb_model_suffix} callback for {filepath}")
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
            api = wandb.Api()
            run_path = f"{trainer.logger.experiment.entity}/{trainer.logger.experiment.project}/{trainer.logger.experiment.id}"
            run_obj = api.run(run_path)
            logged_artifacts = {}
            for artifact in run_obj.logged_artifacts():
                if artifact.type == "model" and (fname := artifact.metadata.get("filepath")):
                    # Only consider artifacts that were uploaded by this callback (using our suffix)
                    if self._wandb_model_suffix in artifact.name:
                        logged_artifacts[fname] = artifact

            # Delete remote artifacts for checkpoint files that no longer exist locally.
            for fname, artifact in list(logged_artifacts.items()):
                if fname not in local_ckpts_all:
                    if not getattr(artifact, "id", None):
                        print(f"Artifact for {fname} not published, skipping deletion")
                        continue
                    try:
                        artifact.aliases = []
                        artifact.save()
                        artifact.delete()
                        print(f"Deleted wandb artifact for {fname}")
                        logged_artifacts.pop(fname)
                    except Exception as e:
                        print(f"Error deleting artifact for {fname}: {e}")

            # Prepare best_k_models using absolute paths.
            best_k_models_abs = {os.path.abspath(str(k)): v for k, v in self.best_k_models.items()}

            # Iterate over this callback's saved checkpoints in ascending order (by extracted step number).
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
                    # Update the artifact's best alias if needed, but leave its step alias intact.
                    artifact = logged_artifacts[fname]
                    updated_aliases = [a for a in artifact.aliases if not a.startswith("best")]
                    if best_alias:
                        updated_aliases.append(best_alias)
                    if set(updated_aliases) != set(artifact.aliases):
                        artifact.aliases = updated_aliases
                        try:
                            artifact.save()
                            print(f"Updated artifact {fname} aliases to {artifact.aliases}")
                        except Exception as e:
                            print(f"Error updating artifact {fname}: {e}")
                else:
                    # If the artifact doesn't exist remotely, upload it.
                    aliases = []
                    if best_alias:
                        aliases.append(best_alias)
                    match = re.search(r'step(\d+)', fname)
                    if match:
                        aliases.append(f"step-{match.group(1)}")
                    artifact = wandb.Artifact(
                        name=f"model-{trainer.logger.experiment.id}-{self._wandb_model_suffix}",
                        type="model",
                        metadata={"filepath": fname}
                    )
                    artifact.add_file(str(path_obj))
                    trainer.logger.experiment.log_artifact(artifact, aliases=aliases)
                    print(f"Uploaded wandb artifact for {fname} with aliases {aliases}")
        except Exception as e:
            print("Error syncing wandb artifacts:", e)