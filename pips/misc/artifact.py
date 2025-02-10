from pathlib import Path
import sys
from typing import List, Optional
import wandb
from rich.console import Console
from rich.table import Table
import re

class Artifact:
    """Manages W&B model artifacts for checkpoint management."""

    def __init__(self, entity: str, project_name: str, run_name: str):
        self.entity = entity
        self.project_name = project_name
        self.run_name = run_name
        self.run_path = f"{entity}/{project_name}/{run_name}"
        self.console = Console()
        self._api = wandb.Api()
        self._run = None

    @property
    def run(self):
        """Lazy loading of run object."""
        if self._run is None:
            try:
                self._run = self._api.run(self.run_path)
            except Exception as e:
                raise ValueError(f"Run '{self.run_name}' not found in project '{self.project_name}': {e}")
        return self._run

    def get_artifacts(self, category: str) -> List[wandb.Artifact]:
        """Get all artifacts for a specific category (best/backup)."""
        if category not in ["best", "backup"]:
            raise ValueError("Category must be either 'best' or 'backup'")

        artifacts = []
        for artifact in self.run.logged_artifacts():
            if artifact.type == "model" and f"-{category}" in artifact.name:
                artifacts.append(artifact)
        return artifacts

    def display_checkpoints_table(self, artifacts: List[wandb.Artifact]) -> None:
        """Display available checkpoints in a formatted table."""
        table = Table(
            title=f"\nCheckpoints for run '{self.run_name}'",
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Checkpoint", style="dim")
        table.add_column("Step", justify="right", style="green")
        table.add_column("Aliases", style="yellow")

        for artifact in artifacts:
            filename = artifact.metadata.get('filepath', 'unknown')
            step_match = re.match(r'.*step(\d+)-.*\.ckpt', filename)
            
            if step_match:
                step = step_match.group(1)
                table.add_row(
                    filename,
                    str(int(step)),
                    ", ".join(artifact.aliases)
                )
            else:
                table.add_row(
                    filename,
                    "N/A",
                    ", ".join(artifact.aliases)
                )

        self.console.print(table)

    def find_matching_artifact(
        self, 
        artifacts: List[wandb.Artifact], 
        step: Optional[int] = None, 
        alias: Optional[str] = None
    ) -> wandb.Artifact:
        """Find artifact matching either step or alias."""
        alias_to_find = f"step-{step:07d}" if step is not None else alias

        for artifact in artifacts:
            if alias_to_find in artifact.aliases:
                return artifact

        # If no matching artifact found, show available checkpoints and raise error
        print("\nNo matching checkpoint found. Available checkpoints:")
        self.display_checkpoints_table(artifacts)
        sys.exit(1)
        # raise ValueError(f"No checkpoint found with alias '{alias_to_find}'")

    def ensure_local_checkpoint(
        self, 
        artifact: wandb.Artifact, 
        checkpoint_dir: Path
    ) -> Path:
        """Ensure checkpoint exists locally, downloading if necessary."""
        if not artifact.metadata.get("filepath"):
            raise ValueError("Artifact metadata missing filepath information")
        
        checkpoint_filename = artifact.metadata["filepath"]
        local_checkpoint_path = checkpoint_dir / self.project_name / self.run_name / "checkpoints" / checkpoint_filename
        
        if not local_checkpoint_path.exists():
            print(f"Checkpoint not found locally, downloading from W&B...")
            
            local_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            artifact_dir = artifact.download()
            downloaded_checkpoint = next(Path(artifact_dir).glob("*.ckpt"))
            
            downloaded_checkpoint.rename(local_checkpoint_path)
            print(f"Downloaded checkpoint to: {local_checkpoint_path}")
        else:
            print(f"Using existing local checkpoint: {local_checkpoint_path}")

        return local_checkpoint_path 