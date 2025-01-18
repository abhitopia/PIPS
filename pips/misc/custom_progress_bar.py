import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing_extensions import override

class CustomRichProgressBar(RichProgressBar):
    # -----------------------------
    # Override refresh to catch KeyError in rendering.
    # -----------------------------
    @override
    def refresh(self) -> None:
        try:
            super().refresh()
        except KeyError:
            if self.train_progress_bar_id is not None:
                self._current_task_id = self.train_progress_bar_id
            elif self.val_progress_bar_id is not None:
                self._current_task_id = self.val_progress_bar_id
            else:
                if self.progress and len(self.progress.tasks) > 0:
                    self._current_task_id = 0
            super().refresh()

    # -----------------------------
    # Custom Training Progress Bar
    # -----------------------------
    @override
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._init_progress(trainer)
        if self.progress is not None:
            self.train_progress_bar_id = self.progress.add_task(
                "[green]Training", total=trainer.max_steps
            )
            self._current_task_id = self.train_progress_bar_id
        self.refresh()

    @override
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Suppress the default epoch progress bar creation.
        pass

    @override
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: any,
        batch_idx: int,
    ) -> None:
        if self.progress is not None and self.train_progress_bar_id is not None:
            self.progress.update(
                self.train_progress_bar_id, completed=trainer.global_step
            )
        self.refresh()

    # -----------------------------
    # Custom Validation Progress Bar
    # -----------------------------
    @override
    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._init_progress(trainer)
        # Set the evaluation dataloader index so total_val_batches_current_dataloader works.
        self._current_eval_dataloader_idx = 0
        if not trainer.sanity_checking and self.progress is not None:
            total_val_batches = self.total_val_batches_current_dataloader
            self.val_progress_bar_id = self.progress.add_task(
                f"[blue]Validation Epoch {trainer.current_epoch} (0/{total_val_batches})",
                total=total_val_batches,
                visible=True
            )
            self._current_task_id = self.val_progress_bar_id
        self.refresh()

    @override
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.is_disabled:
            return
        if not trainer.sanity_checking and self.val_progress_bar_id is not None:
            total_batches = self.total_val_batches_current_dataloader
            new_description = f"[blue]Validation Epoch {trainer.current_epoch} ({batch_idx+1}/{total_batches})"
            self.progress.update(
                self.val_progress_bar_id,
                completed=batch_idx + 1,
                description=new_description
            )
        self.refresh()

    @override
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.sanity_checking and self.val_progress_bar_id is not None:
            self.progress.update(self.val_progress_bar_id, visible=False)
        self.reset_dataloader_idx_tracker()

    # -----------------------------
    # Custom Sanity-Check Progress Bar
    # -----------------------------
    @override
    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Initialize progress and create a dummy sanity check task.
        self._init_progress(trainer)
        if self.progress is not None:
            # Use total=1 (or adjust if needed) for sanity check.
            self.val_sanity_progress_bar_id = self.progress.add_task(
                "[blue]Sanity Check", total=1, visible=False
            )
            self._current_task_id = self.val_sanity_progress_bar_id
        self.refresh()

    @override
    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Hide the sanity-check progress bar.
        if self.progress is not None and self.val_sanity_progress_bar_id is not None:
            self.progress.update(self.val_sanity_progress_bar_id, advance=0, visible=False)
        self.refresh()
