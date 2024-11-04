import os

from lightning import Callback
from lightning.pytorch.loggers import WandbLogger


class SavePyTorchProfilerToWandbCallback(Callback):

    def on_train_start(self, trainer, pl_module):
        if isinstance(trainer.logger, WandbLogger):
            # override dirpath to make it specific to this wandb run
            trainer.profiler.dirpath = os.path.join(
                trainer.profiler.dirpath, trainer.logger.experiment.id
            )

    def on_train_end(self, trainer, pl_module):
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.save(
                glob_str=f"{trainer.profiler.dirpath}/*.pt.trace.json",
                base_path=trainer.profiler.dirpath,
            )
