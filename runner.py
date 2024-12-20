import yaml
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import Logger


# saves a copy of the cli config to the logger (wandb)
class SaveConfigRemoteCallback(SaveConfigCallback):
    def save_config(self, trainer, pl_module, stage):
        if isinstance(trainer.logger, Logger):
            # convert to and from yaml
            config = self.parser.dump(self.config, skip_none=False)
            trainer.logger.log_hyperparams({"config": yaml.safe_load(config)})


def cli_main():

    cli = LightningCLI(  # noqa: F841
        save_config_callback=SaveConfigRemoteCallback,
        save_config_kwargs={"overwrite": True},
        auto_configure_optimizers=False,
        subclass_mode_data=True,
    )


if __name__ == "__main__":
    cli_main()
