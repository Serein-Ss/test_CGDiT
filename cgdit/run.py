from pathlib import Path
from typing import List
import sys

sys.path.append('.')
import hydra
import numpy as np
import torch
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from cgdit.common.utils import log_hyperparameters, PROJECT_ROOT

import wandb


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if "progress_bar" in cfg.logging:
        hydra.utils.log.info("Adding callback <TQDMProgressBar>")
        callbacks.append(
            hydra.utils.instantiate(cfg.logging.progress_bar)
        )

    if "model_summary" in cfg.logging:
        hydra.utils.log.info("Adding callback <ModelSummary>")
        callbacks.append(
            hydra.utils.instantiate(cfg.logging.model_summary)
        )

    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(HydraConfig.get().run.dir),
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
                save_last=cfg.train.model_checkpoints.save_last,
            )
        )

    return callbacks


def run(cfg: DictConfig) -> None:
    """
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /conf
    """
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run=}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.pl_trainer.accelerator = "cpu"
        if "devices" in cfg.train.pl_trainer:
            del cfg.train.pl_trainer.devices
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0

        # Switch wandb mode to offline to prevent online logging
        cfg.logging.wandb.mode = "offline"

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    finetune_path = cfg.train.get('finetune_from_ckpt', None)
    if finetune_path is not None:
        hydra.utils.log.info(f"======== FINE-TUNING MODE ========")
        hydra.utils.log.info(f"Loading pretrained weights from: {finetune_path}")
        checkpoint = torch.load(finetune_path, map_location='cpu', weights_only=False)

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        load_result = model.load_state_dict(state_dict, strict=False)
        hydra.utils.log.info(f"Weights loaded. Missing keys: {len(load_result.missing_keys)}")

        if len(load_result.unexpected_keys) > 0:
            hydra.utils.log.info(f"Unexpected keys: {len(load_result.unexpected_keys)}")

        if cfg.train.get('require_full_finetune_load', False) and (
            load_result.missing_keys or load_result.unexpected_keys
        ):
            raise RuntimeError(
                "Full fine-tune checkpoint loading was requested, but the "
                f"checkpoint has {len(load_result.missing_keys)} missing and "
                f"{len(load_result.unexpected_keys)} unexpected keys."
            )

        if hasattr(model, 'pretrained_load_report'):
            model.pretrained_load_report = {
                "mode": "full_checkpoint_finetune",
                "checkpoint": str(Path(finetune_path).expanduser().resolve()),
                "missing_keys": list(load_result.missing_keys),
                "unexpected_keys": list(load_result.unexpected_keys),
                "full_load_required": bool(
                    cfg.train.get('require_full_finetune_load', False)
                ),
            }

        hydra.utils.log.info(f"================================")

    # Pass scaler from datamodule to model
    # hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.scaler}>")
    # if datamodule.scaler is not None:
    #     model.lattice_scaler = datamodule.lattice_scaler.copy()
    #     model.scaler = datamodule.scaler.copy()
    # torch.save(datamodule.lattice_scaler, hydra_dir / 'lattice_scaler.pt')
    # torch.save(datamodule.scaler, hydra_dir / 'prop_scaler.pt')
    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # Logger instantiation/configuration
    wandb_logger = None
    if "wandb" in cfg.logging:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
            # settings=wandb.Settings(start_method="fork"),
            tags=cfg.core.tags,
        )
        hydra.utils.log.info("W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )

    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    # Load checkpoint (if exist)
    ckpts = list(hydra_dir.glob('*.ckpt'))
    if len(ckpts) > 0:
        ckpt_epochs = np.array([int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
        ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        hydra.utils.log.info(f"found checkpoint: {ckpt}")
    else:
        ckpt = None

    hydra.utils.log.info("Instantiating the Trainer")
    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        # progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
        # resume_from_checkpoint=ckpt,
        **cfg.train.pl_trainer,
    )

    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    hydra.utils.log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)

    hydra.utils.log.info("Starting testing!")
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        hydra.utils.log.info(f"Testing best checkpoint: {best_model_path}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=best_model_path)
    else:
        hydra.utils.log.info("No checkpoint available; testing the in-memory model.")
        trainer.test(model=model, datamodule=datamodule)

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base=None)
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
