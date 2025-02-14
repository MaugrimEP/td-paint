import os
from dataclasses import asdict
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf

import wandb
from src.callbacks.distribution_callback import IDCallback
from conf.checkpoint_params import CheckpointsCallbacks, getModelCheckpoint
from conf.main_params import GlobalConfiguration
from conf.trainer_params import get_trainer
from conf.wandb_params import get_wandb, get_wandb_logger
from data.get_datamodule import get_dm
from src.callbacks.ema import EMA
from src.utils import get_model_class
from utils.utils import batch_size_finder, flatten, learning_rate_finder


@hydra.main(version_base=None, config_name="globalConfiguration", config_path="config_yaml")
def main(_cfg: GlobalConfiguration):
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    # region parameters loading
    keys = set()
    if _cfg.yaml_conf is not None:
        for yaml_file in _cfg.yaml_conf:
            yaml_content = OmegaConf.load(yaml_file)
            yaml_keys = set(flatten(yaml_content).keys())
            assert len(keys & yaml_keys) == 0, f"Duplicate keys in yaml files : {keys & yaml_keys}"
            keys |= set(yaml_content.keys())
            _cfg = OmegaConf.merge(
                _cfg, yaml_content,
            )  # command line configuration + yaml configuration

    _cfg = OmegaConf.merge(
        _cfg, {
            key: val for key, val in OmegaConf.from_cli().items()
            if "/" not in key and (not key.startswith("hydra"))
        }
    )  # command line configuration + yaml configuration + command line configuration
    # endregion

    print(OmegaConf.to_yaml(_cfg))
    cfg: GlobalConfiguration = OmegaConf.to_object(_cfg)

    cfg.ressources_params.set_if_needed()

    pl.seed_everything(cfg.seed)

    if cfg.system_params.torch_params.hub_dir is not None:
        if cfg.system_params.torch_params.hub_dir == "cwd":
            torch.hub.set_dir(os.path.join(os.getcwd(), "torch_hub"))
        else:
            torch.hub.set_dir(cfg.system_params.torch_params.hub_dir)

    if cfg.system_params.torch_params.torch_float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(
            cfg.system_params.torch_params.torch_float32_matmul_precision
        )

    model_class = get_model_class(cfg.model_params.name)

    # wandb
    run = get_wandb(params=cfg.wandb_params, global_dict=asdict(cfg))
    run_wandb = get_wandb_logger(params=cfg.wandb_params, global_dict=asdict(cfg))

    # Setup trainer
    dm = get_dm(cfg.dataset_params)
    model = model_class(cfg.model_params, cfg.dataset_params, cfg.args)

    if cfg.system_params.torch_params.compile:
        model = torch.compile(model)

    if cfg.trainer_params.cudnn_benchmark is not None:
        cudnn.benchmark = True

    # region callbacks
    callbacks = []
    
    if cfg.distribution_params.use:
        dm.setup()
        id_cb = IDCallback(
            cfg.distribution_params,
            train_dataset=dm.train_dataset,
            valid_dataset=dm.valid_dataset,
            test_dataset=dm.test_dataset,
            nb_gpus=int(cfg.trainer_params.devices),
            nb_nodes=cfg.trainer_params.num_nodes,
        )
        callbacks.append(id_cb)
        model.id_callback = id_cb

    modelCheckpoint: CheckpointsCallbacks = getModelCheckpoint(cfg.checkpoint_params)
    callbacks += ([modelCheckpoint.on_monitor] if modelCheckpoint.on_monitor is not None else [])
    callbacks += ([modelCheckpoint.on_epochs] if modelCheckpoint.on_epochs is not None else [])
    callbacks += ([modelCheckpoint.on_steps] if modelCheckpoint.on_steps is not None else [])
    callbacks += ([modelCheckpoint.on_tick] if modelCheckpoint.on_tick is not None else [])

    ema: Optional[EMA] = None
    if cfg.model_params.optimizer.ema.use and not cfg.args.use_gd_ema:
        print("[Init] EMA Callback")
        ema_params = cfg.model_params.optimizer.ema

        # region check EMA params
        if not ema_params.validate_original_weights:
            assert not ema_params.perform_double_validation
        # endregion

        ema = EMA(
            decay=ema_params.decay,
            validate_original_weights=ema_params.validate_original_weights,
            every_n_steps=ema_params.every_n_steps,
            cpu_offload=ema_params.cpu_offload,
        )
        callbacks.append(ema)
        print("[Init] EMA Callback Done")
    model.ema = ema
    # endregion

    trainer = get_trainer(cfg, callbacks, run_wandb)

    batch_size_finder(
        trainer=trainer,
        model=model,
        data_module=dm,
        cfg=cfg,
    )

    learning_rate_finder(
        trainer=trainer,
        model=model,
        data_module=dm,
        cfg=cfg,
    )

    if cfg.checkpoint_params.retrain_retrain_from_checkpoint == "load_weights":
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict=torch.load(cfg.checkpoint_params.retrain_saved_path)['state_dict'],
            strict=False,
        )
        print()
        print(f'Loaded weights from {cfg.checkpoint_params.retrain_saved_path=}')
        print()
        print(f'Missing keys: {missing_keys}')
        print()
        print(f'Unexpected keys: {unexpected_keys}')

        list_missing_not_allowed = []
        list_missing_allowed = []
        for missing_key in missing_keys:
            never_present = all([allowed_missing_key not in missing_key for allowed_missing_key in cfg.checkpoint_params.allowed_missing_keys])
            if never_present:
                list_missing_not_allowed.append(missing_key)
            else:
                list_missing_allowed.append(missing_key)

        list_unexpected_not_allowed = []
        list_unexpected_allowed = []
        for unexpected_key in unexpected_keys:
            never_present = all([allowed_unexpected_key not in unexpected_key for allowed_unexpected_key in cfg.checkpoint_params.allowed_unexpected_keys])
            if never_present:
                list_unexpected_not_allowed.append(unexpected_key)
            else:
                list_unexpected_allowed.append(unexpected_key)

        print()
        print(f'Number of missing keys not allowed: {len(list_missing_not_allowed)}')
        print(f'Number of missing keys allowed: {len(list_missing_allowed)}')
        print()
        print(f'Number of unexpected keys not allowed: {len(list_unexpected_not_allowed)}')
        print(f'Number of unexpected keys allowed: {len(list_unexpected_allowed)}')
        print()
        print(f'List of missing keys not allowed: {list_missing_not_allowed}')
        print()
        print(f'List of unexpected keys not allowed: {list_unexpected_not_allowed}')
        print()
        print(f'List of missing keys allowed: {list_missing_allowed}')
        print()
        print(f'List of unexpected keys allowed: {list_unexpected_allowed}')

        if len(list_missing_not_allowed) > 0 or len(list_unexpected_not_allowed) > 0:
            raise Exception("Missing or unexpected keys not allowed")

    # Train
    if cfg.trainer_params.skip_training:
        print("skip training")
    else:
        trainer.fit(
            model,
            datamodule=dm,
            ckpt_path=cfg.checkpoint_params.retrain_saved_path
            if cfg.checkpoint_params.retrain_retrain_from_checkpoint == "load_train"
            else None,
        )
        print("end fitting")

    if cfg.trainer_params.exit_after_training:
        print("exit after training")
        print(f"<TERMINATE WANDB>")
        wandb.finish()
        print(f"<WANDB TERMINATED>")

        return

    if cfg.checkpoint_params.loading_for_test_mode == "monitor":
        best_model = modelCheckpoint.on_monitor.best_model_path
        print(f"Load {best_model=} for testing")
        model.load_state_dict(torch.load(best_model)["state_dict"])
    elif cfg.checkpoint_params.loading_for_test_mode == "last":
        last_model = os.path.join(cfg.checkpoint_params.dirpath, "last.ckpt")
        print(f"Load last {last_model=}")
        model.load_state_dict(torch.load(last_model)["state_dict"])
    elif cfg.checkpoint_params.loading_for_test_mode == "none":
        print(f"No modelCheckpoint callback, continue")
    else:
        raise Exception(f"Unknown {cfg.checkpoint_params.loading_for_test_mode}")

    print("start testing")
    assert cfg.model_params.metrics.no_metrics is False, "Metrics should be enabled for testing"
    trainer.test(model, datamodule=dm)

    print("<TERMINATE WANDB>")
    wandb.finish()
    print("<WANDB TERMINATED>")


if __name__ == "__main__":
    main()
