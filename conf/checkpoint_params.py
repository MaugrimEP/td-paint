from dataclasses import dataclass
from typing import Optional

from pytorch_lightning.callbacks import ModelCheckpoint


@dataclass
class CheckpointParams:
    #################################
    model_checkpoint_on_monitor: bool = False  # check the metric
    model_checkpoint_on_epochs: bool = True  # save every N epochs
    model_checkpoint_on_steps: bool = False  # save every N steps
    model_checkpoint_on_tick: bool = True  # tick on the last every N duration (only here to update last with, so independent of top k)
    loading_for_test_mode: str = "none"  # [ none | monitor | last ]

    # generals params
    dirpath: str = "_model_save/"
    auto_insert_metric_name: bool = False
    save_weights_only: bool = False
    verbose: bool = True
    save_on_train_epoch_end: Optional[bool] = True  # without this, does not save at all :think:
    save_last: Optional[bool] = False  # create last.ckpt (for each checkpoint: cp checkpoint.ckpt last.ckpt)

    #################################
    #################################

    # region ON MONITOR
    on_monitor__filename: Optional[str] = "{epoch:05d}_best_model"
    monitor: Optional[str] = "valid/step/loss"  # if None save to the last epoch
    mode: str = "min"
    on_monitor__every_n_epochs: Optional[int] = 1
    on_monitor__save_top_k: int = 1
    # endregion

    #################################

    # region ON EPOCHS
    on_epochs__filename: Optional[str] = "{epoch:05d}_epochs_model"
    on_epochs__save_top_k: int = 1
    on_epochs__every_n_epochs: Optional[int] = 1
    # endregion

    # region ON STEPS
    on_steps__filename: Optional[str] = "{epoch:05d}_{steps}_steps_model"
    on_steps__save_top_k: int = 1
    on_steps__every_n_steps: Optional[int] = 30_000
    # endregion
    #################################

    # region ON TICK
    on_tick__filename: Optional[str] = "{epoch:05d}_duration_model"
    on_tick__every_n_epochs: Optional[int] = 1
    # endregion

    #################################
    retrain_retrain_from_checkpoint: str = (
        "dont"  # [ dont | load_weights | load_train ]
    )
    allowed_missing_keys: list[str] = tuple(['train_metrics.', 'valid_metrics.', 'test_metrics.', 'classifier.'])
    allowed_unexpected_keys: list[str] = tuple(['train_metrics.', 'valid_metrics.', 'test_metrics.', 'classifier.'])
    #################################
    retrain_saved_path: str = "_model_save/last.ckpt"


@dataclass
class CheckpointsCallbacks:
    on_monitor: Optional[ModelCheckpoint]
    on_epochs: Optional[ModelCheckpoint]
    on_steps: Optional[ModelCheckpoint]
    on_tick: Optional[ModelCheckpoint]


def getModelCheckpoint(params: CheckpointParams) -> CheckpointsCallbacks:
    if params.model_checkpoint_on_monitor:
        on_monitor = ModelCheckpoint(
            dirpath=params.dirpath,
            filename=params.on_monitor__filename,
            monitor=params.monitor,
            verbose=params.verbose,
            save_last=params.save_last,
            save_top_k=params.on_monitor__save_top_k,
            mode=params.mode,
            auto_insert_metric_name=params.auto_insert_metric_name,
            save_weights_only=params.save_weights_only,
            every_n_epochs=params.on_monitor__every_n_epochs,
            save_on_train_epoch_end=params.save_on_train_epoch_end,
        )
    else:
        on_monitor = None

    if params.model_checkpoint_on_epochs:
        on_epochs = ModelCheckpoint(
            dirpath=params.dirpath,
            filename=params.on_epochs__filename,
            auto_insert_metric_name=params.auto_insert_metric_name,
            save_weights_only=params.save_weights_only,
            monitor="epoch",
            mode="max",
            save_top_k=params.on_epochs__save_top_k,
            verbose=params.verbose,
            save_last=params.save_last,
            every_n_epochs=params.on_epochs__every_n_epochs,
            save_on_train_epoch_end=params.save_on_train_epoch_end,
        )
    else:
        on_epochs = None

    if params.model_checkpoint_on_steps:
        on_steps = ModelCheckpoint(
            dirpath=params.dirpath,
            filename=params.on_steps__filename,
            auto_insert_metric_name=params.auto_insert_metric_name,
            save_weights_only=params.save_weights_only,
            monitor="steps",
            mode="max",
            save_top_k=params.on_steps__save_top_k,
            verbose=params.verbose,
            save_last=params.save_last,
            every_n_train_steps=params.on_steps__every_n_steps,
        )
    else:
        on_steps = None

    if params.model_checkpoint_on_tick:
        on_tick = ModelCheckpoint(
            dirpath=params.dirpath,
            filename=params.on_tick__filename,
            verbose=params.verbose,
            save_last=True,
            save_top_k=0,
            auto_insert_metric_name=params.auto_insert_metric_name,
            save_weights_only=params.save_weights_only,
            every_n_epochs=params.on_monitor__every_n_epochs,
            save_on_train_epoch_end=params.save_on_train_epoch_end,
        )
    else:
        on_tick = None

    return CheckpointsCallbacks(
        on_monitor=on_monitor,
        on_epochs=on_epochs,
        on_steps=on_steps,
        on_tick=on_tick
    )
