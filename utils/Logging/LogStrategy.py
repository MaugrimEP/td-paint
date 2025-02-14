import pytorch_lightning as pl
import torch.nn as nn

from conf.dataset_params import DatasetParams
from conf.model_params import LoggingParams


class LogStrategy(nn.Module):
    def __init__(
        self,
        params: LoggingParams,
        params_data: DatasetParams,
    ):
        super().__init__()
        self.params = params
        self.params_data = params_data

    @staticmethod
    def already_logged(
        batch_idx: int,
        batch_size: int,
        max_quantity: int,
    ) -> int:
        """
        Return how much of the sample should be logged
        """
        already_logged = batch_idx * batch_size
        remaining = max(0, max_quantity - already_logged)
        remaining = min(remaining, batch_size)
        return remaining

    def log_train(
        self,
        stage_prefix: str,
        plMod,
        batch,
        ts,
        x0pred,
        xt,
        idx,
        batch_idx,
    ):
        raise NotImplementedError

    def log_generate(
        self,
        stage_prefix,
        plMod,
        batch,
        batch_idx,
        idx=None,
    ):
        raise NotImplementedError
