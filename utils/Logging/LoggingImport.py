from conf.dataset_params import DatasetParams
from conf.model_params import LoggingParams
from utils.Logging.LogCelebA import LogCelebA
from utils.Logging.LogStrategy import LogStrategy


def get_log_strategy(params: LoggingParams, params_data: DatasetParams) -> LogStrategy:
    name = params.name

    loggers = {
        "celeba": LogCelebA,
    }

    if name not in loggers:
        raise NotImplementedError(f"Logging strategy {name} not implemented")

    return loggers[name](params, params_data)
