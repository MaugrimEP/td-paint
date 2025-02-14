from dataclasses import dataclass
from typing import Optional


@dataclass
class TorchParams:
    hub_dir: Optional[str] = None
    """
    None: use default torch params
    'cwd': use cwd/torch_hub
    'path': use path
    """
    compile: bool = False  # compile the model: pytorch >= 2.0
    torch_float32_matmul_precision: Optional[str] = None  # [ medium | high | highest ]


@dataclass
class SystemParams:
    torch_params: TorchParams = TorchParams()
