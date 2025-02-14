from dataclasses import dataclass
from typing import Optional


@dataclass
class RessourcesParams:
    to_set: bool = False

    n_core: Optional[str] = "8"
    n_thread: Optional[int] = 8
    visible_devices: Optional[list[int]] = None

    def set_if_needed(self ):
        import os
        import torch

        N_CORE = self.n_core
        N_THREADS_TORCH = self.n_thread

        if not self.to_set:
            return

        if N_CORE is not None:
            os.environ["OMP_NUM_THREADS"] = N_CORE
            os.environ["OPENBLAS_NUM_THREADS"] = N_CORE
            os.environ["MKL_NUM_THREADS"] = N_CORE
            os.environ["VECLIB_MAXIMUM_THREADS"] = N_CORE
            os.environ["NUMEXPR_NUM_THREADS"] = N_CORE
        if  N_THREADS_TORCH is not None:
            torch.set_num_threads(N_THREADS_TORCH)

        if self.visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in self.visible_devices])
