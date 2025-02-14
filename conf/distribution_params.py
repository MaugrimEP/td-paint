from dataclasses import dataclass
from typing import Optional


@dataclass
class DistributionDistanceParams:
    use            : bool = False
    
    # FID params
    fid_dims       : int = 2048

    # KID params
    kid_feature    : int = 2048
    kid_subsets    : int = 100
    kid_subset_size: int = 1_000
    kid_degree     : int = 3
    kid_gamma      : Optional[float] = None
    kid_coef       : float = 1.0

    # General params
    init                    : bool = True  # If True and load_initialization_path does not exist, init with real stats
    fid_load_initialization_path: Optional[str] = './_fids/fid_init.ckpt'
    kid_load_initialization_path: Optional[str] = './_kids/kid_init.ckpt'
    number_to_generate      : int = 2_000
    check_frequency         : int = 1  # frequency to compute in step for validation. Test always compute
    compute_first           : bool = False
    stages                  : list[str] = tuple(['valid', 'test'])

    # Dataloader params
    batch_size     : int = 100
    num_workers    : int = 10
    pin_memory     : bool = True
    prefetch_factor: int = 2

    compute_running: bool = True  # If compute during running validation and testing step
    compute_on_ema : bool = True

    # If we .compute() multiple time during the process
    running_compute     : bool = False
    running_compute_freq: int = 1_000
