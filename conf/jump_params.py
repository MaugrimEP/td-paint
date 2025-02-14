from dataclasses import dataclass


@dataclass
class JumpParams:
    use: bool = False
    
    t_T: int = 250
    n_sample: int = 1
    jump_length: int = 10
    jump_n_sample: int = 10
    
    inpa_inj_time_shift: int = 1
