"""
Because Imagenet1k might be very big to evaluate, we can subsample it to a smaller dataset.
Initial validation size is 50K
We have 1K classes, and we might takes 5 images per classes to have a test set of size 5 000
Because we asign a mask for each images at the end, we want to have t he good (image, mask) regardless
of the data splitting
Therefore we will compute the full dataset then at the end perform the filtering
using the same function we used when we wanted to perform the test on multiple runs
but the already done index will be the index which are not part of the validation set

for the validation size, we will takes 2 000 images, so 2 images per classes
"""
import os
from typing import Any

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from tqdm import tqdm

from conf.main_params import GlobalConfiguration
from data.get_datamodule import get_dm
from utils.utils import (display_mask, display_tensor, flatten,
                         read_list_from_file, write_list_to_file)

# PARAMETERS
images_per_class: int = 2
filename = "retained_samples.txt"
list_to_remove = "splits/imagenet_5000_idx_test.txt"  # the list of int to remove, they are already in another computed set
index_already_taken = read_list_from_file(list_to_remove)
number_of_classes = 1_000



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

    dm = get_dm(cfg.dataset_params)
    dm.setup()
    
    # select some images per class to keep
    retained_samples: dict[int, Any] = dict()
    validation_dataset = dm.valid_dataset
    assert len(validation_dataset) == 50_000
    
    for idx, _data, dict_info, _mask in validation_dataset:
        if idx in index_already_taken:
            continue
        classe = dict_info['y'].item()
        idx = idx.item()
        
        already_done_in_this_class = len(retained_samples.get(classe, []))
        if already_done_in_this_class < images_per_class:
            if classe not in retained_samples:
                retained_samples[classe] = []
            retained_samples[classe].append(idx)
    
    total_len = sum(len(v) for v in retained_samples.values())
    assert total_len == already_done_in_this_class * number_of_classes, f"Total length is {total_len} instead of {already_done_in_this_class * number_of_classes=}"
    
    print('saving results to a file')
    # save result to a file
    with open(filename, "w") as f:
        for classe, idxs in retained_samples.items():
            for idx in idxs:
                f.write(f"{idx},")

if __name__ == "__main__":
    main()
