"""
Scripts to isolate the differents part of Places test dataset with associated masks
"""
import os

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from tqdm import tqdm

from conf.main_params import GlobalConfiguration
from data.get_datamodule import get_dm
from utils.utils import (display_mask, display_tensor, flatten,
                         read_list_from_file, write_list_to_file)


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

    cfg.dataset_params.data_params.return_path=True
    dm = get_dm(cfg.dataset_params)
    dm.setup()
    
    assert len(cfg.dataset_params.mask_params.mask_type) == 1, "Select only one mask type"
    mask_name = cfg.dataset_params.mask_params.mask_type[0]
    root_dest = f"/dlocal/home/2022022/tmayet02/datasets/places/masked_places"

    full_str = ""
    full_str += f"mkdir {root_dest}/{mask_name}/ \n"
    for idx, _img, dict_info, mask in tqdm(dm.test_dataset):
        img_path = dict_info["img_path"]
        mask_path = dict_info["mask_path"]
        
        img_path_no_extension, img_extension = os.path.splitext(img_path)
        img_extension = img_extension[1:]
        img_filename = os.path.basename(img_path_no_extension)
        
        save_img_name = f"id_{idx}_{img_filename}"
        save_mask_name = save_img_name + "_mask"
        
        full_str += f"cp {img_path} {root_dest}/{mask_name}/{save_img_name}.{img_extension} \n"
        full_str += f"cp {mask_path} {root_dest}/{mask_name}/{save_mask_name}.png \n"
    # print(full_str)
    
    with open(f"copy_{mask_name}.sh", "w") as f:
        f.write(full_str)
    

if __name__ == "__main__":
    main()
