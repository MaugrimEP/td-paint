"""
Script used to evaluate predictions either from images or from pt files.
Look at conf/evaluation_params.py for the parameters.
"""
import os
from dataclasses import asdict

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb
from conf.main_params import GlobalConfiguration
from conf.wandb_params import get_wandb
from data.get_datamodule import get_dm
from utils.evaluation.utils_diversity import get_prediction_from
from utils.Metric.Metrics import get_metrics
from utils.utils import display_mask, display_tensor, flatten


@hydra.main(
    version_base=None, config_name="globalConfiguration", config_path="config_yaml"
)
def main(_cfg: GlobalConfiguration):
    print(f"Working directory : {os.getcwd()}")
    print(
        f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}" # type: ignore
    )

    # region parameters loading
    keys = set()
    if _cfg.yaml_conf is not None:
        for yaml_file in _cfg.yaml_conf:
            yaml_content = OmegaConf.load(yaml_file)
            yaml_keys = set(flatten(yaml_content).keys())
            assert (
                len(keys & yaml_keys) == 0
            ), f"Duplicate keys in yaml files : {keys & yaml_keys}"
            keys |= set(yaml_content.keys())
            _cfg = OmegaConf.merge(
                _cfg,
                yaml_content,
            )  # type: ignore # command line configuration + yaml configuration

    _cfg = OmegaConf.merge(
        _cfg,
        {
            key: val
            for key, val in OmegaConf.from_cli().items()
            if "/" not in key and (not key.startswith("hydra")) # type: ignore
        },
    )  # type: ignore # command line configuration + yaml configuration + command line configuration
    # endregion

    print(OmegaConf.to_yaml(_cfg))
    cfg: GlobalConfiguration = OmegaConf.to_object(_cfg) # type: ignore

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

    # wandb
    run = get_wandb(params=cfg.wandb_params, global_dict=asdict(cfg))

    dm = get_dm(cfg.dataset_params)
    dm.setup()
    test_dataset = dm.test_dataset
    test_metrics_png = get_metrics(cfg.model_params.metrics)(cfg.model_params, cfg.dataset_params)

    device = cfg.evaluation_params.device 
    test_metrics_png = test_metrics_png.to(device)

    nb_diversity = cfg.evaluation_params.diversity_nb
    for i, (idx, img, info, mask) in enumerate(tqdm(test_dataset)): # type: ignore
        _preds = []
        for i_diversity in range(nb_diversity):
            _pred_png = get_prediction_from(params=cfg.evaluation_params, idx_in_100=i, idx_in_ds=idx.item(), idx_diversity=i_diversity)
            _preds.append(_pred_png)
        preds = torch.stack(_preds, dim=0)  # [diversity_nb, c, h, w ]
        assert list(preds.shape) == [nb_diversity] + list(img.shape)

        preds = preds.to(device)
        img = img.to(device)
        
        # data should have the batch dimension
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)
        preds = preds

        test_metrics_png.get_dict_generation_cond(
            data=img.repeat(nb_diversity, 1, 1, 1),
            prediction=preds,
            mask=mask.repeat(nb_diversity, 1, 1, 1),
        )
        test_metrics_png.get_dict_generation_diversity(
            batch=img.reshape(1, 1, 3, 256, 256).repeat(1, nb_diversity, 1, 1, 1),
            prediction=preds.reshape(1, nb_diversity, 3, 256, 256),
        )
    
    lpips_face_png = test_metrics_png.lpips_clamp_face.compute()
    ssim_face_png = test_metrics_png.ssim_clamp_face.compute()
    kid_mean, kid_std = test_metrics_png.kid_clamp_face.compute()
    diversity_face_png = test_metrics_png.diversity.compute()
    wandb.log({
        'lpips_face_rgb': lpips_face_png,
        'ssim_face_rgb': ssim_face_png,
        'kid_mean': kid_mean,
        'kid_std': kid_std,
        'diversity_face_rgb': diversity_face_png,
    })

    print("<TERMINATE WANDB>")
    wandb.finish()
    print("<WANDB TERMINATED>")


if __name__ == "__main__":
    main()
