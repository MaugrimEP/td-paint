from typing import Optional
from einops import rearrange
import pytorch_lightning as pl
import torch
from jaxtyping import Float, Int, jaxtyped
from lightning_utilities.core.rank_zero import rank_zero_only
import random
import wandb
from beartype import beartype as typechecker
from torchvision.utils import save_image

from utils.Logging.LogStrategy import LogStrategy
from utils.utils import (
    display_mask,
    display_tensor,
)


def norm_func(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-1, 1).contiguous().detach()


class LogCelebA(LogStrategy):
    def log_generate_uncond(
        self,
        plMod: pl.LightningModule,
        stage_prefix: str,
        batch_idx: int,
        batch: Float[torch.Tensor, "b 3 h w"],
        idx,
    ):
        remaining = self.already_logged(
            batch_idx=batch_idx,
            batch_size=batch.shape[0],
            max_quantity=self.params.log_generate_uncond.max_quantity,
        )
        if self.params.log_generate_uncond.early_leave:
            if remaining <= 0:
                return
            number_to_sample = remaining
        else:
            number_to_sample = batch.shape[0]

        with torch.no_grad():
            sample, classes, logged_data, dict_info = plMod.generate_samples(
                batch_size=number_to_sample,
            )

        # log images
        self.log_generate_log_images_uncond(
            plMod=plMod,
            batch_idx=batch_idx,
            stage_prefix=stage_prefix,
            sample=sample,
            logged_data=logged_data,
            classes=classes,
        )

    def log_generate_cond(
        self,
        plMod,
        stage_prefix: str,
        batch_idx: int,
        batch: Float[torch.Tensor, "b 3 h w"],
        mask: Int[torch.Tensor, "b 1 h w"],
        dict_info: dict,
        idx,
    ):
        remaining = self.already_logged(
            batch_idx=batch_idx,
            batch_size=batch.shape[0],
            max_quantity=self.params.log_generate_cond.max_quantity,
        )
        if self.params.log_generate_cond.early_leave:
            if remaining <= 0:
                return
            number_to_sample = remaining
        else:
            number_to_sample = batch.shape[0]

        batch = batch[:number_to_sample]
        mask = mask[:number_to_sample]
        if idx is not None:
            idx = idx[:number_to_sample]
        for key, value in dict_info.items():
            dict_info[key] = value[:number_to_sample]

        with torch.no_grad():
            sample, classes, logged_data, dict_info = plMod.generate_samples(
                batch_size=number_to_sample,
                mask=mask,
                gt=batch,
                dict_info=dict_info,
            )
        ellapsed_time_per_batch = dict_info["ellapsed"]
        ellasped_time_per_sample = ellapsed_time_per_batch / number_to_sample
        plMod.log(f"{stage_prefix}/ellapsed_time_per_batch", ellapsed_time_per_batch, on_epoch=True, on_step=True, sync_dist=True)
        plMod.log(f"{stage_prefix}/ellasped_time_per_sample", ellasped_time_per_sample, on_epoch=True, on_step=True, sync_dist=True)

        # log metrics
        self.log_generate_metrics_cond(
            stage_prefix=stage_prefix,
            plMod=plMod,
            batch=batch,
            mask=mask,
            pred=sample,
            idx=idx,
        )

        # log images
        self.log_generate_log_images_cond(
            plMod=plMod,
            batch_idx=batch_idx,
            stage_prefix=stage_prefix,
            batch=batch,
            mask=mask,
            classes=classes,
            logged_data=logged_data,
            sample=sample,
            idx=idx,
        )

    def log_generate_diversity(
        self,
        plMod: pl.LightningModule,
        stage_prefix: str,
        batch_idx: int,
        batch: Float[torch.Tensor, "b 3 h w"],
        mask: Int[torch.Tensor, "b 1 h w"],
        dict_info: dict,
        idx,
    ):
        original_batch_size = batch.shape[0]

        remaining = self.already_logged(
            batch_idx=batch_idx,
            batch_size=batch.shape[0],
            max_quantity=self.params.log_generate_diversity.max_quantity,
        )
        if self.params.log_generate_diversity.early_leave:
            if remaining <= 0:
                return
            number_to_sample = remaining
        else:
            number_to_sample = original_batch_size

        batch = batch[:number_to_sample]
        mask = mask[:number_to_sample]
        if idx is not None:
            idx = idx[:number_to_sample]
        
        ################################################

        # region preprocessing for diversity logging
        variation_quantity = self.params.log_generate_diversity.variation_quantity or original_batch_size
        generate_all_in_batch = self.params.log_generate_diversity.generate_all_in_batch
        # endregion

        if not generate_all_in_batch:  # reduce the batch to one element
            random_sample = random.randint(0, batch.shape[0] - 1)

            batch = batch[random_sample:random_sample + 1]
            mask = mask[random_sample:random_sample + 1]
            if idx is not None:
                idx = idx[random_sample:random_sample + 1]
            for key, value in dict_info.items():
                dict_info[key] = value[random_sample: random_sample + 1]

        # add the number of variation to the batch
        batch = batch.unsqueeze(1).repeat(1, variation_quantity, 1, 1, 1).flatten(end_dim=1)
        mask = mask.unsqueeze(1).repeat(1, variation_quantity, 1, 1, 1).flatten(end_dim=1)
        if idx is not None:
            idx = idx.unsqueeze(1).repeat(1, variation_quantity).flatten()
        for key, value in dict_info.items():
            dict_info[key] = value.unsqueeze(1).repeat(1, variation_quantity).flatten(end_dim=1)
        
        # generate the images
        samples = []
        classess = None
        logged_datas: list[list[dict]] = []
        for i in range(0, batch.shape[0], original_batch_size):
            _sample, _classes, _logged_data, _dict_info = plMod.generate_samples(
                batch_size=batch[i:i + original_batch_size].shape[0],
                mask=mask[i:i + original_batch_size],
                gt=batch[i:i + original_batch_size],
                dict_info={k: v[i: i + original_batch_size] for k, v in dict_info.items()},
            )
            samples.append(_sample)
            if _classes is not None:
                if classess is None:
                    classess = []
                classess.append(_classes)
            logged_datas.append(_logged_data)
            
        # put back the variation dimension
        batch = batch.unsqueeze(1).reshape(-1, variation_quantity, *batch.shape[1:])
        mask = mask.unsqueeze(1).reshape(-1, variation_quantity, *mask.shape[1:])
        if idx is not None:
            idx = idx.unsqueeze(1).reshape(-1, variation_quantity)
        for key, value in dict_info.items():
            dict_info[key] = value.unsqueeze(1).reshape(-1, variation_quantity, *value.shape[1:])

        # concat and put back the variation dimension for generated samples
        samples = torch.cat(samples, dim=0).reshape(-1, variation_quantity, *samples[0].shape[1:])
        if classess is not None:
            classess = torch.cat(classess, dim=0).reshape(-1, variation_quantity, *classess[0].shape[1:])
        tmp: list[dict] = logged_datas[0]  # final length should be equal to equal len of inner dict from logged_datas
        for i, logged_data in enumerate(logged_datas[1:]):  # : list[dict]
            for j, dict_step in enumerate(logged_data):
                for key, value in dict_step.items():
                    if isinstance(value, torch.Tensor):  # we have idx and t which are int, other are tensors
                        tmp[j][key] = torch.cat([tmp[j][key], value], dim=0)
        for i, dict_step in enumerate(tmp):
            for key, value in dict_step.items():
                if isinstance(value, torch.Tensor):  # we have idx and t which are int, other are tensors
                    tmp[i][key] = value.reshape(-1, variation_quantity, *value.shape[1:])
        logged_datas = tmp

        # log metrics
        self.log_generate_metrics_diversity(
            stage_prefix=stage_prefix,
            plMod=plMod,
            batch=batch,
            mask=mask,
            pred=samples,
            idx=idx,
        )

        # log images
        self.log_generate_log_images_diversity(
            plMod=plMod,
            batch_idx=batch_idx,
            stage_prefix=stage_prefix,
            batch=batch,
            mask=mask,
            classes=classess,
            pred=samples,
            idx=idx,
        )
##################

    @jaxtyped
    @typechecker
    def log_generate_log_images_cond(
        self,
        plMod,
        batch_idx: int,
        stage_prefix: str,
        batch: Float[torch.Tensor, 'b 3 h w'],
        mask: Int[torch.Tensor, 'b 1 h w'],
        classes: Optional[Int[torch.Tensor, 'b']],
        logged_data: list[dict],
        sample: Float[torch.Tensor, 'b 3 h w'],
        idx: Int[torch.Tensor, 'b'],
    ):
        xt = torch.cat([data['xt'] for data in logged_data], dim=-1)
        x0 = torch.cat([data['x0'] for data in logged_data], dim=-1)

        full_tensor = torch.cat(
            [xt, x0], dim=-2
        )  # concat them in height dimension

        # add the GT and MASK
        mask_and_gt = torch.cat([
            mask.expand(-1, 3, -1, -1) * 2 - 1,
            norm_func(batch)
        ], dim=-2).cpu()
        # currently mask is in [0, 1] we put it in [-1, 1] like the rest
        mask_and_gt = mask_and_gt
        
        # add final generation and a black square at the end
        black_square = torch.full_like(sample, -1)
        final_and_back = torch.cat([sample, black_square], dim=-2).cpu() # cat accros height

        full_tensor = torch.cat([full_tensor, mask_and_gt, final_and_back], dim=-1)
        if classes is not None:
            classes_str = [f"c:{classes[i].item()} " for i in range(classes.shape[0])]
        else:
            classes_str = ["" for i in range(xt.shape[0])]
        
        caption_list = [f"id:{i} {c}xt and x0_hat" for i, c in zip(idx, classes_str)]
        
        self.save_to_wandb(
            stage_prefix=stage_prefix,
            plMod=plMod,
            image_list=full_tensor,
            caption_list=caption_list,
            batch_idx=batch_idx,
            max_quantity=self.params.log_generate_cond.max_quantity,
        )
        
        self.save_to_disk_cond(
            plMod=plMod,
            stage_prefix=stage_prefix,
            idx=idx,
            image_list=sample,
            mask_list=mask,
            classes=classes,
        )

    @jaxtyped
    @typechecker
    def log_generate_log_images_uncond(
        self,
        plMod,
        batch_idx: int,
        stage_prefix: str,
        sample: Float[torch.Tensor, 'b 3 h w'],
        logged_data: list[dict],
        classes: Optional[Int[torch.Tensor, 'b']],
    ):
        xt = torch.cat([data['xt'] for data in logged_data], dim=-1)
        x0 = torch.cat([data['x0'] for data in logged_data], dim=-1)

        full_tensor = torch.cat(
            [xt, x0], dim=-2
        )  # concat them in height dimension

        if classes is not None:
            classes_str = [f"c:{classes[i].item()} " for i in range(classes.shape[0])]
        else:
            classes_str = ["" for i in range(xt.shape[0])]
            
        caption_list = [f"{classe}xt and x0_hat" for classe in classes_str]
        
        self.save_to_wandb(
            stage_prefix=stage_prefix,
            plMod=plMod,
            image_list=full_tensor,
            caption_list=caption_list,
            batch_idx=batch_idx,
            max_quantity=self.params.log_generate_uncond.max_quantity,
        )
        self.save_to_disk_uncond(
            plMod=plMod,
            stage_prefix=stage_prefix,
            image_list=sample,
            classes=classes,
        )

    @jaxtyped
    @typechecker
    def log_generate_log_images_diversity(
        self,
        stage_prefix: str,
        plMod: pl.LightningModule,
        batch_idx: int,
        batch: Float[torch.Tensor, 'b diversity c h w'],
        mask: Int[torch.Tensor, 'b diversity 1 h w'],
        classes: Optional[Int[torch.Tensor, 'b diversity']],
        pred: Float[torch.Tensor, 'b diversity c h w'],  # list is number of steps logged
        idx: Int[torch.Tensor, 'b diversity'],
    ) -> None:
        original_pred = pred.clone()
        batch_size, diversity, c, h, w = batch.shape
        # concat the generated samples accros the diversity
        pred = rearrange(pred, 'b diversity c h w -> b c h (diversity w)')

        # remove diversity where we only sample one: batch mask and idx
        batch = batch[:, 0]
        mask = mask[:, 0]
        idx = idx[:, 0]

        # add GT and mask
        img_cat = torch.cat([
            batch.cpu(),
            mask.cpu().repeat(1, 3, 1, 1) * 2 - 1,
            pred.cpu(),
        ], dim=-1)

        if classes is not None:
            classes_str = [f"c:{classes[i, 0].item()} " for i in range(classes.shape[0])]  # TODO when allows for multiples classes in diversty should refactor that
        else:
            classes_str = ["" for i in range(batch_size)]
            
        caption_list = [
            (f'id:{idx[i]} ' if idx is not None else '') + f'{classes_str[i]}gt and variations'
            for i in range(img_cat.shape[0])
        ]
        
        self.save_to_wandb(
            stage_prefix=stage_prefix,
            plMod=plMod,
            image_list=img_cat,
            caption_list=caption_list,
            batch_idx=batch_idx,
            max_quantity=self.params.log_generate_diversity.max_quantity,
        )
        self.save_to_disk_diversity(
            plMod=plMod,
            stage_prefix=stage_prefix,
            idx=idx,
            image_list=original_pred,
            mask_list=mask,
            classes=classes,
        )

    @jaxtyped
    @typechecker
    def log_train(
        self,
        stage_prefix: str,
        plMod: pl.LightningModule,
        batch: Float[torch.Tensor, 'b 3 h w'],
        ts: Int[torch.Tensor, 'b 1 h w'] | Int[torch.Tensor, 'b'],
        x0pred: Float[torch.Tensor, 'b 3 h w'],
        xt: Float[torch.Tensor, 'b 3 h w'],
        idx: Int[torch.Tensor, 'b'],
        batch_idx: int,
    ):
        t_is_map = ts.dim() != 1
        images = torch.cat([batch, xt, x0pred], dim=-1).clamp(-1, 1)
        if t_is_map:
            # put current map into RGB
            ts_map = (1 - (ts / plMod.max_steps)) * 2 - 1
            ts_map = ts_map.repeat(1, 3, 1, 1)
            images = torch.cat([ts_map, images], dim=-1)

        caption_list = []
        for j, (i, image) in enumerate(zip(idx, images)):
            t_caption = f't:{ts[j].item()}' if not t_is_map else 't_map'
            caption = f"id:{i} {t_caption} x0 xt x0_hat"
            caption_list.append(caption)
            
        self.save_to_wandb(
            stage_prefix=stage_prefix,
            plMod=plMod,
            image_list=images,
            caption_list=caption_list,
            batch_idx=batch_idx,
            max_quantity=self.params.log_steps.max_quantity,
        )

    @jaxtyped
    @typechecker
    def log_generate_metrics_cond(
        self,
        stage_prefix: str,
        plMod: pl.LightningModule,
        batch: Float[torch.Tensor, 'b 3 h w'],
        mask: Int[torch.Tensor, 'b 1 h w'],
        pred: Float[torch.Tensor, 'b 3 h w'],
        idx: Int[torch.Tensor, 'b'],
    ) -> None:
        metrics_obs = plMod.get_metric_object()
        metric_dict = metrics_obs.get_dict_generation_cond(
            data=batch,
            prediction=pred,
            mask=mask,
        )
        for metric_name, value in metric_dict.items():
            plMod.log_g(stage_prefix, metric_name, value)

    @jaxtyped
    @typechecker
    def log_generate_metrics_diversity(
        self,
        stage_prefix: str,
        plMod: pl.LightningModule,
        batch: Float[torch.Tensor, 'b diversity c h w'],
        mask: Int[torch.Tensor, 'b diversity 1 h w'],
        pred: Float[torch.Tensor, 'b diversity c h w'],
        idx: Int[torch.Tensor, 'b diversity'],
    ) -> None:

        metrics_obs = plMod.get_metric_object()
        metric_dict = metrics_obs.get_dict_generation_diversity(
            batch=batch,
            prediction=pred,
        )
        for metric_name, value in metric_dict.items():
            plMod.log_g(stage_prefix, metric_name, value)

    @rank_zero_only
    @jaxtyped
    @typechecker
    def save_to_wandb(
        self,
        plMod: pl.LightningModule,
        stage_prefix: str,
        image_list: Float[torch.Tensor, 'b 3 h w'],
        caption_list: list[str],
        batch_idx: int,
        max_quantity: int,
    ):
        wandb_image_list = [
            wandb.Image(image, caption=caption)
            for image, caption in zip(image_list, caption_list)
        ]
        
        remaining = self.already_logged(
            batch_idx=batch_idx,
            batch_size=len(wandb_image_list),
            max_quantity=max_quantity,
        )

        if remaining <= 0:
            return
        remaining = min(remaining, len(wandb_image_list))
        
        wandb_image_list = wandb_image_list[:remaining]

        plMod.logger.experiment.log({f"{stage_prefix}/image": wandb_image_list})

    @jaxtyped
    @typechecker
    def save_to_disk_uncond(
        self,
        plMod: pl.LightningModule,
        stage_prefix: str,
        image_list: Float[torch.Tensor, 'b 3 h w'],
        classes: Optional[Int[torch.Tensor, 'b']],
    ) -> None:
        stage_prefix = stage_prefix.replace("/", "_")
        logging_params = self.params.log_generate_uncond
        prefix = "uncond"

        save_img_to_disk = plMod.get_stage() in logging_params.save_image_to_disk_stage
        save_pt_to_disk = plMod.get_stage() in logging_params.save_pt_to_disk_stage
        if not save_img_to_disk and not save_pt_to_disk:
            return

        for i, img_i in enumerate(image_list):
            classe_txt = ""
            if classes is not None:
                classe_txt = f"_classe_{classes[i].item()}"

            filename = f'{logging_params.save_path}/{stage_prefix}_{prefix}_{i}{classe_txt}'
            if save_img_to_disk:
                save_image(img_i, fp=f'{filename}.png', value_range=(-1, 1), normalize=True)
            if save_pt_to_disk:
                torch.save(img_i, f'{filename}.pt')

    @jaxtyped
    @typechecker
    def save_to_disk_cond(
        self,
        plMod: pl.LightningModule,
        stage_prefix: str,
        idx: Int[torch.Tensor, 'b'],
        image_list: Float[torch.Tensor, 'b 3 h w'],
        mask_list: Int[torch.Tensor, 'b 1 h w'],
        classes: Optional[Int[torch.Tensor, 'b']],
    ) -> None:
        stage_prefix = stage_prefix.replace("/", "_")
        logging_params = self.params.log_generate_cond
        prefix = "cond"

        save_img_to_disk = plMod.get_stage() in logging_params.save_image_to_disk_stage
        save_pt_to_disk = plMod.get_stage() in logging_params.save_pt_to_disk_stage
        save_mask = logging_params.save_mask
        if not save_img_to_disk and not save_pt_to_disk:
            return

        mask_list = mask_list.float()
        for i, (idx_i, img_i, mask_i) in enumerate(zip(idx, image_list, mask_list)):
            idx_i = idx_i.item()
            classe_txt = ""
            if classes is not None:
                classe_txt = f"_classe_{classes[i].item()}"

            img_filename = f'{logging_params.save_path}/{stage_prefix}_{prefix}_{idx_i}_img{classe_txt}'
            mask_filename = f'{logging_params.save_path}/{prefix}_{idx_i}_mask'
            if save_img_to_disk:
                save_image(img_i, fp=f'{img_filename}.png', value_range=(-1, 1), normalize=True)
                if save_mask: save_image(mask_i, fp=f'{mask_filename}.png')
            if save_pt_to_disk:
                torch.save(img_i, f'{img_filename}.pt')
                if save_mask: torch.save(mask_i, f'{mask_filename}.pt')

    @jaxtyped
    @typechecker
    def save_to_disk_diversity(
        self,
        plMod: pl.LightningModule,
        stage_prefix: str,
        idx: Int[torch.Tensor, 'b'],
        image_list: Float[torch.Tensor, 'b diversity 3 h w'],
        mask_list: Int[torch.Tensor, 'b 1 h w'],
        classes: Optional[Int[torch.Tensor, 'b diversity']],
    ) -> None:
        stage_prefix = stage_prefix.replace("/", "_")
        logging_params = self.params.log_generate_diversity
        prefix = "div"
        nb_div = image_list.shape[1]

        save_img_to_disk = plMod.get_stage() in logging_params.save_image_to_disk_stage
        save_pt_to_disk = plMod.get_stage() in logging_params.save_pt_to_disk_stage
        save_mask = logging_params.save_mask
        if not save_img_to_disk and not save_pt_to_disk:
            return

        mask_list = mask_list.float()
        for i, (idx_i, img_i_div, mask_i) in enumerate(zip(idx, image_list, mask_list)):
            idx_i = idx_i.item()

            # region save mask once
            mask_filename = f'{logging_params.save_path}/{prefix}_{idx_i}_mask'
            if save_img_to_disk and save_mask:
                save_image(mask_i, fp=f'{mask_filename}.png')
            if save_pt_to_disk and save_mask:
                torch.save(mask_i, f'{mask_filename}.pt')
            # endregion

            for div_i, img_i in enumerate(img_i_div):
                classe_txt = ""
                if classes is not None:
                    classe_txt = f"_classe_{classes[i, div_i].item()}"

                img_filename = f'{logging_params.save_path}/{stage_prefix}_{prefix}_{idx_i}_{div_i}_img{classe_txt}'
                if save_img_to_disk:
                    save_image(img_i, fp=f'{img_filename}.png', value_range=(-1, 1), normalize=True)
                if save_pt_to_disk:
                    torch.save(img_i, f'{img_filename}.pt')
