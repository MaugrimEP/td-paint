import copy
import time
import random
from typing import Any, Literal, Optional
from torchvision.utils import save_image
import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn.functional as F
from einops import repeat
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype as typechecker
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from src.Backbones.guided_diffusion.gaussian_diffusion import _extract_into_tensor

from conf.dataset_params import DatasetParams
from conf.guided_diffusion_params import GuidedDiffusionParams
from conf.model_params import ModelParams
from src.Backbones.guided_diffusion.fp16_util import MixedPrecisionTrainer
from src.Backbones.guided_diffusion.resample import (
    create_named_schedule_sampler,
    LossAwareSampler,
)
from src.Backbones.guided_diffusion.script_util import create_classifier, create_classifier_and_diffusion, create_model_and_diffusion
from utils.Logging.LoggingImport import get_log_strategy
from utils.Metric.Metrics import get_metrics
from utils.masks_lama.masks_lama import MixedMaskGenerator
from utils.utils import (
    display_mask, display_tensor,
    is_logging_time,
    patch_to_full,
)

NUM_CLASSES = 1000


def extract(a, t, x_shape):
    a = a.to(t.device)
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DiffusionModel(pl.LightningModule):
    def __init__(
        self,
        params: ModelParams,
        params_data: DatasetParams,
        args: GuidedDiffusionParams,
    ):
        super().__init__()
        self.params = params
        self.params_data = params_data
        self.args = args
        
        # region just check some arguments
        if args.use_fp16:
           assert args.use_gd_optimizer, "use_fp16 is only available with use_gd_optimizer" 
           assert args.classifier_use_fp16, "use_fp16 should also set classifier_use_fp16" 
        # endregion

        self.model, self.diffusion = create_model_and_diffusion(
            image_size=args.network_img_size or params_data.data_params.image_size,
            class_cond=args.class_cond,
            learn_sigma=args.learn_sigma,
            num_channels=args.num_channels,
            num_res_blocks=args.num_res_blocks,
            channel_mult=args.channel_mult,
            num_heads=args.num_heads,
            num_head_channels=args.num_head_channels,
            num_heads_upsample=args.num_heads_upsample,
            attention_resolutions=args.attention_resolutions,
            dropout=args.dropout,
            diffusion_steps=args.diffusion_steps,
            noise_schedule=args.noise_schedule,
            timestep_respacing=args.timestep_respacing,
            use_kl=args.use_kl,
            predict_xstart=args.predict_xstart,
            rescale_timesteps=args.rescale_timesteps,
            rescale_learned_sigmas=args.rescale_learned_sigmas,
            use_checkpoint=False,
            use_scale_shift_norm=args.use_scale_shift_norm,
            resblock_updown=args.resblock_updown,
            use_fp16=args.use_fp16,
            use_new_attention_order=args.use_new_attention_order,
            args=args,
            model_params=params,
        )
        if self.args.model_path is not None:
            self.model.load_state_dict(torch.load(self.args.model_path))
        
        if args.use_classifier:
            print("Loading classifier")
            self.classifier = create_classifier(
                image_size=args.image_size,
                classifier_use_fp16=args.classifier_use_fp16,
                classifier_width=args.classifier_width,
                classifier_depth=args.classifier_depth,
                classifier_attention_resolutions=args.classifier_attention_resolutions,
                classifier_use_scale_shift_norm=args.classifier_use_scale_shift_norm,
                classifier_resblock_updown=args.classifier_resblock_updown,
                classifier_pool=args.classifier_pool,
            )
            self.classifier.load_state_dict(torch.load(args.classifier_path))
            if args.classifier_use_fp16:
                self.classifier.convert_to_fp16()
            self.classifier.eval()

        self.schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, self.diffusion
        )
        self.max_steps = len(self.schedule_sampler.weights())
        
        lmp = params_data.mask_params.lama_mask_params
        self.lmp_proba = params_data.mask_params.lama_mask_proba
        self.lama_mask_generator = MixedMaskGenerator(
            irregular_proba  =lmp.irregular_proba  , irregular_kwargs  =lmp.irregular_params,
            box_proba        =lmp.box_proba        , box_kwargs        =lmp.box_params,
            segm_proba       =lmp.segm_proba       , segm_kwargs       =lmp.segm_params,
            squares_proba    =lmp.squares_proba    , squares_kwargs    =lmp.squares_params,
            superres_proba   =lmp.superres_proba   , superres_kwargs   =lmp.superres_params,
            outpainting_proba=lmp.outpainting_proba, outpainting_kwargs=lmp.outpainting_params,
            # invert_proba=lmp.invert_proba,
        )

        self.train_metrics = get_metrics(params.metrics)(params, params_data)
        self.valid_metrics = get_metrics(params.metrics)(params, params_data)
        self.test_metrics = get_metrics(params.metrics)(params, params_data)

        self.log_strategy = get_log_strategy(params.logging, params_data)

        if self.args.use_gd_optimizer:
            self.mp_trainer = MixedPrecisionTrainer(
                model=self.model,
                use_fp16=args.use_fp16,
                fp16_scale_growth=args.fp16_scale_growth,
            )
            self.automatic_optimization = False
            self.took_step = None
        else:
            self.took_step = True

        if self.args.use_gd_ema:
            if self.args.use_gd_optimizer:
                ema_params = copy.deepcopy(self.mp_trainer.master_params)
                self.ema_scalar_vector_named_params = ema_params[0]
                self.ema_matrix_named_params = ema_params[1]
            else:
                ema_params = copy.deepcopy(self.model.named_parameters())
                self.ema_params = ema_params
        else:
            pass  # nothing to do, use ema plugin if needed
        
        self.diffusion.log_generation_time_steps()

    def configure_optimizers(self):         # todo check here
        optimizers_params = self.params.optimizer

        if self.args.use_gd_optimizer:
            model_params = [{"params": self.mp_trainer.master_params}]
        else:
            model_params = [{"params": self.model.parameters()}]

        match optimizers_params.optimizer:
            case "adam":
                optimizer = optim.Adam(
                    model_params,
                    lr=optimizers_params.learning_rate,
                    betas=optimizers_params.betas,
                    weight_decay=optimizers_params.weight_decay,
                )
            case "adamw":
                optimizer = optim.AdamW(
                    model_params,
                    lr=optimizers_params.learning_rate,
                    betas=optimizers_params.betas,
                    weight_decay=optimizers_params.weight_decay,
                )
            case "sgd":
                optimizer = optim.SGD(
                    model_params,
                    lr=optimizers_params.learning_rate,
                    momentum=optimizers_params.momentum,
                    weight_decay=optimizers_params.weight_decay,
                )
            case _:
                raise ValueError(
                    f"{optimizers_params.optimizer=} is not an available optimizer."
                )

        warmup_params = self.params.optimizer.learning_rate_warmup
        warmup_scheduler = None
        if warmup_params.use_scheduler:
            warmup_scheduler = LinearLR(
                optimizer=optimizer,
                start_factor=warmup_params.start_factor,
                end_factor=warmup_params.end_factor,
                total_iters=warmup_params.total_iters,
                last_epoch=warmup_params.last_epoch,
                verbose=warmup_params.verbose,
            )

        cosinus_params = self.params.optimizer.cosinus_params
        cosinus_scheduler = None
        if cosinus_params.use_scheduler:
            assert (
                self.trainer.max_epochs != -1 and self.trainer.max_steps == -1
            ), "Using cosine scheduler only white epoch and not step"

            cosinus_scheduler = CosineAnnealingLR(
                optimizer=optimizer,
                T_max=cosinus_params.T_max - (warmup_params.total_iters if warmup_params.use_scheduler else 0),
                eta_min=cosinus_params.eta_min,
                last_epoch=cosinus_params.last_epoch,
                verbose=cosinus_params.verbose,
            )

        scheduler = None
        if warmup_params.use_scheduler or cosinus_params.use_scheduler:
            list_s = []
            list_ms = []
            if warmup_params.use_scheduler:
                list_s.append(warmup_scheduler)
                list_ms.append(warmup_params.total_iters + 1)
            if cosinus_params.use_scheduler:
                list_s.append(cosinus_scheduler)
            scheduler = SequentialLR(
                optimizer=optimizer,
                schedulers=list_s,
                milestones=list_ms if len(list_s) > 1 else [],
                verbose=True,
            )

        if scheduler is not None:
            return [optimizer], [scheduler]
        else:
            return optimizer

    def log_g(self, train_stage: str, logged: str, value: Any, **kwargs):
        is_train = "train" in train_stage
        self.log(
            f"{train_stage}/{logged}",
            value,
            **kwargs,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )

    def training_step(self, batch, batch_idx):
        train_stage = "train"
        if self.args.use_gd_optimizer:
            self.mp_trainer.zero_grad()

        loss = self._step(batch, train_stage, batch_idx=batch_idx)

        if self.args.use_gd_optimizer:
            optimizer = self.optimizers().optimizer

            self.mp_trainer.backward(self, loss)
            self.mp_trainer.optimize(optimizer, lightning_module=self)

        if self.args.use_gd_ema:
            if self.took_step:
                self._update_ema()

        self.log_g(train_stage, "lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        return loss

    @jaxtyped
    @typechecker
    def get_t_weights_map_condition(
        self,
        batch_size: int, c: int, h: int, w: int,
        t_mode: Literal['vanilla', 'map_vanilla', 'one_per_pixel'],
        t_strategy: Literal['random', 'condition'],
        condition_prop: list[float],
        t_clean_value: int,
        patch_sizes: list[int],
        patch_weights: list[int],
        ) -> tuple[
        Int[torch.Tensor, "b"] | Int[torch.Tensor, "b 1 h w"],  # times
        Float[torch.Tensor, "b 1 1 1"] | Float[torch.Tensor, "b 1 h w"],  # weights
        Optional[Float[torch.Tensor, "b 3 h w"]],  # condition
    ]:
        assert len(patch_sizes) == len(patch_weights)
        # sample random patch size in the tuple
        patch_size = [random.choices(population=patch_sizes, weights=patch_weights)[0] for _ in range(batch_size)]

        times = []
        weights = []
        condition_pixels = []
        for bi in range(batch_size):
            time, weight, condition_pixel = self._get_t_weights_map_condition(
                batch_size=1, c=c, h=h, w=w,
                t_mode=t_mode,
                t_strategy=t_strategy,
                condition_prop=condition_prop,
                t_clean_value=t_clean_value,
                patch_size=patch_size[bi],
            )
            times.append(time)
            weights.append(weight)
            condition_pixels.append(condition_pixel)

        times = torch.cat(times, dim=0)
        weights = torch.cat(weights, dim=0)
        if any([i is None for i in condition_pixels]):
            assert all([i is None for i in condition_pixels])
            condition_pixels = None
        else:
            condition_pixels = torch.cat(condition_pixels, dim=0)

        return times, weights, condition_pixels

    @jaxtyped
    @typechecker
    def _get_t_weights_map_condition(
        self,
        batch_size: int,
        c: int,
        h: int,
        w: int,
        t_mode: Literal["vanilla", "map_vanilla", "one_per_pixel"],
        t_strategy: Literal["random", "condition"],
        condition_prop: list[float],
        t_clean_value: int,
        patch_size: int,
    ) -> tuple[
        Int[torch.Tensor, "b"] | Int[torch.Tensor, "b 1 h w"],  # times
        Float[torch.Tensor, "b 1 1 1"] | Float[torch.Tensor, "b 1 h w"],  # weights
        Optional[Float[torch.Tensor, "b 3 h w"]],  # condition
    ]:
        hpatch = h // patch_size
        nb_patchs = hpatch * hpatch

        match t_mode:
            case "vanilla":
                t, weights = self.schedule_sampler.sample(
                    batch_size=batch_size, device=self.device
                )
                weights = weights.reshape(batch_size, 1, 1, 1)
            case "map_vanilla":
                t, weights = self.schedule_sampler.sample(
                    batch_size=batch_size, device=self.device
                )
                t = repeat(t, "b -> b 1 hpatch1 hpatch2", hpatch1=hpatch, hpatch2=hpatch)
                weights = repeat(weights, "b -> b 1 h w", h=h, w=w)
            case "one_per_pixel":
                t, weights = self.schedule_sampler.sample(
                    batch_size=batch_size * hpatch * hpatch, device=self.device
                )
                t = t.reshape(batch_size, 1, hpatch, hpatch)
                weights = weights.reshape(batch_size, 1, hpatch, hpatch)
                weights = patch_to_full(map=weights, patch_size=patch_size)
            case _:
                raise ValueError(f"{t_mode=} is not an available t_mode.")

        condition_pixel = None
        if t.dim() == 4:
            match t_strategy:
                case "random":
                    condition_pixel = None
                case "condition":
                    assert t.dim() == 4
                    if self.lmp_proba > random.random():
                        lama_mask = [self.lama_mask_generator(img=torch.ones([c, h, w]), iter_i=None, raw_image=None) for _ in range(batch_size)]
                        condition_pixel = torch.stack([torch.tensor(i) for i in lama_mask]).to(self.device)
                        condition_pixel = condition_pixel == 1  # cast to bool
                        condition_pixel = ~condition_pixel  # in lama, black is keep, us is white is keep

                        # if we are using lama mask, we must reset t to not take into account the patches
                        # put back t to the full dimension, t was in patch dimension, and the mask is in full dimension
                        t = patch_to_full(t, patch_size=patch_size)

                        # now we change the t according to the condition proba
                        t = t.clone()
                        t[condition_pixel] = t_clean_value
                        condition_pixel = condition_pixel.float()
                    else:
                        if len(condition_prop) == 1:  # same proba for all
                            condition_prop = torch.full([batch_size, 1], condition_prop[0], device=t.device)
                        elif len(condition_prop) == 2:  # proba is between min and max
                            min_proba, max_proba = condition_prop
                            condition_prop = torch.rand(batch_size, device=t.device) * (max_proba - min_proba) + min_proba
                            condition_prop = condition_prop.reshape(batch_size, 1)
                        else:
                            raise ValueError(f"{condition_prop=} should be of length 1 or 2.")

                        len_keep_per_batch = condition_prop * nb_patchs
                        condition_pixel = torch.full_like(t, False, dtype=torch.bool, device=t.device)
                        for bi in range(batch_size):
                            len_keep_for_this_batch = len_keep_per_batch[bi].int().item()  # how much we keep for the current element
                            len_keep_for_this_batch = min(len_keep_for_this_batch, nb_patchs - 1)  # cannot keep all patchs, one has to be removed

                            noise = torch.rand([nb_patchs], device=self.device)  # noise to sample random patches
                            ids_shuffle = torch.argsort(noise)  # ascend: small is condition, large is forward diffusion
                            ids_condition = ids_shuffle[:len_keep_for_this_batch]

                            condition_pixel_curr = condition_pixel[bi].flatten()
                            condition_pixel_curr[ids_condition] = True
                            condition_pixel_curr = condition_pixel_curr.reshape(condition_pixel[bi].shape)
                            condition_pixel[bi] = condition_pixel_curr

                            # now we change the t according to the condition proba
                            t = t.clone()
                            t[condition_pixel] = t_clean_value
                            condition_pixel = condition_pixel.float()
                case _:
                    raise ValueError(f"{t_strategy=} is not an available t_strategy.")

        if t.dim() != 1 and t.shape != (batch_size, 1, h, w):
            # put back t to the full dimension, t was in patch dimension
            t = patch_to_full(t, patch_size=patch_size)
        if condition_pixel is not None and condition_pixel.shape[2:] != t.shape[2:]:
            # put condition pixel wise if it was patch wise
            condition_pixel = patch_to_full(condition_pixel, patch_size=patch_size)
        if condition_pixel is not None and condition_pixel.shape[1] == 1:  # broadcast condition to the image nb of channels
            condition_pixel = condition_pixel.repeat(1, c, 1, 1)

        return t, weights, condition_pixel

    def _step(
        self,
        batch: tuple[
            Int[torch.Tensor, "b"],  # idx
            Float[torch.Tensor, "b c h w"],  # img
            dict,  # dict info
            Float[torch.Tensor, "b 1 h w"],  # mask
        ],
        stage_prefix: str,
        batch_idx: int,
    ) -> torch.Tensor:
        stage = self.get_stage()
        idx, data, dict_info, mask = batch

        batch_size, c, h, w = data.shape

        t, weights, condition_pixel = self.get_t_weights_map_condition(
            batch_size=batch_size, c=c, h=h, w=w,
            t_mode=self.args.t_mode,
            t_strategy=self.args.t_strategy,
            condition_prop=self.args.condition_proba,
            t_clean_value=self.args.t_clean_value,
            patch_sizes=self.args.patch_size_train,
            patch_weights=self.args.patch_weight_train,
        )

        losses = self.diffusion.training_losses(
            model=self.model,
            x_start=data,
            t=t,
            model_kwargs=dict_info,
        )
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )
        assert losses["loss"].dim() == 4
        if self.args.t_strategy == "condition" and not self.args.learn_the_condition:
            assert condition_pixel.shape == losses["loss"].shape
            not_condition_pixel = 1 - condition_pixel
            number_of_target_pixels = not_condition_pixel.sum()
            loss = (losses["loss"] * weights * not_condition_pixel).sum() / number_of_target_pixels
        else:
            loss = (losses["loss"] * weights).mean()

        # Log
        self.log_g(f"{stage_prefix}/step", "loss", loss.item(), prog_bar=True)

        # Log image from training step
        if is_logging_time(
                self.params.logging.log_steps,
                current_epoch=self.current_epoch,
                batch_idx=batch_idx,
                stage=stage,
        ):
            self.log_strategy.log_train(
                stage_prefix=f"{stage_prefix}/step",
                plMod=self,
                batch=data,
                ts=t,
                x0pred=losses['pred_xstart'],
                xt=losses['xt'],
                idx=idx,
                batch_idx=batch_idx,
            )

        # Log image generate uncond
        if is_logging_time(
            self.params.logging.log_generate_uncond,
            current_epoch=self.current_epoch,
            batch_idx=batch_idx,
            stage=stage,
        ):
            self.log_strategy.log_generate_uncond(
                stage_prefix=f"{stage_prefix}/generate/uncond",
                plMod=self,
                batch=data,
                batch_idx=batch_idx,
                idx=idx,
            )
        # Log image generate cond
        if is_logging_time(
                self.params.logging.log_generate_cond,
                current_epoch=self.current_epoch,
                batch_idx=batch_idx,
                stage=stage,
        ):
            self.log_strategy.log_generate_cond(
                stage_prefix=f"{stage_prefix}/generate/cond",
                plMod=self,
                batch=data,
                mask=mask,
                dict_info=dict_info,
                batch_idx=batch_idx,
                idx=idx,
            )
        # Log image generate diversity
        if is_logging_time(
            self.params.logging.log_generate_diversity,
            current_epoch=self.current_epoch,
            batch_idx=batch_idx,
            stage=stage,
            ):
            self.log_strategy.log_generate_diversity(
                stage_prefix=f'{stage_prefix}/generate/diversity',
                plMod =self,
                batch=data,
                mask=mask,
                dict_info =dict_info,
                batch_idx =batch_idx,
                idx= idx,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        ema_params = self.params.optimizer.ema
        is_ema = ema_params.use and not ema_params.validate_original_weights
        perform_double_pass = (
            ema_params.use
            and ema_params.validate_original_weights
            and ema_params.perform_double_validation
        )

        if not perform_double_pass:
            self.set_ema()
            addition = ("/ema" if is_ema else "/van") if ema_params.use else ""
            self._step(tuple(batch), f"valid{addition}", batch_idx=batch_idx)
            self.unset_set_ema()
        else:
            self._step(tuple(batch), "valid/van", batch_idx=batch_idx)

            self.ema_swap_model_weigts()
            self._step(tuple(batch), "valid/ema", batch_idx=batch_idx)
            self.ema_swap_model_weigts()

    def test_step(self, batch, batch_idx):
        ema_params = self.params.optimizer.ema
        is_ema = ema_params.use and not ema_params.validate_original_weights
        perform_double_pass = (
            ema_params.use
            and ema_params.validate_original_weights
            and ema_params.perform_double_validation
        )

        if not perform_double_pass:
            self.set_ema()
            addition = ("/ema" if is_ema else "/van") if ema_params.use else ""
            self._step(tuple(batch), f"test{addition}", batch_idx=batch_idx)
            self.unset_set_ema()
        else:
            self._step(tuple(batch), "test/van", batch_idx=batch_idx)

            self.ema_swap_model_weigts()
            self._step(tuple(batch), "test/ema", batch_idx=batch_idx)
            self.ema_swap_model_weigts()

    # region EMA FUNCTIONS
    def _update_ema(self):
        if not self.args.use_gd_ema:
            return

        if self.args.use_gd_optimizer:
            rate = self.params.optimizer.ema.decay
            ema_params = [self.ema_scalar_vector_named_params, self.ema_matrix_named_params]
            for targ, src in zip(ema_params, self.mp_trainer.master_params):
                targ = targ.to(src.device)
                targ.detach().mul_(rate).add_(src, alpha=1 - rate)
        else:
            rate = self.params.optimizer.ema.decay
            ema_params = self.ema_params
            for (targ_name, targ), (src_name, src) in zip(ema_params, self.model.named_parameters()):
                targ = targ.to(src.device)
                targ.detach().mul_(rate).add_(src, alpha=1 - rate)

    def swap_tensors(self, tensor1, tensor2):
        tmp = torch.empty_like(tensor1)
        tmp.copy_(tensor1)
        tensor1.copy_(tensor2)
        tensor2.copy_(tmp)

    def ema_swap_model_weigts(self):
        if self.ema is not None:  # use callback function
            self.ema.swap_model_weights(self.trainer)
            return

        if self.args.use_gd_optimizer:
            model_state_dict = self.model.state_dict()
            ema_params = [self.ema_scalar_vector_named_params, self.ema_matrix_named_params]
            ema_state_dict = self.mp_trainer.master_params_to_state_dict(ema_params)
            for param, ema_param in zip(model_state_dict.items(), ema_state_dict.items()):
                name1, param1 = param
                name2, param2 = ema_param
                assert name1 == name2
                param2 = param2.to(param1.device)
                self.swap_tensors(param1, param2)
        else:
            model_named_params = self.model.named_parameters()
            ema_params = self.ema_params
            for (params_name, param1), (ema_name, ema_param) in zip(model_named_params, ema_params):
                assert params_name == ema_name
                param2 = ema_param.to(param1.device)
                self.swap_tensors(param1, param2)

    def set_ema(self):
        """
        Function run before validation step or test step in the case of not performing double pass
        """
        if self.ema is not None:  # if we use our EMA callback, nothing to do
            return

        # if we need to compute the step on the EMA, need to swap the parameters
        ema_params = self.params.optimizer.ema
        is_ema = ema_params.use and not ema_params.validate_original_weights
        if is_ema:
            self.ema_swap_model_weigts()

    def unset_set_ema(self):
        """
        Function run after validation step or test step in the case of not performing double pass
        """
        if self.ema is not None:  # if we use our EMA callback, nothing to do
            return

        # if we need to compute the step on the EMA, need to swap the parameters back
        ema_params = self.params.optimizer.ema
        is_ema = ema_params.use and not ema_params.validate_original_weights
        if is_ema:
            self.ema_swap_model_weigts()
    # endregion END EMA FUNCTIONS
    ########################################################################################################################
    @jaxtyped
    @typechecker
    def cond_fn(
        self,
        x: Float[torch.Tensor, "b c h w"],
        t: Int[torch.Tensor, "b"],
        y: Int[torch.Tensor, "b"],
        gt: Optional[Float[torch.Tensor, "b 3 h w"]] = None,
        mask: Optional[Int[torch.Tensor, "b 1 h w"]] = None,  # in like [0, 1000] rescaled
        original_t: Optional[Int[torch.Tensor, "b"]] = None,  # in like [0, 250] not rescaled
    ):
        if gt is not None and mask is not None and self.args.t_mode != "vanilla":  # important for ours, where there is different level of noise
            # need to apply forward diffusion to gt, make it match the x noise level, then combine with mask and x
            diffusion = self.diffusion
            alpha_cumprod = _extract_into_tensor(
                diffusion.alphas_cumprod, original_t, x.shape,
            )
            noised_gt = gt * torch.sqrt(alpha_cumprod) + torch.randn_like(x) * torch.sqrt(1 - alpha_cumprod)
            x = noised_gt * mask + (1 - mask) * x
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = self.classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * self.args.classifier_scale
    
    @jaxtyped
    @typechecker
    def model_fn(
        self,
        x: Float[torch.Tensor, "b c h w"],
        t: Int[torch.Tensor, "b"],
        y: Int[torch.Tensor, "b"],
    ):
        return self.model(x, t, y if self.args.class_cond else None)

    @jaxtyped
    @typechecker
    def generate_samples(
        self,
        batch_size: int,
        mask: Optional[Int[torch.Tensor, "b 1 h w"]] = None,
        gt: Optional[Float[torch.Tensor, "b 3 h w"]] = None,
        dict_info: Optional[dict] = None,
    ):
        args = self.args
        model_kwargs = {"mask": mask, "gt": gt}
        if args.class_cond:
            if not self.args.use_random_class and dict_info is not None and 'y' in dict_info.keys():
                classes = dict_info['y']
            else:
                classes = torch.randint(
                    low=0,
                    high=NUM_CLASSES,
                    size=(batch_size,),
                    device=self.device,
                )
            model_kwargs["y"] = classes
        else:
            classes = None
        sample_fn = (
            self.diffusion.p_sample_loop
            if not args.use_ddim
            else self.diffusion.ddim_sample_loop  # TODO support ddim sample loop
        )
        start_time = time.time()
        sample, logged_data = sample_fn(
            model=self.model,
            shape=(batch_size, 3, args.image_size, args.image_size),
            noise=None,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=self.cond_fn if self.args.use_classifier else None,
        )
        end_time = time.time()
        ellapsed = end_time - start_time
        dict_info = {"ellapsed": ellapsed}
        sample = sample.clamp(-1, 1).contiguous()
        if self.params.logging.combine_sample_with_mask and mask is not None:
            sample = gt * mask + (1 - mask) * sample

        return sample, classes, logged_data, dict_info

    def get_stage(self) -> str:
        if self.trainer.training:
            return "train"
        elif self.trainer.validating or self.trainer.sanity_checking:
            return "valid"
        elif self.trainer.testing or self.trainer.predicting:
            return "test"
        else:
            raise Exception("Stage not supported.")

    def get_metric_object(self):
        return {
            "train": self.train_metrics,
            "valid": self.valid_metrics,
            "test": self.test_metrics,
        }[self.get_stage()]
