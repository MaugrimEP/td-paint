import datetime
import math
import time
from os import path
from typing import Union

import pytorch_lightning as pl
import torch
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from pytorch_lightning import Callback, LightningModule, Trainer
from torch.utils.data import ConcatDataset
from torchmetrics.image import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm

from conf.distribution_params import DistributionDistanceParams


class IDCallback(Callback):
    def __init__(
        self,
        params: DistributionDistanceParams,
        train_dataset,
        valid_dataset,
        test_dataset,
        nb_gpus: int,
        nb_nodes: int,
    ):
        super().__init__()
        self.p = params
        self.nb_nodes = nb_nodes
        self.nb_gpus = nb_gpus
        self.nb_to_see_per_gpu = math.ceil(
            self.p.number_to_generate / (nb_gpus * nb_nodes)
        )
        self.running_compute_freq_per_gpu = math.ceil(
            self.p.running_compute_freq / (nb_gpus * nb_nodes)
        )
        self.running_compute_freq_last = 0
        self.current_examples_seen = 0
        print(
            f"\n[id.__init__] number of fakes samples to generate per GPU: {self.nb_to_see_per_gpu=}"
        )

        self.full_dataset = ConcatDataset([train_dataset, valid_dataset, test_dataset])
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.full_dataset,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            pin_memory=params.pin_memory,
            prefetch_factor=params.prefetch_factor,
        )

        self.fid = FrechetInceptionDistance(
            feature=self.p.fid_dims,
            reset_real_features=False,
            normalize=True,
        )
        self.kid = KernelInceptionDistance(
            feature=params.kid_feature,
            subsets=params.kid_subsets,
            subset_size=params.kid_subset_size,
            degree=params.kid_degree,
            gamma=params.kid_gamma,
            coef=params.kid_coef,
            reset_real_features=False,
            normalize=True,
        )
        """
        If argument normalize is True images are expected to be dtype float and have values in the [0,1] range, else if 
        normalize is set to False images are expected to have dtype uint8 and take values in the [0, 255] range.
        """

        # KID INIT
        if self.p.kid_load_initialization_path is not None and path.exists(self.p.kid_load_initialization_path):
            # LOAD kid REAL STATS FROM A FILE
            print(f"\nkid Callback START Loading real stats from {self.p.kid_load_initialization_path}")
            loaded = torch.load(self.p.kid_load_initialization_path)
            self.kid.real_features = loaded["real_features"]
            print("\nEND Loading real stats")

        elif self.p.init:
            # COMPUTE kid REAL STATS AND SAVE THEM
            self.init_kid_real_stats()

        else:
            # EITHER SHOULD LOAD STATS OR INIT THEM
            raise ValueError(f"Path {self.p.kid_load_initialization_path=} does not exist and {self.p.init=} is False")

        # FID INIT
        if self.p.fid_load_initialization_path is not None and path.exists(self.p.fid_load_initialization_path):
            # LOAD FID REAL STATS FROM A FILE
            print(
                f"\nFID Callback START Loading real stats from {self.p.fid_load_initialization_path}"
            )
            loaded = torch.load(self.p.fid_load_initialization_path)
            self.fid.real_features_sum = loaded["real_features_sum"]
            self.fid.real_features_cov_sum = loaded["real_features_cov_sum"]
            self.fid.real_features_num_samples = loaded["real_features_num_samples"]
            print("\nEND Loading real stats")

        elif self.p.init:
            # COMPUTE FID REAL STATS AND SAVE THEM
            self.init_fid_real_stats()

        else:
            # EITHER SHOULD LOAD STATS OR INIT THEM
            raise ValueError(f"Path {self.p.fid_load_initialization_path=} does not exist and {self.p.init=} is False")

    def init_fid_real_stats(self):
        """
        Init the fid callback on the real dataset and save the stats to the disk.
        """
        assert (
            self.nb_gpus == 1 and self.nb_nodes == 1
        ), "fid init_real_stats should be run on a single GPU, single Node"
        path_without_file = path.dirname(self.p.fid_load_initialization_path)
        if not path.exists(path_without_file):
            raise ValueError(f"Path {path_without_file=} does not exist")

        params = self.p
        dataloader = self.dataloader

        self.fid = self.fid.cuda()
        print("\n[id:init_fid_real_stats] START Computing real stats...")
        i = 0
        for i, batch in enumerate(tqdm(dataloader), start=1):
            batch = self.batch_to_cuda(batch)
            self.fid.update(self.normalize_batch(tuple(batch)), real=True)
        print(
            f"\n[id:init_fid_real_stats] END Computing real stats, seen {i * params.batch_size=} examples during {i=} batches"
        )
        self.fid.sync()
        self.fid = self.fid.cpu()

        state = {
            "real_features_sum": self.fid.real_features_sum,
            "real_features_cov_sum": self.fid.real_features_cov_sum,
            "real_features_num_samples": self.fid.real_features_num_samples,
        }
        torch.save(state, self.p.fid_load_initialization_path)
        print(
            f"\n[id:init_fid_real_stats] END Saving real stats at {self.p.fid_load_initialization_path}"
        )

    def batch_to_cuda(self, batch):
        idx, data, dict_info, mask = batch
        data = data.cuda()
        return idx, data, dict_info, mask

    def init_kid_real_stats(self):
        """
        Init the kid callback on the real dataset and save the stats to the disk.
        """
        assert (
            self.nb_gpus == 1 and self.nb_nodes == 1
        ), "kid init_real_stats should be run on a single GPU, single Node"
        path_without_file = path.dirname(self.p.kid_load_initialization_path)
        if not path.exists(path_without_file):
            raise ValueError(f"Path {path_without_file=} does not exist")

        params = self.p
        dataloader = self.dataloader

        self.kif = self.kid.cuda()
        print("\n[id:init_kid_real_stats] START Computing real stats...")
        i = 0
        for i, batch in enumerate(tqdm(dataloader), start=1):
            batch = self.batch_to_cuda(batch)
            self.kid.update(self.normalize_batch(tuple(batch)), real=True)
        print(
            f"\n[id:init_kid_real_stats] END Computing real stats, seen {i * params.batch_size=} examples during {i=} batches"
        )
        self.kid.sync()
        self.kid = self.kid.cpu()

        state = {
            "real_features": self.kid.real_features,
        }
        torch.save(state, self.p.kid_load_initialization_path)
        print(
            f"\n[id:init_real_stats] END Saving real stats at {self.p.kid_load_initialization_path}"
        )

    def _on_all_batch_end(self, pl_module, trainer) -> tuple[float, int, int]:
        """
        Used on OnValidationEpochEnd and OnTestEpochEnd
        """
        self.kid.to(pl_module.device)
        self.fid.to(pl_module.device)
        size_to_generate = self.nb_to_see_per_gpu
        size_to_generate -= self.current_examples_seen  # remove possibles seen examples during training

        has_swapped = False
        if self.p.compute_on_ema and not pl_module.ema.is_ema:
            pl_module.ema.swap_model_weights(trainer)
            has_swapped = True
        print(f"Currently running on {pl_module.ema.is_ema=}")
        print(
            f"\n[id:_on_all_batch_end] START Computing fake stats for {size_to_generate=} samples of batch size {self.p.batch_size} per GPU..."
        )
        seen_examples = 0
        pbar = tqdm(total=size_to_generate)
        start_time = time.time()
        while True:
            if seen_examples >= size_to_generate:
                break
            for batch in self.dataloader:
                if seen_examples >= size_to_generate:
                    break

                idx, data, dict_info, mask = batch
                batch_size = data.shape[0]
                with torch.no_grad():
                    sample, classes, logged_data = pl_module.generate_samples(
                        batch_size=batch_size,
                        mask=None,
                        gt=None,
                    )
                self.kid.update(self.normalize_sample(sample), real=False)
                self.fid.update(self.normalize_sample(sample), real=False)
                self.check_compute_running(epoch=pl_module.current_epoch, logger=pl_module, trainer=trainer)

                seen_examples += batch_size
                pbar.update(batch_size)
        pbar.close()
        end_time = time.time()
        string_elapsed_time = str(datetime.timedelta(seconds=end_time - start_time))
        print(
            f"\n[id:_on_all_batch_end] END Computing fake stats: {string_elapsed_time=} : {size_to_generate=} {self.p.batch_size=}, fid:{self.fid.compute().item()}, kid:{self.kid.compute()}"
        )

        if has_swapped:
            pl_module.ema.swap_model_weights(trainer)

        examples_seens_during_step = self.current_examples_seen

        # RETURN
        # compute time,
        # number example generated during this function
        # number example generated during steps
        return end_time - start_time, seen_examples, examples_seens_during_step

    def process_batch(
        self,
        batch_fakes: Float[torch.Tensor, "b 3 h w"],
    ) -> None:
        """
        Used for the train/valid/test steps when we conditionally generate samples anyway
        """
        raise NotImplementedError("process_batch is not implemented")
        if self.current_examples_seen >= self.nb_to_see_per_gpu:
            return
        # otherwise we add the current batch for the kid

        nb_examples = self.get_batch_size(batch_fakes)

        self.current_examples_seen += nb_examples
        fakes = self.normalize_batch(batch_fakes)  # TODO change here
        self.kid.to(fakes.device)
        self.kid.update(fakes, real=False)
        self.fid.to(fakes.device)
        self.fid.update(fakes, real=False)

    def is_time_to_compute_valid(self, current_epoch: int) -> bool:
        return (
            "valid" in self.p.stages
            and current_epoch % self.p.check_frequency == 0
            and (current_epoch != 0 if not self.p.compute_first else True)
        )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if "valid" not in self.p.stages:
            return

        if not self.is_time_to_compute_valid(current_epoch=pl_module.current_epoch):
            return

        elapsed_time, n_generated, example_seen = self._on_all_batch_end(
            pl_module=pl_module, trainer=trainer
        )

        trainer.strategy.barrier()
        kid_mean, kid_std = self.kid.compute()
        pl_module.log("valid/kid/mean", kid_mean, sync_dist=True)
        pl_module.log("valid/kid/std", kid_std, sync_dist=True)
        pl_module.log("valid/kid/elapsed_time", elapsed_time, sync_dist=True, reduce_fx=sum)
        pl_module.log("valid/kid/n_generated", n_generated, sync_dist=True, reduce_fx=sum)
        pl_module.log("valid/kid/seen_valid", example_seen, sync_dist=True, reduce_fx=sum)
        pl_module.log("valid/kid/total", example_seen + n_generated, sync_dist=True, reduce_fx=sum)

        pl_module.log("valid/fid/value", self.fid.compute(), sync_dist=True)
        pl_module.log("valid/fid/elapsed_time", elapsed_time, sync_dist=True, reduce_fx=sum)
        pl_module.log("valid/fid/n_generated", n_generated, sync_dist=True, reduce_fx=sum)
        pl_module.log("valid/fid/seen_valid", example_seen, sync_dist=True, reduce_fx=sum)
        pl_module.log("valid/fid/total", example_seen + n_generated, sync_dist=True, reduce_fx=sum)
        trainer.strategy.barrier()
        self.reset()
        trainer.strategy.barrier()

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if "test" not in self.p.stages:
            return

        elapsed_time, n_generated, example_seen = self._on_all_batch_end(
            pl_module=pl_module, trainer=trainer
        )

        trainer.strategy.barrier()
        kid_mean, kid_std = self.kid.compute()
        pl_module.log("test/kid/mean", kid_mean, sync_dist=True)
        pl_module.log("test/kid/std", kid_std, sync_dist=True)
        pl_module.log("test/kid/elapsed_time", elapsed_time, sync_dist=True, reduce_fx=sum)
        pl_module.log("test/kid/n_generated", n_generated, sync_dist=True, reduce_fx=sum)
        pl_module.log("test/kid/seen_valid", example_seen, sync_dist=True, reduce_fx=sum)
        pl_module.log("test/kid/total", example_seen + n_generated, sync_dist=True, reduce_fx=sum)

        pl_module.log("test/fid/value", self.fid.compute(), sync_dist=True)
        pl_module.log("test/fid/elapsed_time", elapsed_time, sync_dist=True, reduce_fx=sum)
        pl_module.log("test/fid/n_generated", n_generated, sync_dist=True, reduce_fx=sum)
        pl_module.log("test/fid/seen_valid", example_seen, sync_dist=True, reduce_fx=sum)
        pl_module.log("test/fid/total", example_seen + n_generated, sync_dist=True, reduce_fx=sum)
        trainer.strategy.barrier()
        self.reset()
        trainer.strategy.barrier()

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        trainer.strategy.barrier()
        self.reset()
        trainer.strategy.barrier()
        
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        trainer.strategy.barrier()
        self.reset()
        trainer.strategy.barrier()

    @jaxtyped(typechecker=typechecker)
    def normalize_batch(
        self,
        batch: tuple[
            Int[torch.Tensor, "b"],  # idx
            Float[torch.Tensor, "b c h w"],  # img
            dict,  # dict info
            Int[torch.Tensor, "b 1 h w"],  # mask
        ],
    ) -> Float[torch.Tensor, "b 3 h w"]:
        idx, data, dict_info, mask = batch
        data = self.normalize_sample(data)
        return data
    
    @jaxtyped(typechecker=typechecker)
    def normalize_sample(
        self,
        sample: Float[torch.Tensor, "b 3 h w"],
    ) -> Float[torch.Tensor, "b 3 h w"]:
        """
        id callback expects images in [0,1] of type float if normalize is True
        Also perform the different transformation for the different datasets
        """
        data = sample / 2 + 0.5
        return data

    def check_compute_running(self, epoch: int, logger, trainer):
        """
        Used to compute the running kid stats every running_compute_freq_per_gpu examples
        """
        if not self.p.running_compute:
            return
        if (
            self.current_examples_seen - self.running_compute_freq_last
            <= self.running_compute_freq_per_gpu
        ):
            return
        
        stage = logger.get_stage()

        self.running_compute_freq_last = self.current_examples_seen
        trainer.strategy.barrier()
        curr_kid_mean, curr_kid_std = self.kid.compute()
        logger.log_dict(
            {
                f"{stage}/running/kid/mean": curr_kid_mean,
                f"{stage}/running/kid/std": curr_kid_std,
                f"{stage}/running/kid/seen": self.current_examples_seen,
                f"{stage}/running/kid/epoch": epoch,
                f"{stage}/running/fid/value": self.fid.compute(),
                f"{stage}/running/fid/seen": self.current_examples_seen,
                f"{stage}/running/fid/epoch": epoch,
            }
        )
        trainer.strategy.barrier()

    def reset(self):
        self.current_examples_seen = 0
        self.running_compute_freq_last = 0
        self.kid.reset()
        self.fid.reset()
