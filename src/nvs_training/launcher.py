"""
Training launcher for novel view synthesis.

This module provides the main training and evaluation launcher
for the KoNet model with GeoPE attention mechanism.
"""

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
from einops import rearrange
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from ..nvs_data import EvalDataset, TrainDataset
from ..nvs_models import Camera, KoNetDecoderOnlyModel, KoNetDecoderOnlyModelConfig
from ..geope_utils.functional import random_SO3
from ..geope_utils.runner import Launcher, LauncherConfig, nested_to_device
from .losses import CombinedLoss
from .utils import write_tensor_to_image

import torch_npu
from torch_npu.contrib import transfer_to_npu


@dataclass
class KoNetLauncherConfig(LauncherConfig):
    """Configuration for KoNet training launcher."""
    
    # Dataset configuration
    dataset_patch_size: int = 256
    dataset_supervise_views: int = 6
    dataset_batch_scenes: int = 4
    train_zoom_factor: float = 1.0
    random_zoom: bool = False

    # Optimization configuration
    use_torch_compile: bool = False

    # Model configuration
    model_config: Any = field(
        default_factory=lambda: KoNetDecoderOnlyModelConfig(ref_views=2)
    )

    # Training configuration
    max_steps: int = 1000000  # Override parent
    ckpt_every: int = 1000  # Override parent
    print_every: int = 100
    visual_every: int = 100
    lr: float = 4e-4
    warmup_steps: int = 2500

    # Loss configuration
    perceptual_loss_weight: float = 0.5

    # Evaluation configuration
    test_every: int = 10000  # Override parent
    test_n: Optional[int] = None
    test_input_views: int = 2
    test_supervise_views: int = 3
    test_zoom_factor: tuple[float, ...] = (1.0,)
    aug_with_world_origin_shift: bool = False
    aug_with_world_rotation: bool = False

    # Video rendering
    render_video: bool = False
    test_index_fp: Optional[str] = None


class KoNetLauncher(Launcher):
    """Training launcher for KoNet model."""
    
    config: KoNetLauncherConfig

    def preprocess(
        self, 
        data: Dict, 
        input_views: int
    ) -> Tuple[Tensor, Camera, Camera, Tensor]:
        """
        Preprocess training data.
        
        Args:
            data: Raw data dictionary
            input_views: Number of input reference views
            
        Returns:
            Tuple of (ref_images, ref_cameras, tar_cameras, tar_images)
        """
        data = nested_to_device(data, self.device)

        images = data["image"] / 255.0
        intrinsics = data["K"]
        camtoworlds = data["camtoworld"]
        image_paths = data["image_path"]
        
        assert images.ndim == 5, f"Expected 5D images, got {images.shape}"
        batch_size, num_views, height, width, channels = images.shape

        # Apply data augmentation
        aug = torch.eye(4, device=self.device).repeat(batch_size, 1, 1)
        if self.config.aug_with_world_origin_shift:
            shifts = torch.randn((batch_size, 3), device=self.device)
            aug[:, :3, 3] = shifts
        if self.config.aug_with_world_rotation:
            rotations = random_SO3((batch_size,), device=self.device)
            aug[:, :3, :3] = rotations
        camtoworlds = torch.einsum("bij,bvjk->bvik", aug, camtoworlds)

        # Split into reference and target views
        ref_images = images[:, :input_views]
        tar_images = images[:, input_views:]
        ref_cameras = Camera(
            K=intrinsics[:, :input_views],
            camtoworld=camtoworlds[:, :input_views],
            width=width,
            height=height,
        )
        tar_cameras = Camera(
            K=intrinsics[:, input_views:],
            camtoworld=camtoworlds[:, input_views:],
            width=width,
            height=height,
        )
        ref_paths = np.array(image_paths)[:input_views]
        tar_paths = np.array(image_paths)[input_views:]

        return ref_images, ref_cameras, tar_cameras, tar_images

    def train_initialize(self) -> Dict[str, Any]:
        """Initialize training components."""
        # Setup data
        scenes = sorted(glob.glob("/data/re10k/train/metadata/*.json"))
        dataset = TrainDataset(
            scenes,
            patch_size=self.config.dataset_patch_size,
            zoom_factor=self.config.train_zoom_factor,
            random_zoom=self.config.random_zoom,
            supervise_views=self.config.dataset_supervise_views,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.dataset_batch_scenes,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        self.logging_on_master(f"Total scenes: {len(dataset)}")

        # Setup model
        model = KoNetDecoderOnlyModel(self.config.model_config).to(self.device)
        if self.config.use_torch_compile:
            model = torch.compile(model)
        
        # Setup loss function
        loss_fn = CombinedLoss(
            mse_weight=1.0,
            perceptual_weight=self.config.perceptual_loss_weight,
            perceptual_net_type="alex"
        ).to(self.device)
        
        print(f"Model is initialized in rank {self.world_rank}")

        # Setup optimizer
        params_decay = {
            "params": [p for n, p in model.named_parameters() if "norm" not in n],
            "weight_decay": 0.5,
        }
        params_no_decay = {
            "params": [p for n, p in model.named_parameters() if "norm" in n],
            "weight_decay": 0.0,
        }
        optimizer = torch.optim.AdamW(
            [params_decay, params_no_decay], lr=self.config.lr, betas=(0.9, 0.95)
        )

        # Setup scheduler
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.01,
                    total_iters=self.config.warmup_steps,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config.max_steps - self.config.warmup_steps,
                ),
            ]
        )

        # Setup metrics
        psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        lpips_fn = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(self.device)

        state = {
            "model": model,
            "loss_fn": loss_fn,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "dataloader": dataloader,
            "dataiter": iter(dataloader),
            "ssim_fn": ssim_fn,
            "psnr_fn": psnr_fn,
            "lpips_fn": lpips_fn,
        }
        print(f"Launcher(train) is initialized in rank {self.world_rank}")
        return state

    def train_iteration(
        self, 
        step: int, 
        state: Dict[str, Any], 
        acc_step: int, 
        *args, 
        **kwargs
    ) -> Tensor:
        """Execute one training iteration."""
        dataloader = state["dataloader"]
        dataiter = state["dataiter"]
        loss_fn = state["loss_fn"]
        model = state["model"]
        model.train()

        try:
            data = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader)
            data = next(dataiter)
            state["dataiter"] = dataiter

        input_views = data["K"].shape[1] - self.config.dataset_supervise_views
        ref_images, ref_cameras, tar_cameras, tar_images = self.preprocess(
            data, input_views=input_views
        )

        # Forward pass
        with torch.amp.autocast("cuda", enabled=self.config.amp, dtype=self.amp_dtype):
            outputs = model(ref_images, ref_cameras, tar_cameras)
            outputs = torch.sigmoid(outputs)
            loss = loss_fn(outputs, tar_images)

        # Logging and visualization
        if (
            self.config.visual_every > 0
            and step % self.config.visual_every == 0
            and self.world_rank == 0
            and acc_step == 0
        ):
            write_tensor_to_image(
                rearrange(outputs, "b v h w c-> (b h) (v w) c"),
                f"{self.visual_dir}/outputs.png",
            )
            write_tensor_to_image(
                rearrange(tar_images, "b v h w c-> (b h) (v w) c"),
                f"{self.visual_dir}/gts.png",
            )
            write_tensor_to_image(
                rearrange(ref_images, "b v h w c-> (b h) (v w) c"),
                f"{self.visual_dir}/inputs.png",
            )

        if (
            step % self.config.print_every == 0
            and self.world_rank == 0
            and acc_step == 0
        ):
            mse = F.mse_loss(outputs, tar_images)
            outputs_flat = rearrange(outputs, "b v h w c-> (b v) c h w")
            tar_images_flat = rearrange(tar_images, "b v h w c-> (b v) c h w")
            psnr = state["psnr_fn"](outputs_flat, tar_images_flat)
            ssim = state["ssim_fn"](outputs_flat, tar_images_flat)
            lpips = state["lpips_fn"](outputs_flat, tar_images_flat)
            
            self.logging_on_master(
                f"Step: {step}, Loss: {loss:.3f}, PSNR: {psnr:.3f}, "
                f"SSIM: {ssim:.3f}, LPIPS: {lpips:.3f}, "
                f"LR: {state['scheduler'].get_last_lr()[0]:.3e}"
            )
            self.writer.add_scalar("train/loss", loss, step)
            self.writer.add_scalar("train/psnr", psnr, step)
            self.writer.add_scalar("train/ssim", ssim, step)
            self.writer.add_scalar("train/lpips", lpips, step)
            
        return loss

    def test_initialize(
        self,
        model: Optional[torch.nn.Module] = None,
    ) -> Dict[str, Any]:
        """Initialize test components."""
        # Setup data
        dataset = None
        dataloaders = dict()
        if not self.config.render_video and self.config.test_index_fp is None:
            assert (
                self.config.test_input_views == 2
                and self.config.test_supervise_views == 3
            ), "Invalid input views and supervise views for RE10K, should be 2 and 3 respectively."
        
        folder = "/data/re10k/test"
        for zoom_factor in self.config.test_zoom_factor:
            dataset = EvalDataset(
                folder=folder,
                patch_size=self.config.dataset_patch_size,
                zoom_factor=zoom_factor,
                first_n=self.config.test_n,
                rank=self.world_rank,
                world_size=self.world_size,
                input_views=self.config.test_input_views,
                supervise_views=self.config.test_supervise_views,
                render_video=self.config.render_video,
                test_index_fp=self.config.test_index_fp,
            )
            dataloaders[f"zoom{zoom_factor}"] = (
                self.config.test_input_views,
                torch.utils.data.DataLoader(
                    dataset, batch_size=1, num_workers=2, pin_memory=True
                ),
            )
        self.logging_on_master(f"Total scenes: {len(dataset)}")

        # Setup model
        if model is None:
            model = KoNetDecoderOnlyModel(self.config.model_config).to(self.device)
            if self.config.use_torch_compile:
                model = torch.compile(model)
            print(f"Model is initialized in rank {self.world_rank}")

        # Setup metrics
        psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        lpips_fn = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(self.device)

        state = {
            "model": model,
            "dataloaders": dataloaders,
            "psnr_fn": psnr_fn,
            "ssim_fn": ssim_fn,
            "lpips_fn": lpips_fn,
        }
        print(f"Launcher(Test) is initialized in rank {self.world_rank}")
        return state

    @torch.inference_mode()
    def test_iteration(self, step: int, state: Dict[str, Any]) -> None:
        """Execute test iteration."""
        dataloaders = state["dataloaders"]
        model = state["model"]
        model.eval()

        for label, (input_views, dataloader) in dataloaders.items():
            psnrs, lpips, ssims = [], [], []
            canvas = []  # for visualization
            
            for data in tqdm.tqdm(dataloader, desc="Testing"):
                ref_images, ref_cameras, tar_cameras, tar_images = self.preprocess(
                    data, input_views=input_views
                )
                ref_paths, tar_paths = data["image_path"][:input_views], data["image_path"][input_views:]
                
                # Forward pass
                with torch.amp.autocast(
                    "cuda", enabled=self.config.amp, dtype=self.amp_dtype
                ):
                    outputs = model(ref_images, ref_cameras, tar_cameras)
                    outputs = torch.sigmoid(outputs)

                if self.config.render_video:
                    assert outputs.shape[0] == 1
                    print(outputs.shape)
                    path_splits = tar_paths[0, 0].split("/")
                    scene_name = path_splits[-3]
                    print(f"video saved in {self.test_dir}/{scene_name}.mp4")
                    # Save video using imageio
                    imageio.mimwrite(
                        f"{self.test_dir}/{scene_name}.mp4",
                        (outputs[0].cpu().numpy() * 255).astype(np.uint8),
                        format="ffmpeg",
                        fps=15,
                    )
                else:
                    # Save images for visualization
                    if len(canvas) < 10:
                        canvas_left = rearrange(ref_images, "b v h w c -> (b h) (v w) c")
                        canvas_right = rearrange(
                            torch.cat([tar_images, outputs], dim=3),
                            "b v h w c -> (b h) (v w) c",
                        )
                        canvas_mid = torch.ones(
                            len(canvas_left), 20, 3, device=self.device
                        )
                        canvas.append(
                            torch.cat([canvas_left, canvas_mid, canvas_right], dim=1)
                        )

                    # Compute metrics
                    outputs_flat = rearrange(outputs, "b v h w c -> (b v) c h w")
                    tar_images_flat = rearrange(tar_images, "b v h w c -> (b v) c h w")
                    psnrs.append(state["psnr_fn"](outputs_flat, tar_images_flat))
                    ssims.append(state["ssim_fn"](outputs_flat, tar_images_flat))
                    lpips.append(state["lpips_fn"](outputs_flat, tar_images_flat))

            if self.config.render_video:
                return

            # Save visualization canvas
            canvas = torch.cat(canvas, dim=0)
            write_tensor_to_image(
                canvas, f"{self.test_dir}/rank{self.world_rank}_{label}views.png"
            )

            def distributed_average(data: List[float], name: str) -> float:
                """Compute distributed average of metrics."""
                # Collect metrics from all ranks
                collected_sizes = [None] * self.world_size
                torch.distributed.all_gather_object(collected_sizes, len(data))
                collected = [
                    torch.empty(size, device=self.device) for size in collected_sizes
                ]
                torch.distributed.all_gather(
                    collected, torch.tensor(data, device=self.device)
                )
                collected = torch.cat(collected)

                if torch.isinf(collected).any():
                    self.logging_on_master(
                        f"Inf found in {label} views, {sum(torch.isinf(collected))} inf values for {name}."
                    )
                    collected = collected[~torch.isinf(collected)]
                if torch.isnan(collected).any():
                    self.logging_on_master(
                        f"NaN found in {label} views, {sum(torch.isnan(collected))} nan values for {name}."
                    )
                    collected = collected[~torch.isnan(collected)]

                avg = collected.mean().item()
                return avg, len(collected)

            avg_psnr, n_total = distributed_average(psnrs, "psnr")
            avg_lpips, n_total = distributed_average(lpips, "lpips")
            avg_ssim, n_total = distributed_average(ssims, "ssim")

            self.logging_on_master(
                f"PSNR{label}: {avg_psnr:.3f}, SSIM{label}: {avg_ssim:.3f}, LPIPS{label}: {avg_lpips:.3f} "
                f"evaluated on {n_total} scenes at step {step}."
            )

            if self.world_rank == 0:
                self.writer.add_scalar(f"test/psnr{label}", avg_psnr, step)
                self.writer.add_scalar(f"test/ssim{label}", avg_ssim, step)
                self.writer.add_scalar(f"test/lpips{label}", avg_lpips, step)
                with open(f"{self.test_dir}/metrics.json", "w") as f:
                    json.dump(
                        {
                            "label": label,
                            "step": step,
                            "n_total": n_total,
                            "psnr": avg_psnr,
                            "ssim": avg_ssim,
                            "lpips": avg_lpips,
                        },
                        f,
                    )


if __name__ == "__main__":
    """Example usage:
    
    # 2GPUs dry run
    OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
        src/nvs_training/launcher.py lvsm-dry-run --model_config.encoder.num_layers 2
    """

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="torchmetrics")

    configs = {
        "konet": (
            "Knowledge Network with GeoPE attention",
            KoNetLauncherConfig(),
        ),
        "konet-dry-run": (
            "Dry run for testing and debugging",
            KoNetLauncherConfig(
                amp=True,
                amp_dtype="fp16",
                dataset_batch_scenes=1,
                max_steps=10,
                test_every=5,
                test_n=10,
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    launcher = KoNetLauncher(cfg)
    launcher.run()
