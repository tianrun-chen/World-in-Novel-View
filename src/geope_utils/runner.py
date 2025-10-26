"""
Base training runner for GeoPE models.

This module provides the base launcher class for training and evaluation
of GeoPE-based models with support for FSDP and distributed training.
"""

import glob
import itertools
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
import yaml
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from functools import partial

# FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    CPUOffload,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)

from torch_npu.npu import amp


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def nested_to_device(data: Dict, device) -> Dict:
    """Recursively move tensors to device."""
    if isinstance(data, dict):
        return {k: nested_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [nested_to_device(v, device) for v in data]
    elif isinstance(data, Tensor):
        return data.to(device)
    else:
        return data


@dataclass
class LauncherConfig:
    """Base configuration for training launcher."""
    
    # Output configuration
    output_dir: str = "results/dbg"
    max_steps: int = 100
    auto_resume: bool = False
    resume: Optional[str] = None
    only_model: bool = False

    # Checkpoint configuration
    ckpt_every: int = 10
    ckpt_keeps: int = 3

    # Training configuration
    amp: bool = False
    amp_dtype: Literal["bf16", "fp16"] = "fp16"
    check_nan_in_params: bool = False
    acc: int = 1
    fixed_seed: bool = True
    seed: int = 42
    grad_clip: float = 0.0

    # Test configuration
    test_every: int = -1
    test_only: bool = False

    # FSDP configuration
    use_fsdp: bool = True
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_cpu_offload: bool = True
    fsdp_mixed_precision: str = "fp16"
    fsdp_auto_wrap_policy: str = "size_based"
    fsdp_min_num_params: int = 1_000_000
    fsdp_transformer_layer_cls_to_wrap: Optional[List[str]] = None

    # Subdirectories
    ckpt_subdir: str = "ckpts"
    stats_subdir: str = "stats"
    visual_subdir: str = "visuals"
    test_subdir: str = "tests"


class Launcher:
    """Base launcher class for training and evaluation."""
    
    def __init__(self, config: LauncherConfig) -> None:
        self.config = config

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        self.device = torch.device(f"npu:{self.local_rank}")

        # Setup output directories
        self.output_dir = self.config.output_dir
        self.ckpt_dir = f"{self.output_dir}/{self.config.ckpt_subdir}"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{self.output_dir}/{self.config.stats_subdir}"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.visual_dir = f"{self.output_dir}/{self.config.visual_subdir}"
        os.makedirs(self.visual_dir, exist_ok=True)
        self.test_dir = f"{self.output_dir}/{self.config.test_subdir}"
        os.makedirs(self.test_dir, exist_ok=True)

        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[
            self.config.amp_dtype
        ]
        
        # FSDP has its own mixed precision handling
        self.use_grad_scaler = (
            not self.config.use_fsdp 
            and self.config.amp 
            and self.config.amp_dtype == "fp16"
        )

        if self.world_rank == 0:
            self.writer = SummaryWriter(log_dir=f"{self.config.output_dir}/tb")
            if not self.config.test_only:
                (Path(self.output_dir) / "config.yaml").write_text(yaml.dump(config))
                print(f"Wrote config to {self.output_dir}/config.yaml")

    def run(self):
        """Run training or testing."""
        if self.config.test_only:
            self.test()
        else:
            self.train()

    def train_initialize(self) -> Dict[str, Any]:
        """
        Initialize training components.
        
        Subclasses should override this method to provide
        model, data, and other training components.
        """
        state = {}
        # For FSDP, model must be created on CPU
        model = torch.nn.Linear(1, 1)
        if not self.config.use_fsdp:
            model = model.to(self.device)
        state["model"] = model
        state["train_dataiter"] = itertools.repeat(
            {"x": torch.randn(1, 1), "y": torch.randn(1, 1)}
        )
        return state

    def test_initialize(
        self, model: Optional[torch.nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Initialize test components.
        
        Subclasses should override this method to provide
        model, data, and other test components.
        """
        state = {}
        if model is not None:
            state["model"] = model
        else:
            state["model"] = torch.nn.Linear(1, 1)
        state["test_dataiter"] = itertools.repeat(
            {"x": torch.randn(1, 1), "y": torch.randn(1, 1)},
            times=3 if self.world_rank == 0 else 2,
        )
        return state

    def train_iteration(self, step: int, state: Any, acc_step: int = 0) -> Tensor:
        """
        Execute one training iteration.
        
        Subclasses should override this method.
        """
        model = state["model"]
        optimizer = state["optimizer"]
        train_dataiter = state["train_dataiter"]

        model.train()
        data = next(train_dataiter)
        data = nested_to_device(data, self.device)

        output = model(data["x"])
        loss = F.mse_loss(output, data["y"])
        return loss

    @torch.inference_mode()
    def test_iteration(self, step: int, state: Any, acc_step: int = 0) -> Any:
        """
        Execute test iteration.
        
        Subclasses should override this method.
        """
        model = state["model"]
        test_dataiter = state["test_dataiter"]

        model.eval()
        losses = []
        for data in test_dataiter:
            data = nested_to_device(data, self.device)
            with amp.autocast():
                output = model(data["x"])
                loss = F.mse_loss(output, data["y"])
                losses.append(loss.item())

        # Collect losses from all ranks
        collected_sizes = [None] * self.world_size
        torch.distributed.all_gather_object(collected_sizes, len(losses))

        collected_metrics = [
            torch.empty(size, device=self.device) for size in collected_sizes
        ]
        torch.distributed.all_gather(
            collected_metrics, torch.tensor(losses, device=self.device)
        )
        collected_metrics = torch.cat(collected_metrics)

        avg_loss = collected_metrics.mean().item()
        self.print_on_master(f"Average loss: {avg_loss}")
        if self.world_rank == 0:
            self.writer.add_scalar("test/loss", avg_loss, step)
        return avg_loss

    def load_state_dict_to_model(
        self, state_dict: Dict, model: torch.nn.Module
    ) -> None:
        """Load state dict to model."""
        self._loosely_load_state_dict_to_model(state_dict, model)

    def load_state_dict_to_optimizer(
        self, state_dict: Dict, optimizer: torch.optim.Optimizer
    ) -> None:
        """Load state dict to optimizer."""
        optimizer.load_state_dict(state_dict)

    def load_state_dict_to_scheduler(
        self, state_dict: Dict, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    ) -> None:
        """Load state dict to scheduler."""
        if scheduler is not None:
            scheduler.load_state_dict(state_dict)

    def save_checkpoint(self, step: int, state: Any) -> None:
        """Save training checkpoint."""
        is_last_step = (step == self.config.max_steps)
        is_ckpt_step = (step > 0 and step % self.config.ckpt_every == 0) or (step == 1)
        if not is_ckpt_step and not is_last_step:
            return

        model = state["model"]
        optimizer = state["optimizer"]
        
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dicts = {
                "model": model.state_dict(),
                "optimizer": FSDP.optim_state_dict(model, optimizer),
                "step": step,
            }
            
            ckpt_path = f"{self.ckpt_dir}/step-{step:09d}-rank{self.world_rank}.pt"
            torch.save(state_dicts, ckpt_path)
            print(f"[Rank {self.world_rank}] Saved sharded checkpoint to {ckpt_path}")

        torch.distributed.barrier()

        if self.world_rank == 0:
            # Clean up old checkpoints
            all_ckpts = glob.glob(f"{self.ckpt_dir}/step-*-rank0.pt")
            steps_to_clean = sorted([int(os.path.basename(f).split('-')[1]) for f in all_ckpts])[:-self.config.ckpt_keeps]

            for s_clean in steps_to_clean:
                for rank_file in glob.glob(f"{self.ckpt_dir}/step-{s_clean:09d}-rank*.pt"):
                    try:
                        os.remove(rank_file)
                    except OSError as e:
                        self.logging_on_master(f"Error cleaning old sharded checkpoint: {e}")

    def maybe_resume(self, state: Any) -> int:
        """Load checkpoint if needed."""
        step = 0

        ckpt_candidates = []
        if self.config.auto_resume:
            ckpt_candidates = sorted(glob.glob(f"{self.ckpt_dir}/*"), reverse=True)
        elif self.config.resume:
            assert os.path.exists(
                self.config.resume
            ), f"Checkpoint {self.config.resume} not found."
            ckpt_candidates = [self.config.resume]

        for ckpt_path in ckpt_candidates:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            except Exception as e:
                self.print_on_master(
                    f"Error loading checkpoint {ckpt_path}: {e}. Try next candidate."
                )
                continue

            self.load_state_dict_to_model(ckpt["model"], state["model"])
            if not self.config.only_model and not self.config.test_only:
                self.load_state_dict_to_optimizer(ckpt["optimizer"], state["optimizer"])
                self.load_state_dict_to_scheduler(
                    ckpt["scheduler"], state.get("scheduler", None)
                )
                step = ckpt.get("step", 0) + 1
            elif self.config.test_only:
                step = ckpt.get("step", 0)
            self.print_on_master(
                f"Resuming from ckpt: {ckpt_path}. set step to: {step}"
            )
            break
        return step

    def _loosely_load_state_dict_to_model(
        self, state_dict: Dict, model: nn.Module
    ) -> None:
        """Load state dict with loose matching."""
        
        def _load_logic(model_to_load, state_dict_to_load):
            state_dict_to_load = {
                k.replace("_orig_mod.", ""): v for k, v in state_dict_to_load.items()
            }
            model_to_load = getattr(model_to_load, "_orig_mod", model_to_load)
            
            state_dict_filtered = {}
            for k, v in state_dict_to_load.items():
                if k not in model_to_load.state_dict():
                    continue
                if model_to_load.state_dict()[k].shape != v.shape:
                    print(
                        f"Warning: {k} shape mismatch: {model_to_load.state_dict()[k].shape} vs {v.shape}"
                    )
                    continue
                state_dict_filtered[k] = v
            
            incompatible_keys = model_to_load.load_state_dict(state_dict_filtered, strict=False)
            if incompatible_keys.missing_keys:
                self.print_on_master(f"Missing keys when loading: {incompatible_keys.missing_keys}")
            if incompatible_keys.unexpected_keys:
                self.print_on_master(f"Unexpected keys when loading: {incompatible_keys.unexpected_keys}")

            self.print_on_master(
                f"Loosely loaded ckpt to model: "
                f"{len(state_dict_to_load)} keys in ckpt, "
                f"{len(state_dict_filtered)} keys loaded, "
                f"{len(model_to_load.state_dict())} keys in model."
            )

        if self.config.use_fsdp:
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            ):
                _load_logic(model, state_dict)
        else:
            _load_logic(model, state_dict)

    def train(self):
        """Execute training loop."""
        self.print_on_master(f"Distributed worker: {self.world_rank + 1} / {self.world_size}")

        if self.config.fixed_seed:
            set_random_seed(self.config.seed + self.world_rank)
        torch.npu.set_device(self.local_rank)

        # Initialize model and data
        state = self.train_initialize()
        model = state["model"]

        # Initialize distributed environment
        torch.distributed.init_process_group(backend="nccl")

        # Wrap model with FSDP or DDP
        if self.config.use_fsdp:
            self.logging_on_master("FSDP is enabled. Wrapping model...")
            
            mp_policy_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": None}
            fsdp_mp_dtype = mp_policy_map.get(self.config.fsdp_mixed_precision)
            mixed_precision_policy = None
            if fsdp_mp_dtype is not None:
                mixed_precision_policy = MixedPrecision(
                    param_dtype=fsdp_mp_dtype, reduce_dtype=fsdp_mp_dtype, buffer_dtype=fsdp_mp_dtype
                )

            sharding_strategy_map = {
                "FULL_SHARD": ShardingStrategy.FULL_SHARD, 
                "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
                "NO_SHARD": ShardingStrategy.NO_SHARD, 
                "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
            }
            fsdp_sharding_strategy = sharding_strategy_map[self.config.fsdp_sharding_strategy]

            cpu_offload = CPUOffload(offload_params=True) if self.config.fsdp_cpu_offload else None
            
            # Auto wrap policy
            if self.config.fsdp_auto_wrap_policy == "transformer_based":
                transformer_layer_cls_to_wrap: Set[nn.Module] = set()
                if not self.config.fsdp_transformer_layer_cls_to_wrap:
                    raise ValueError("transformer_based policy requires fsdp_transformer_layer_cls_to_wrap to be set.")
                
                for cls_name in self.config.fsdp_transformer_layer_cls_to_wrap:
                    found_cls = None
                    for module in list(sys.modules.values()):
                        if hasattr(module, cls_name):
                            found_cls = getattr(module, cls_name)
                            break
                    if found_cls:
                        transformer_layer_cls_to_wrap.add(found_cls)
                    else:
                        raise ImportError(f"Could not find class {cls_name} for FSDP wrapping.")

                auto_wrap_policy = partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=transformer_layer_cls_to_wrap,
                )
            else:  # size_based
                auto_wrap_policy = partial(
                    size_based_auto_wrap_policy, 
                    min_num_params=self.config.fsdp_min_num_params
                )
            
            model = FSDP(
                model, 
                auto_wrap_policy=auto_wrap_policy, 
                mixed_precision=mixed_precision_policy,
                sharding_strategy=fsdp_sharding_strategy, 
                cpu_offload=cpu_offload,
                device_id=self.device, 
                limit_all_gathers=True,
            )
            self.logging_on_master(f"Model wrapped with FSDP using {self.config.fsdp_auto_wrap_policy} policy.")
        
        else:  # DDP
            model = model.to(self.device)
            if self.world_size > 1:
                model = DDP(model, device_ids=[self.local_rank])
        
        state["model"] = model

        # Create optimizer and scheduler
        optimizer, scheduler = self.create_optimizer_and_scheduler(model)
        state["optimizer"] = optimizer
        state["scheduler"] = scheduler

        # Load checkpoint
        init_step = self.maybe_resume(state)
        
        if self.config.test_every > 0:
            test_state = self.test_initialize(model=state["model"])

        self.logging_on_master(
            f"Total trainable parameters: "
            f"{sum(p.numel() for p in state['model'].parameters() if p.requires_grad)}"
        )
        
        if self.use_grad_scaler:
            grad_scaler = amp.GradScaler()

        # Training loop
        t1 = time.time()
        for step in range(init_step, self.config.max_steps + 1):
            for acc_step in range(self.config.acc):
                loss = self.train_iteration(step, state, acc_step)
                loss = loss / self.config.acc

                if loss.isnan():
                    self.logging_on_master(f"Warning: [step={step}] rank={self.world_rank} | loss is NaN.")
                    if not self.use_grad_scaler:
                        self.logging_on_master("Exiting due to NaN loss without grad scaler.")
                        exit()
                    continue

                if self.use_grad_scaler:
                    grad_scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Check for NaN in parameters
                if self.config.check_nan_in_params:
                    for name, param in state["model"].named_parameters():
                        if torch.isnan(param).any():
                            print(f"[step={step}] rank={self.world_rank} | {name} has NaN.")
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                print(f"[step={step}] rank={self.world_rank} | {name} grad has NaN.")
            
            # Update model
            if self.use_grad_scaler:
                grad_scaler.unscale_(optimizer)
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                if self.config.grad_clip > 0:
                    if self.config.use_fsdp:
                        model.clip_grad_norm_(self.config.grad_clip)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimizer.step()

            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

            self.save_checkpoint(step, state)

            if (self.config.test_every > 0 and step % self.config.test_every == 0 and step > 0):
                _ = self.test_iteration(step, test_state)

        t2 = time.time()
        print(f"Training time: {t2 - t1:.2f} seconds.")
        torch.distributed.destroy_process_group()

    def create_optimizer_and_scheduler(self, model: nn.Module):
        """Create optimizer and scheduler."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = None
        return optimizer, scheduler

    def test(self):
        """Execute test loop."""
        assert (
            self.config.resume is not None or self.config.auto_resume
        ), "Resume checkpoint or auto_resume must be provided for testing."
        print("Distributed worker: %d / %d" % (self.world_rank + 1, self.world_size))

        if self.config.fixed_seed:
            set_random_seed(self.config.seed + self.world_rank)
        torch.npu.set_device(self.local_rank)

        # Initialize model and load checkpoint
        state = self.test_initialize()
        init_step = self.maybe_resume(state)

        for key in ["model"]:
            assert key in state, f"{key} is not in state."

        torch.distributed.init_process_group(backend="nccl")

        # Move to device
        for k, v in state.items():
            if isinstance(v, torch.nn.Module):
                v = v.to(self.device)
                state[k] = v

        # Run test
        _ = self.test_iteration(init_step, state)

        torch.distributed.destroy_process_group()

    def print_on_master(self, msg: str) -> None:
        """Print message only on master rank."""
        if self.world_rank == 0:
            print(msg)

    def logging_on_master(self, msg: str) -> None:
        """Log message with timestamp on master rank."""
        if self.world_rank == 0:
            msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
            logger = open(f"{self.output_dir}/log.txt", "a")
            logger.write(msg + "\n")
            logger.close()
            print(msg)


if __name__ == "__main__":
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    cfg = tyro.cli(LauncherConfig)
    launcher = Launcher(cfg)
    launcher.run()
