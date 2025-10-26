"""
Main entry point for GeoPE Novel View Synthesis training and evaluation.

This module provides the main CLI interface for training and evaluating
the KoNet model with GeoPE attention mechanism.
"""

import warnings
from typing import Dict, Tuple

import tyro
from nvs_training.launcher import KoNetLauncher, KoNetLauncherConfig


def main():
    """Main entry point for training and evaluation."""
    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="torchmetrics")
    
    # Define available configurations
    configs: Dict[str, Tuple[str, KoNetLauncherConfig]] = {
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
        "konet-fsdp": (
            "KoNet with FSDP for large-scale training",
            KoNetLauncherConfig(
                use_fsdp=True,
                fsdp_sharding_strategy="FULL_SHARD",
                fsdp_cpu_offload=True,
                fsdp_mixed_precision="fp16",
                dataset_batch_scenes=2,
            ),
        ),
    }
    
    # Parse command line arguments
    cfg = tyro.extras.overridable_config_cli(configs)
    
    # Create and run launcher
    launcher = KoNetLauncher(cfg)
    launcher.run()


if __name__ == "__main__":
    main()
