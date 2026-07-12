import os
import argparse
import yaml
import torch

from utils.builder import build_module
from utils.misc import init_distributed_mode, is_main_process


def parse_args():
    parser = argparse.ArgumentParser(description="MetNet Nowcasting Training")
    parser.add_argument(
        "--projects_configs_dir",
        type=str,
        default="./configs",
        help="config directory",
    )
    parser.add_argument("--workspace", type=str, required=True, help="workspace to save checkpoints")
    parser.add_argument("--version", type=str, required=True, help="experiment version name")
    parser.add_argument("--gpus", type=str, default="0", help="gpu ids, comma separated or 'all'")
    parser.add_argument("--ngpus", type=int, default=1, help="number of gpus per node")
    if torch.__version__.startswith("2."):
        parser.add_argument("--local-rank", type=int, default=0)
    else:
        parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    if args.gpus == "all":
        args.gpus = [str(i) for i in range(args.ngpus)]
    else:
        args.gpus = args.gpus.split(",")
    if args.ngpus != len(args.gpus):
        args.ngpus = len(args.gpus)
    return args


def load_config(config_dir: str):
    config_path = os.path.join(config_dir, "rainfall_forecast.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_training_modules():
    import datasets.nowcast_dataset  # noqa: F401
    import criterions.metnet_loss  # noqa: F401
    import runners.metnet_trainer  # noqa: F401
    import models.metnet.metnet  # noqa: F401


def main():
    args = parse_args()
    load_training_modules()
    init_distributed_mode(args)

    config = load_config(args.projects_configs_dir)
    config["config_dir"] = args.projects_configs_dir
    config["workspace"] = os.path.join(args.workspace, args.version)
    config["local_rank"] = args.local_rank
    config["world_size"] = args.world_size if args.distributed else 1

    runner_cfg = config["runner"].copy()
    runner_cfg.update(
        {
            "dataset": config["dataset"],
            "model": config["model"],
            "criterion": config["criterion"],
            "workspace": config["workspace"],
            "local_rank": config["local_rank"],
            "world_size": config["world_size"],
        }
    )

    if is_main_process():
        os.makedirs(config["workspace"], exist_ok=True)
        merged_path = os.path.join(config["workspace"], "config.yaml")
        with open(merged_path, "w") as f:
            yaml.safe_dump(config, f)

    trainer = build_module(
        runner_cfg,
        parent="runners",
    )
    trainer.run()


if __name__ == "__main__":
    main()
