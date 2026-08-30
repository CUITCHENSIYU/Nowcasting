import argparse
import os
import shutil

import torch
import yaml

from utils.builder import build_module
from utils.misc import init_distributed_mode, is_main_process


def parse_args():
    parser = argparse.ArgumentParser(description="MetNet Nowcasting Training")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="path to yaml config file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="directory to save logs and checkpoints",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="gpu ids, comma separated (e.g. 0 or 0,1,2) or 'all'",
    )
    if torch.__version__.startswith("2."):
        parser.add_argument("--local-rank", type=int, default=0)
    else:
        parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    if args.gpus == "all":
        n_visible = torch.cuda.device_count() if torch.cuda.is_available() else 1
        args.gpus = [str(i) for i in range(n_visible)]
    else:
        args.gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    args.ngpus = len(args.gpus)
    return args


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_training_modules():
    import datasets.nowcast_dataset  # noqa: F401
    import datasets.nowcast_dataset_rnn  # noqa: F401
    import datasets.nowcast_dataset_skillfull  # noqa: F401
    import criterions.metnet_loss  # noqa: F401
    import runners.metnet_trainer  # noqa: F401
    import runners.rnn_trainer  # noqa: F401
    import runners.skillfull_trainer  # noqa: F401
    import models.metnet.metnet  # noqa: F401
    import models.predrnn.rnn  # noqa: F401
    import models.skillfull_nowcasting.skillfull_nowcasting  # noqa: F401


def main():
    args = parse_args()
    load_training_modules()
    init_distributed_mode(args)

    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config not found: {config_path}")

    config = load_config(config_path)
    config["config_path"] = config_path
    config["workspace"] = os.path.abspath(args.output)
    config["local_rank"] = args.local_rank
    config["world_size"] = args.world_size if args.distributed else 1
    config["gpus"] = args.gpus

    runner_cfg = config["runner"].copy()
    runner_cfg.update(
        {
            "dataset": config["dataset"],
            "model": config["model"],
            "criterion": config.get("criterion"),
            "workspace": config["workspace"],
            "local_rank": config["local_rank"],
            "world_size": config["world_size"],
        }
    )

    if is_main_process():
        os.makedirs(config["workspace"], exist_ok=True)
        shutil.copy2(config_path, os.path.join(config["workspace"], "config.yaml"))

    trainer = build_module(
        runner_cfg,
        parent="runners",
    )
    trainer.run()


if __name__ == "__main__":
    main()
