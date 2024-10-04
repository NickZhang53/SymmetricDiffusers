import argparse
import json
import os

import ml_collections
import torch
import torch.distributed as dist
from loguru import logger

from eval import eval
from train import train


def get_config(config_json):
    logger.info(f"Reading config from JSON: {config_json}")
    with open(config_json, "r") as f:
        config = ml_collections.ConfigDict(json.loads(f.read()))
    return config


def prepare():
    torch.set_printoptions(profile="full")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_json", type=str, default="./configs", help="Path to config json file."
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="./checkpoint", help="Path to folder to save checkpoints."
    )
    parser.add_argument("--gpu", default="0", type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return args


if __name__ == "__main__":
    args = prepare()
    config = get_config(args.config_json)

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    if not config.eval_only:
        train(config, args.ckpt_dir)

    if local_rank == 0:
        eval(config, args.ckpt_dir)

    dist.destroy_process_group()
