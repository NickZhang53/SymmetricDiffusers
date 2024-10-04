import os

import torch
import torch.distributed as dist
import torch.optim as optim
from loguru import logger
from torch.nn.parallel import DistributedDataParallel

import utils
import wandb
from datasets import batch_chunk_image, get_train_loader
from eval import validate
from lr_schedulers import get_schedule_fn


def train_diffusion(
    diffusion_utils,
    reverse_model,
    optimizer,
    scheduler,
    train_loader,
    config,
    start_epoch,
    ckpt_dir,
):
    n = config.num_pieces
    diffusion_utils.train()
    local_rank = int(os.environ["LOCAL_RANK"])

    for epoch in range(start_epoch, config.train.epochs):
        if local_rank == 0:
            logger.info(f"Epoch {epoch}:")
        reverse_model.train()

        for i, data in enumerate(train_loader):
            if config.dataset in [
                "unscramble-noisy-MNIST",
                "unscramble-CIFAR10",
            ]:
                inputs, _ = data
                gt_pieces, _, __ = batch_chunk_image(inputs, n)
                gt_pieces = gt_pieces.cuda()

            elif config.dataset == "sort-MNIST":
                random_pieces, gt_perm_list = data
                random_pieces, gt_perm_list = random_pieces.cuda(), gt_perm_list.cuda()
                gt_pieces = utils.permute_image_perm_list(gt_perm_list, random_pieces)

            elif config.dataset == "tsp":
                _, points, __, tour = data  # points: [batch, n, 2]; tour: [batch, n+1]
                points, tour = points.cuda(), tour.cuda()
                tour = tour[:, :-1]  # [batch, n]
                gt_pieces = utils.permute_embd(tour, points)

            else:
                raise NotImplementedError

            # gt_pieces is x_start
            loss = diffusion_utils.training_losses(gt_pieces, reverse_model)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reverse_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if config.train.record_wandb and local_rank == 0:
                wandb.log({"diffusion_loss": loss, "learning_rate": scheduler.get_last_lr()[0]})

            if i > 0 and i % 50 == 0 and local_rank == 0:
                logger.info(f"Epoch {epoch}, minibatch {i}, current loss = {loss.item()}")

        if local_rank == 0:
            finished = epoch == config.train.epochs - 1
            utils.save_checkpoint(
                config, epoch, reverse_model, optimizer, scheduler, finished, ckpt_dir
            )
            validate(config, reverse_model, ckpt_dir)


def train(config, ckpt_dir):
    local_rank = int(os.environ["LOCAL_RANK"])

    model = utils.init_model(config)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DistributedDataParallel(model, device_ids=[local_rank])
    model.train()

    train_loader = get_train_loader(config)

    num_training_steps = len(train_loader) * config.train.epochs

    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total number of parameters: {total_params}")
        logger.info(f"Total number of training steps: {num_training_steps}")

    if config.train.scheduler == "transformer":
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=config.train.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=1e-2,
            eps=1e-9,
        )
    else:
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=1e-4,
        )

    scheduler = get_schedule_fn(
        config.train.scheduler,
        num_training_steps=num_training_steps,  # cosine-decay scheduler only
        warmup_steps=config.train.warmup_steps,  # transformer scheduler only
        dim_embed=config.transformer.embd_dim,  # transformer scheduler only
    )(optimizer)

    start_epoch = 0
    start_epoch, model, optimizer, scheduler, finished = utils.load_checkpoint(
        config, model, optimizer, scheduler, ckpt_dir
    )
    dist.barrier()
    if finished:
        exit(0)

    if config.train.record_wandb and local_rank == 0:
        project_name = "latent-permutation-diffusion"
        project_name = config.dataset
        wandb.init(
            project=project_name,
            name=config.train.run_name,
            config=config.to_dict(),
        )

    diffusion_utils = utils.init_diffusion_utils(config)

    train_diffusion(
        diffusion_utils,
        model,
        optimizer,
        scheduler,
        train_loader,
        config,
        start_epoch,
        ckpt_dir,
    )

    if config.train.record_wandb and local_rank == 0:
        wandb.run.summary["total_params"] = total_params

    dist.barrier()
    if local_rank == 0:
        torch.save(model.module.state_dict(), f"{ckpt_dir}/{config.train.run_name}.pth")

    return model
