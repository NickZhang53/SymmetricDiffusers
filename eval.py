import os

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from scipy.stats import kendalltau

import utils
import wandb
from datasets import batch_chunk_image, get_test_loader


@torch.inference_mode()
def eval_image_dataset(config, ckpt_dir):
    model = utils.init_model(config)
    model.load_state_dict(torch.load(f"{ckpt_dir}/{config.train.run_name}.pth"))

    if config.train.record_wandb and config.eval_only:
        model.train()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.run.summary["total_params"] = total_params

    model.eval()

    diffusion_utils = utils.init_diffusion_utils(config)

    test_loader = get_test_loader(config)

    kendall_taus = []
    mean_mse = []
    mean_rmse = []
    mean_l1 = []
    l1_loss = torch.nn.L1Loss()

    perm_all_correct = 0
    perm_piece_correct = 0

    image_all_correct = 0
    image_piece_correct = 0

    total_images = 0
    total_pieces = 0

    for i, data in enumerate(test_loader):
        logger.info(f"Mini-batch {i}")
        if config.dataset == "unscramble-noisy-MNIST" or config.dataset == "sort-MNIST":
            random_pieces, gt_perm_list = data
            random_pieces, gt_perm_list = random_pieces.cuda(), gt_perm_list.cuda()
            pieces = utils.permute_image_perm_list(gt_perm_list, random_pieces)

        elif config.dataset == "unscramble-CIFAR10":
            inputs, _ = data
            pieces, random_pieces, gt_perm_list = batch_chunk_image(inputs, config.num_pieces)
            pieces, random_pieces, gt_perm_list = (
                pieces.cuda(),
                random_pieces.cuda(),
                gt_perm_list.cuda(),
            )
            gt_perm_list = torch.argsort(gt_perm_list)

        else:
            raise NotImplementedError

        if config.beam_search:
            ordered_pieces, predicted_perm_list = diffusion_utils.p_sample_beam_search(
                random_pieces, model
            )
        else:
            ordered_pieces, predicted_perm_list = diffusion_utils.p_sample_loop(
                random_pieces, model, deterministic=True
            )

        # Obtain the Kendall-Tau correlation coefficient for the target
        # and predicted list of permutation matrices.
        for p1, p2 in zip(gt_perm_list, predicted_perm_list):
            p1, p2 = p1.cpu(), p2.cpu()
            kendall_taus.append(kendalltau(p1, p2)[0])
            perm_all_correct += torch.equal(p1, p2)
            perm_piece_correct += torch.eq(p1, p2).sum().item()

        total_images += gt_perm_list.size(0)
        total_pieces += torch.numel(gt_perm_list)

        compare_images = ~torch.isclose(
            pieces.flatten(start_dim=2), ordered_pieces.flatten(start_dim=2)
        )  # shape [B, num_piecces**2, num_pixels]
        compare_result = compare_images.sum(dim=(1, 2)) == 0
        correct_images = compare_result.sum()

        correct_image_pieces = (compare_images.sum(-1) == 0).sum()

        image_all_correct += correct_images.item()
        image_piece_correct += correct_image_pieces.item()

        mse = F.mse_loss(pieces, ordered_pieces, reduction="mean")  # \sqrt( / bchw)
        mean_mse.append(mse.cpu())
        rmse = mse.sqrt()
        mean_rmse.append(rmse.cpu())
        l1 = l1_loss(pieces, ordered_pieces)  # mean absolute error
        mean_l1.append(l1.cpu())

    mean_kendall_tau = np.mean(kendall_taus)
    mean_mse = torch.stack(mean_mse).mean()
    mean_rmse = torch.stack(mean_rmse).mean()
    mean_l1 = torch.stack(mean_l1).mean()

    perm_accuracy = perm_all_correct / total_images
    perm_prop_correct_pieces = perm_piece_correct / total_pieces

    image_accuracy = image_all_correct / total_images
    image_prop_correct_pieces = image_piece_correct / total_pieces

    if config.train.record_wandb:
        wandb.run.summary["mean_kendall_tau"] = mean_kendall_tau
        wandb.run.summary["mean_mse"] = mean_mse
        wandb.run.summary["mean_rmse"] = mean_rmse
        wandb.run.summary["mean_l1"] = mean_l1
        wandb.run.summary["permutation_accuracy"] = perm_accuracy
        wandb.run.summary["permutation_prop_correct_pieces"] = perm_prop_correct_pieces
        wandb.run.summary["pixel_wise_accuracy"] = image_accuracy
        wandb.run.summary["pixel_wise_prop_correct_pieces"] = image_prop_correct_pieces
        wandb.finish()

    logger.info(f"Mean Kendall-Tau: {mean_kendall_tau}")
    logger.info(f"Mean mse: {mean_mse}")
    logger.info(f"Mean root mse: {mean_rmse}")
    logger.info(f"Mean l1 loss: {mean_l1}")
    logger.info(f"Permutation Accuracy: {perm_accuracy}")
    logger.info(f"Permutation Prop. of correct pieces: {perm_prop_correct_pieces}")
    logger.info(f"Pixel-wise Accuracy: {image_accuracy}")
    logger.info(f"Pixel-wise Prop. of correct pieces: {image_prop_correct_pieces}")


@torch.inference_mode()
def validate(config, model=None, ckpt_dir="./saved_models"):
    if config.dataset == "tsp":
        validate_tsp(config, model, ckpt_dir)


@torch.inference_mode()
def validate_tsp(config, model, ckpt_dir="./saved_models"):
    if model is None:
        local_rank = int(os.environ["LOCAL_RANK"])
        model = utils.init_model(config)
        ckpt = torch.load(
            f"{ckpt_dir}/ckpt_{config.train.run_name}.pth", map_location=f"cuda:{local_rank}"
        )
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()

    diffusion_utils = utils.init_diffusion_utils(config)

    test_loader = get_test_loader(config)

    logger.info(f"ON TEST SET:")
    mean_predicted_tour_length = []
    mean_gap = []
    mean_gt = []
    total_pieces = len(test_loader) * config.eval_batch_size * config.num_pieces
    correct_pieces = 0

    for i, data in enumerate(test_loader):
        _, random_pieces, __, tour = data  # random_pieces: points [batch, n, 2]; tour: [batch, n+1]
        random_pieces, tour = random_pieces.cuda(), tour.cuda()

        evaluator = utils.TSPEvaluator(random_pieces)
        gt_tour_length = evaluator.evaluate(tour, tour_is_cycle=True)

        if config.beam_search:
            ordered_pieces, predicted_perm_list = diffusion_utils.p_sample_beam_search(
                random_pieces, model
            )
        else:
            ordered_pieces, predicted_perm_list = diffusion_utils.p_sample_loop(
                random_pieces, model, deterministic=True
            )

        if i == 0:
            logger.info(f"gt tour (first graph in batch): {tour[0, :-1]}")
            logger.info(f"predicted tour (first graph in batch): {predicted_perm_list[0]}")
        correct_pieces += (tour[:, :-1] == predicted_perm_list).sum().cpu().item()

        predicted_tour_length = evaluator.evaluate(predicted_perm_list)
        gap = (predicted_tour_length - gt_tour_length) / gt_tour_length

        mean_predicted_tour_length.append(predicted_tour_length.cpu())
        mean_gap.append(gap.cpu())
        mean_gt.append(gt_tour_length.cpu())

    mean_predicted_tour_length = torch.stack(mean_predicted_tour_length).mean()
    mean_gap = torch.stack(mean_gap).mean()
    mean_gt = torch.stack(mean_gt).mean()
    prop_correct_pieces = correct_pieces / total_pieces

    logger.info(f"Mean Ground Truth Tour Length: {mean_gt}")
    logger.info(f"Mean Predicted Tour Length: {mean_predicted_tour_length}")
    logger.info(f"Mean Gap: {mean_gap}")
    logger.info(f"Prop. of Correct Pieces: {prop_correct_pieces}")

    if config.train.record_wandb:
        wandb.log({"tour_len": mean_predicted_tour_length})


@torch.inference_mode()
def eval_tsp(config, ckpt_dir, model=None):
    if model is None:
        local_rank = int(os.environ["LOCAL_RANK"])
        model = utils.init_model(config)
        model.load_state_dict(
            torch.load(f"{ckpt_dir}/{config.train.run_name}.pth", map_location=f"cuda:{local_rank}")
        )
    model.eval()

    diffusion_utils = utils.init_diffusion_utils(config)

    test_loader = get_test_loader(config)

    mean_predicted_tour_length = []
    mean_gap = []
    mean_gt = []

    for i, data in enumerate(test_loader):
        logger.info(f"Mini-batch {i}")
        _, random_pieces, __, tour = data  # random_pieces: points [batch, n, 2]; tour: [batch, n+1]
        random_pieces, tour = random_pieces.cuda(), tour.cuda()

        evaluator = utils.TSPEvaluator(random_pieces)
        gt_tour_length = evaluator.evaluate(tour, tour_is_cycle=True)

        if config.beam_search:
            ordered_pieces, predicted_perm_list = diffusion_utils.p_sample_beam_search(
                random_pieces, model
            )
        else:
            ordered_pieces, predicted_perm_list = diffusion_utils.p_sample_loop(
                random_pieces, model, deterministic=True
            )

        predicted_tour_length = evaluator.evaluate(predicted_perm_list)
        gap = (predicted_tour_length - gt_tour_length) / gt_tour_length

        mean_predicted_tour_length.append(predicted_tour_length.cpu())
        mean_gap.append(gap.cpu())
        mean_gt.append(gt_tour_length.cpu())

    mean_predicted_tour_length = torch.stack(mean_predicted_tour_length).mean()
    mean_gap = torch.stack(mean_gap).mean()
    mean_gt = torch.stack(mean_gt).mean()

    logger.info(f"Mean Ground Truth Tour Length: {mean_gt}")
    logger.info(f"Mean Predicted Tour Length: {mean_predicted_tour_length}")
    logger.info(f"Mean Gap: {mean_gap}")

    if config.train.record_wandb:
        wandb.run.summary["mean_gt_tour_length"] = mean_gt
        wandb.run.summary["mean_predicted_tour_length"] = mean_predicted_tour_length
        wandb.run.summary["mean_gap"] = mean_gap
        wandb.finish()


@torch.inference_mode()
def eval(config, ckpt_dir):
    if config.train.record_wandb and config.eval_only:
        project_name = config.dataset
        wandb.init(
            project=project_name, name=f"EVAL_{config.train.run_name}", config=config.to_dict()
        )

    if config.dataset in [
        "unscramble-noisy-MNIST",
        "sort-MNIST",
        "unscramble-CIFAR10",
    ]:
        eval_image_dataset(config, ckpt_dir)
    elif config.dataset == "tsp":
        eval_tsp(config, ckpt_dir)
    else:
        raise NotImplementedError
