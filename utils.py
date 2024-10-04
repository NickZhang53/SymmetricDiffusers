import math
import os

import torch

import models
import PL_distribution as PL
from diffusion import DiffusionUtils


# =============================================================================
# Model Related
# =============================================================================


def save_checkpoint(
    config, epoch, model, optimizer, scheduler, finished, ckpt_dir="./saved_models"
):
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "finished": finished,
        },
        f"{ckpt_dir}/ckpt_{config.train.run_name}.pth",
    )


def load_checkpoint(config, model, optimizer, scheduler, ckpt_dir="./saved_models"):
    local_rank = int(os.environ["LOCAL_RANK"])

    ckpt_path = f"{ckpt_dir}/ckpt_{config.train.run_name}.pth"
    nxt_epoch = 0
    finished = False
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{local_rank}")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        nxt_epoch = ckpt["epoch"]
        finished = ckpt["finished"]
    return nxt_epoch, model, optimizer, scheduler, finished


def init_model(config):
    d_out_adjust = "0"
    if config.train.diffusion.transition == "swap" and config.train.diffusion.reverse == "original":
        d_out_adjust = "1"
    elif config.train.diffusion.reverse == "generalized_PL":
        d_out_adjust = "square"

    use_pos_enc = True

    model = models.ReverseDiffusion(
        config.dataset,
        config.CNN.in_channels,
        config.num_pieces,
        config.image_size,
        config.CNN.hidden_channels1,
        config.CNN.kernel_size1,
        config.CNN.stride1,
        config.CNN.padding1,
        config.CNN.hidden_channels2,
        config.CNN.kernel_size2,
        config.CNN.stride2,
        config.CNN.padding2,
        config.num_digits,
        config.transformer.embd_dim,
        config.transformer.nhead,
        config.transformer.d_hid,
        config.transformer.n_layers,
        config.transformer.dropout,
        d_out_adjust,
        use_pos_enc,
    ).cuda()

    return model


def init_diffusion_utils(config):
    perm_fix_first = config.dataset == "tsp"

    diffusion_utils = DiffusionUtils(
        config.train.diffusion.num_timesteps,
        config.train.sample_N,
        config.train.diffusion.transition,
        config.train.diffusion.latent,
        config.train.reinforce_N,
        config.train.reinforce_ema_rate,
        config.train.entropy_reg_rate,
        config.train.diffusion.reverse,
        config.train.diffusion.reverse_steps,
        config.train.loss,
        config.beam_size,
        perm_fix_first,
    )

    return diffusion_utils


def get_ddp_generator(seed=3407):
    local_rank = int(os.environ["LOCAL_RANK"])
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g


# =============================================================================
# Other Utils
# =============================================================================


def perm_list_to_mat(pi):
    I = torch.eye(pi.size(-1)).to(pi.device)
    return I[pi].float()


def permute_image_perm_list(perm_list, x):
    """
    Apply the permutation to a batch of image chunks
    Args:
        perm_list: [batch_shape, n]
        x: [batch_shape, n, c, h, w]

    Returns:
        Permuted set of image chunks.
    """
    x, perm_list = torch.broadcast_tensors(x, perm_list[(...,) + (None,) * 3])
    return torch.gather(x, -4, perm_list)


def permute_int_list(perm_list, x):
    """
    Apply the permutation to x

    Args:
        perm_list: [batch_shape, n]
        x: [batch_shape, n]

    Returns:
        shape [batch_shape, n]
    """
    x, perm_list = torch.broadcast_tensors(x, perm_list)
    return torch.gather(x, -1, perm_list).long()


def permute_list(perm_list, x):
    """
    Apply the permutation to x

    Args:
        perm_list: [batch_shape, n]
        x: [batch_shape, n]

    Returns:
        shape [batch_shape, n]
    """
    x, perm_list = torch.broadcast_tensors(x, perm_list)
    return torch.gather(x, -1, perm_list)


def permute_embd(perm_list, x):
    """
    Args:
        perm_list: [batch_shape, n]
        x: [batch_shape, n, d]

    Returns:
        shape [batch_shape, n, d]
    """
    x, perm_list = torch.broadcast_tensors(x, perm_list.unsqueeze(-1))
    return torch.gather(x, -2, perm_list)


@torch.no_grad()
def insert_back_to_idx(x, idx):
    """
    Args:
        x: shape [batch_shape, n]
        idx: shape [batch_shape]
    """
    range_tensor = torch.arange(x.size(-1)).to(x.device)
    mask = range_tensor >= idx.unsqueeze(-1)
    rolled_x = torch.roll(x, shifts=1, dims=-1)
    rearranged = torch.where(mask, rolled_x, x)
    rearranged[range_tensor == idx.unsqueeze(-1)] = x[..., -1].flatten()

    return rearranged


@torch.no_grad()
def insert_idx_to_back(x, idx):
    """
    Args:
        x: shape [batch_shape, n]
        idx: shape [batch_shape]
    """
    range_tensor = torch.arange(x.size(-1)).to(x.device)
    mask = range_tensor >= idx.unsqueeze(-1)
    rolled_x = torch.roll(x, shifts=-1, dims=-1)
    rearranged = torch.where(mask, rolled_x, x)
    index_elements = torch.gather(x, -1, idx.unsqueeze(-1)).squeeze(-1)
    rearranged[..., -1] = index_elements

    return rearranged


@torch.no_grad()
def insert_back_to_idx_images(x, idx):
    """
    Args:
        x: shape [b, n, c, h, w]
        idx: shape [b]
    """
    range_tensor = torch.arange(x.size(-4)).to(x.device)  # [n]
    roll_mask = (range_tensor >= idx.unsqueeze(-1))[(...,) + (None,) * 3]  # [b, n, 1, 1, 1]
    rolled_x = torch.roll(x, shifts=1, dims=-4)
    rearranged = torch.where(roll_mask, rolled_x, x)

    replace_mask = (range_tensor == idx.unsqueeze(-1))[(...,) + (None,) * 3]
    result = torch.where(replace_mask, x[..., [-1], :, :, :], rearranged)

    return result


@torch.no_grad()
def insert_idx_to_back_images(x, idx):
    """
    Args:
        x: shape [batch_shape, n, c, h, w]
        idx: shape [batch_shape]
    """
    range_tensor = torch.arange(x.size(-4)).to(x.device)
    roll_mask = (range_tensor >= idx.unsqueeze(-1))[(...,) + (None,) * 3]  # [b, n, 1, 1, 1]
    rolled_x = torch.roll(x, shifts=-1, dims=-4)
    rearranged = torch.where(roll_mask, rolled_x, x)

    idx = idx[(...,) + (None,) * 4].expand(*((-1,) * (x.dim() - 3)), *x.shape[-3:])
    index_elements = torch.gather(x, -4, idx)  # [b, 1, c, h, w]
    rearranged[..., [-1], :, :, :] = index_elements

    return rearranged


@torch.no_grad()
def swap_by_idx(x, idx):
    """
    Args:
        x: shape [batch_shape, n]
        idx: shape [batch_shape, 2]
    """
    x_swapped = x.clone()
    first = x.gather(-1, idx[..., 0:1])
    second = x.gather(-1, idx[..., 1:2])
    x_swapped.scatter_(-1, idx[..., 0:1], second)
    x_swapped.scatter_(-1, idx[..., 1:2], first)

    return x_swapped


@torch.no_grad()
def swap_by_idx_images(x, idx):
    """
    Args:
        x: shape [batch_shape, n, c, h, w]
        idx: shape [batch_shape, 2]
    """
    idx = idx[(...,) + (None,) * 3].expand(*((-1,) * (x.dim() - 3)), *x.shape[-3:])
    first_idx = idx[..., [0], :, :, :]
    second_idx = idx[..., [1], :, :, :]
    first = torch.gather(x, -4, first_idx)
    second = torch.gather(x, -4, second_idx)

    x_swapped = x.clone()
    x_swapped.scatter_(-4, first_idx, second)
    x_swapped.scatter_(-4, second_idx, first)

    return x_swapped


@torch.no_grad()
def complete_range(x, n):
    """
    Args:
        x: shape [batch_shape, k]
        n: int
        1 <= k <= n
    Returns:
        shape [batch_shape, n]
    """
    device = x.device
    batch_shape = x.shape[:-1]
    k = x.size(-1)

    all_numbers = torch.arange(n, device=device)
    comparison = x.unsqueeze(-1) == all_numbers  # [batch_shape, k, n]

    mask = comparison.any(-2)  # [batch_shape, n]
    missing_numbers = torch.masked_select(all_numbers, ~mask).reshape(*batch_shape, n - k)

    return torch.cat([x, missing_numbers], dim=-1)


@torch.no_grad()
def find_perm_images(x1, x2):
    """
    Find the perm such that applying the perm to x1 gives x2
    x1 --perm--> x2

    Args:
        x_1: shape [batch_shape, n, c, h, w]
        x_2: shape [batch_shape, n, c, h, w]

    Returns:
        shape [batch_shape, n]
    """
    equality_matrix = torch.cdist(x1.flatten(start_dim=-3), x2.flatten(start_dim=-3), p=1)
    perm_list = torch.argmax((equality_matrix < 1e-8).int(), dim=-2)  # shape [batch, n]
    return perm_list


@torch.no_grad()
def find_perm(x1, x2):
    """
    Find the perm such that applying the perm to x1 gives x2
    x1 --perm--> x2

    Args:
        x_1: shape [batch_shape, n]
        x_2: shape [batch_shape, n]

    Returns:
        shape [batch_shape, n]
    """
    x1 = x1.float().unsqueeze(-1)
    x2 = x2.float().unsqueeze(-1)
    equality_matrix = torch.cdist(x1, x2, p=1)
    perm_list = torch.argmax((equality_matrix < 1e-8).int(), dim=-2)  # shape [batch, n]
    return perm_list


@torch.no_grad()
def log_prob_normal_dist_images(x, mean, var=1.0, no_const=False):
    """
    Computes log p(x) under N(x | mean, var I)

    Args:
        x: shape [batch_shape, n, c, h, w]
        mean: shape [batch_shape, n, c, h, w]

    Returns:
        shape [batch_shape]
    """
    x = x.flatten(start_dim=-4)
    mean = mean.flatten(start_dim=-4)
    D = x.size(-1)

    mse = -((x - mean) * (x - mean)).sum(-1) / (2 * var)

    if no_const:
        return mse
    else:
        return -D * math.log(2 * math.pi) / 2 - math.log(var) / 2 + mse


@torch.no_grad()
def count_rising_sequence(perm):
    """
    Args:
        perms: [batch, n]

    Returns:
        [batch]
    """
    return (torch.diff(perm) < 0).sum(-1) + 1


@torch.no_grad()
def log_binom(n, k):
    return (n + 1).lgamma() - (k + 1).lgamma() - ((n - k) + 1).lgamma()


@torch.no_grad()
def batch_randperm(batch, n):
    uniform_scores = torch.zeros(batch, n).cuda()
    randperms = PL.sample(uniform_scores, 1).squeeze(0)  # [batch, n]
    return randperms


class TSPEvaluator(object):
    def __init__(self, points):
        """
        Args:
            points: shape [batch, n, 2]
        """
        self.dist_mat = torch.cdist(points, points)  # [batch, n, n]

    @torch.no_grad()
    def evaluate(self, tour, tour_is_cycle=False):
        """
        Args:
            tour: shape [batch, n + 1]
        Returns:
            total_cost: shape [batch]
        """
        device = tour.device
        n = self.dist_mat.size(-1)
        tour_batch_shape = tour.shape[:-1]
        dist_mat_batch_shape = self.dist_mat.shape[:-2]
        batch_shape = torch.broadcast_shapes(tour_batch_shape, dist_mat_batch_shape)

        if not tour_is_cycle:
            tour = torch.cat([tour, tour[..., [0]]], -1)

        tour = tour.expand(*batch_shape, -1)
        dist_mat = self.dist_mat.expand(*batch_shape, -1, -1)

        total_cost = torch.zeros(batch_shape, device=device)

        for i in range(n):
            start = tour[..., [i], None].expand(*((-1,) * tour.dim()), n)  # [batch, 1, n]
            # logger.debug(f"dist_mat.shape = {dist_mat.shape}")
            # logger.debug(f"start.shape = {start.shape}")
            start_dist = torch.gather(dist_mat, -2, start).squeeze(-2)  # [batch, n]
            cost = torch.gather(start_dist, -1, tour[..., [i + 1]]).squeeze(-1)  # [batch]
            total_cost += cost
        return total_cost


def add_zero_to_perm(perm, add1=True):
    """
    Args:
        perm: shape [batch_shape, n]
    Returns:
        shape [batch_shape, n+1]
    """
    if add1:
        perm = perm + 1
    first_zeros = torch.zeros(perm.shape[:-1] + (1,), device=perm.device).long()
    result_perm = torch.cat([first_zeros, perm], dim=-1)
    return result_perm


def mask_scores_pos_zero(scores):
    """
    Force selecting 0 at 0th position
    Args:
        scores: shape [batch_shape, n, n]
    Returns:
        shape [batch_shape, n, n]
    """
    n = scores.size(-1)
    mask = torch.zeros(n, n, device=scores.device)
    mask[0, 1:] = float("-inf")
    mask[1:, 0] = float("-inf")
    scores = scores + mask
    return scores


def incidence_matrix_mask(n):
    """
    Returns the attention mask (-inf) given by the incidence matrix of K_n
    Args:
        n: int
    Return:
        tensor, shape [n, n choose 2]
    """
    num_edges = n * (n - 1) // 2
    incidence_matrix = torch.full((n, num_edges), float("-inf"))

    triu_indices = torch.triu_indices(n, n, offset=1)

    incidence_matrix[triu_indices[0], torch.arange(num_edges)] = 0
    incidence_matrix[triu_indices[1], torch.arange(num_edges)] = 0

    return incidence_matrix.cuda()


def points_to_pairwise_dist(points):
    """
    Args:
        points: [batch, n, 2]
    Returns:
        [batch, n choose 2]
    """
    batch_shape = points.shape[:-2]
    n = points.size(-2)
    device = points.device

    dist_mat = torch.cdist(points, points)  # [batch_shape, n, n]
    upper_tri_mask = torch.triu(torch.ones(n, n, device=device), diagonal=1)  # [n, n]

    dist_mat_flatten = dist_mat.flatten(start_dim=-2)  # [batch_shape, n^2]
    mask = upper_tri_mask.flatten() * torch.arange(n * n, device=device)  # [n^2]
    indices = mask[mask != 0].expand(*batch_shape, -1).long()
    flattened_dist = torch.gather(dist_mat_flatten, -1, indices)

    return flattened_dist
