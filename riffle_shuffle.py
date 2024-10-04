import math

import torch
import torch.nn.functional as F

import utils

# =============================================================================
# Sampling
# =============================================================================


@torch.no_grad()
def inverse_riffle_shuffle_perms(x, left_right):
    """
    Args:
        x: shape [batch_shape, n]
        left_right: shape [batch_shape, n], True means left
    """
    indices = torch.argsort(left_right.int(), dim=-1, descending=True, stable=True)
    shuffled_x = torch.gather(x, -1, indices)
    return shuffled_x


@torch.no_grad()
def inverse_riffle_shuffle(x, left_right):
    """
    Args:
        x: shape [batch_shape, n, c, h, w]
        left_right: shape [batch_shape, n], True means left
        Does not broadcast!
    """
    indices = torch.argsort(left_right.int(), dim=-1, descending=True, stable=True)
    indices = indices[(...,) + (None,) * 3].expand(*((-1,) * (x.dim() - 3)), *x.shape[-3:])
    shuffled_x = torch.gather(x, -4, indices)
    return shuffled_x


@torch.no_grad()
def sample_inverse_riffle_shuffle(logits, x, deterministic=False):
    """
    Args:
        logits: logit for dropping to the left, shape [batch_shape, n]
        x: shape [batch_shape, n, c, h, w]
        Does not broadcast!
    """
    if deterministic:
        left_right = 0.5 <= torch.sigmoid(logits)
    else:
        left_right = torch.rand(size=logits.shape, device=logits.device) <= torch.sigmoid(logits)
    return inverse_riffle_shuffle(x, left_right), left_right


@torch.no_grad()
def sample_inverse_riffle_shuffle_perms(logits, x, deterministic=False):
    """
    Args:
        logits: logit for dropping to the left, shape [batch_shape, n]
        x: shape [batch_shape, n, c, h, w]
        Does not broadcast!
    """
    if deterministic:
        left_right = 0.5 <= torch.sigmoid(logits)
    else:
        left_right = torch.rand(size=logits.shape, device=logits.device) <= torch.sigmoid(logits)
    return inverse_riffle_shuffle_perms(x, left_right), left_right


@torch.no_grad()
def sample_riffle_shuffle(x, x_images=False, a=torch.tensor(2)):
    """
    Args:
        x: shape [batch_shape, n, c, h, w]
        a: shape [batch_shape]
    """
    device = x.device
    if x_images:
        rand_pts_shape = x.shape[:-3]
    else:
        rand_pts_shape = x.shape
    a = a.to(device)

    rand_pts = torch.rand(size=rand_pts_shape, device=device)  # [batch, n]
    rand_pts, _ = torch.sort(rand_pts)
    rand_pts = torch.frac(a.unsqueeze(-1) * rand_pts)
    perms = torch.argsort(rand_pts)

    if x_images:
        result_x = utils.permute_image_perm_list(perms, x)
    else:
        result_x = utils.permute_int_list(perms, x)

    return result_x, perms


@torch.no_grad()
def sample_riffle_shuffle_no_identity(x):
    """
    Args:
        x: shape [batch_shape, n, c, h, w]
    """
    device = x.device
    batch_shape = x.shape[:-4]

    x = x.flatten(end_dim=-5)
    sample_result = torch.zeros(x.shape, device=device, dtype=torch.long)
    sample_result_perm = torch.zeros(x.shape[:-3], device=device, dtype=torch.long)

    while (sample_result.sum(dim=(-1, -2, -3, -4)) == 0).sum() > 0:
        samples, sampled_perms = sample_riffle_shuffle(x)
        is_identity = (~torch.isclose(x, samples)).sum(dim=(-1, -2, -3, -4)) == 0
        sample_result = torch.where(is_identity[(...,) + (None,) * 4], sample_result, samples)
        sample_result_perm = torch.where(
            is_identity[(...,) + (None,)], sample_result_perm, sampled_perms
        )

    return sample_result.unflatten(0, batch_shape), sample_result_perm.unflatten(0, batch_shape)


# =============================================================================
# Log Probabilities
# =============================================================================


def log_prob_inverse_riffle_shuffle(logits, x_tm1, x_t):
    """
    Args:
        logits: logit for dropping to the left, shape [batch_shape, n]
        x_tm1: shape [batch_shape, n, c, h, w]
        x_t: shape [batch_shape, n, c, h, w]

        x_t --inverse riffle shuffle--> x_tm1

    Returns:
        shape [batch_shape]
    """
    n = logits.size(-1)
    device = logits.device

    inverse_perm_list = utils.find_perm_images(
        x_t, x_tm1
    )  # identity or exactly two rising sequences
    perm_list = torch.argsort(inverse_perm_list)

    rises = ~(inverse_perm_list[..., :-1] < inverse_perm_list[..., 1:])  # shape [batch, n-1]
    cut_idx = torch.argmax(rises.int(), dim=-1, keepdim=True).detach()  # shape [batch, 1]

    left_log_prob = F.logsigmoid(logits)  # shape [batch, n]
    right_log_prob = -logits + left_log_prob  # shape [batch, n]

    log_probs = torch.where(perm_list <= cut_idx, left_log_prob, right_log_prob)  # shape [batch, n]
    log_probs = log_probs.sum(-1)  # shape [batch]

    # identity permutation
    mask = torch.ones(n + 1, n, device=device)
    mask = torch.tril(mask, diagonal=-1).bool()
    identity_log_probs = torch.where(
        mask, left_log_prob.unsqueeze(-2), right_log_prob.unsqueeze(-2)
    )  # shape [batch, n+1, n]
    identity_log_probs = identity_log_probs.sum(-1)
    identity_log_probs = torch.logsumexp(identity_log_probs, dim=-1)  # shape [batch]
    is_identity = (
        (~torch.isclose(x_tm1, x_t)).sum(dim=(-1, -2, -3, -4)) == 0
    ).detach()  # shape [batch]

    return torch.where(is_identity, identity_log_probs, log_probs)


def log_prob_inverse_riffle_shuffle_indices(logits, left_right):
    """
    Args:
        logits: logit for dropping to the left, shape [batch_shape, n]
        left_right: shape [batch_shape, n], True means left

    Returns:
        shape [batch_shape]
    """
    n = logits.size(-1)
    device = logits.device
    left_right = left_right.detach()

    left_log_prob = F.logsigmoid(logits)  # shape [batch, n]
    right_log_prob = -logits + left_log_prob  # shape [batch, n]

    log_probs = torch.where(left_right, left_log_prob, right_log_prob)  # shape [batch, n]
    log_probs = log_probs.sum(-1)  # shape [batch]

    # identity permutation
    mask = torch.ones(n + 1, n, device=device)
    mask = torch.tril(mask, diagonal=-1).bool()
    identity_log_probs = torch.where(
        mask, left_log_prob.unsqueeze(-2), right_log_prob.unsqueeze(-2)
    )  # shape [batch, n+1, n]
    identity_log_probs = identity_log_probs.sum(-1)
    identity_log_probs = torch.logsumexp(identity_log_probs, dim=-1)  # shape [batch]

    # is identity iff left_right is in the form of 1...10...0, which happens iff non-increasing
    is_identity = (torch.diff(left_right.int(), dim=-1) >= 0).all(dim=-1)  # shape [batch]

    return torch.where(is_identity, identity_log_probs, log_probs)


@torch.no_grad()
def log_prob_riffle_shuffle(x_tm1, x_t, guarantee_reachable=False):
    """
    Args:
        x_tm1: shape [batch_shape, n, c, h, w]
        x_t: shape [batch_shape, n, c, h, w]

        x_tm1 --riffle shuffle--> x_t, assume uniform riffle shuffle

    Returns:
        shape [batch_shape]
    """
    n = x_t.size(-4)

    log_prob = -n * torch.log(torch.tensor(2))  # 1 / 2^n

    is_identity = (~torch.isclose(x_tm1, x_t)).sum(dim=(-1, -2, -3, -4)) == 0  # shape [batch]
    identity_log_prob = torch.log(torch.tensor(n + 1)) - n * torch.log(
        torch.tensor(2)
    )  # (n + 1) / 2^n

    log_prob = torch.where(is_identity, identity_log_prob, log_prob)

    if guarantee_reachable:
        return log_prob

    perm_list = utils.find_perm_images(x_tm1, x_t)
    inverse_perm_list = torch.argsort(perm_list)  # count rising sequences
    rises = ~(inverse_perm_list[..., :-1] < inverse_perm_list[..., 1:])
    reachable_perms = rises.sum(-1) <= 1

    log_prob = torch.where(reachable_perms, log_prob, float("-inf"))

    return log_prob


@torch.no_grad()
def log_prob_t_riffle_shuffle(perm, t):
    """
    Args:
        perm: shape [batch, n]
        t: shape [batch]

    Returns:
        shape: [batch]
    """
    n = perm.size(-1)
    perm = torch.argsort(perm)  # ! important!
    r = utils.count_rising_sequence(perm)
    # logger.debug(f"t = {t[0]}")
    # logger.debug(f"r = {r[0, 0]}")
    log_prob = -n * math.log(2) * t + utils.log_binom(n + 2**t - r, torch.tensor(n))
    return log_prob


@torch.no_grad()
def log_prob_riffle_shuffle_between_images(image_from, image_to, t):
    """
    Args:
        image_from: shape [batch, n, c, h, w]
        image_to: shape [batch, n, c, h, w]
        t: shape [batch]

    Returns:
        shape: [batch]
    """
    perm = utils.find_perm_images(image_from, image_to)
    return log_prob_t_riffle_shuffle(perm, t)
