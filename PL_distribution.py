import math
from itertools import permutations

import torch
import torch.nn.functional as F

import utils


def perm_likelihood_nonmat(s, pi):
    n = s.size(-1)
    likelihood = 0
    for j in range(n):
        rest = []
        for k in range(j, n):
            rest.append(s[pi[k]])
        likelihood -= s[pi[j]] - torch.logsumexp(torch.stack(rest), 0)
    return likelihood


# =============================================================================
# Sampling
# =============================================================================


@torch.no_grad()
def sample(scores, N, deterministic=False):
    """
    Args:
        scores: shape [batch_shape, n]
        N: int
        deterministic: bool, True means sampling the mode of the distribution

    Returns:
        shape [N, batch_shape, n]
    """
    # logger.debug("In PL sample, not beam search")
    device = scores.device
    scores = scores.unsqueeze(0).expand(N, *scores.size())

    u = torch.rand_like(scores).to(device)  # shape [N, *, n]
    u = torch.clamp(u, min=torch.finfo(u.dtype).tiny, max=1.0)

    gumbel = scores - (not deterministic) * torch.log(-torch.log(u))
    samples = torch.argsort(gumbel, descending=True)

    return samples


@torch.no_grad()
def sample_generalized_PL(scores, deterministic=False):
    """
    Args:
        scores: shape [batch, n, n]

    Returns:
        shape [batch, n]
    """
    device = scores.device
    batch_shape = scores.shape[:-2]
    n = scores.size(-1)

    u = torch.rand_like(scores).to(device)  # shape [batch, n, n]
    u = torch.clamp(u, min=torch.finfo(u.dtype).tiny, max=1.0)
    scores = scores - (not deterministic) * torch.log(-torch.log(u))

    selected_mask = torch.zeros((*batch_shape, n), device=device)  # [batch, n]
    samples = torch.zeros((*batch_shape, 0), device=device)  # [batch, 0]

    for i in range(n):
        cur_scores = scores[..., i, :] + selected_mask  # [batch, n]
        selected = torch.argmax(cur_scores, -1, keepdim=True)  # [batch, 1]
        samples = torch.cat([samples, selected], dim=-1)  # [batch, i+1]
        selected_mask = torch.where(
            torch.arange(n, device=device) == selected, float("-inf"), selected_mask
        )

    return samples.long()


@torch.no_grad()
def sample_generalized_PL_beam_search(scores, beam_size):
    """
    Args:
        scores: shape [batch, n, n]
        beam_size: int

    Returns:
        result: shape [batch, beam_size, n]
        result_log_probs: shape [batch, beam_size]
    """
    device = scores.device
    batch_shape = scores.shape[:-2]
    n = scores.size(-1)

    first_scores = scores[..., 0, :]  # [batch, n]
    first_log_probs = F.log_softmax(first_scores, -1)  # [batch, n]
    result_log_probs, result = torch.topk(
        first_log_probs, k=min(beam_size, n), dim=-1
    )  # [batch, beam]
    result = result.unsqueeze(-1)  # [batch, beam, 1]

    for i in range(1, n):
        selected_mask = (result.unsqueeze(-1) == torch.arange(n, device=device)).any(
            dim=-2
        )  # [batch, beam, n]
        selected_mask = torch.where(selected_mask, float("-inf"), 0)  # [batch, beam, n]
        cur_scores = scores[..., [i], :] + selected_mask  # [batch, beam, n]
        cur_log_probs = F.log_softmax(cur_scores, -1)  # [batch, beam, n]

        candidates_log_probs, candidates_idx = torch.topk(
            cur_log_probs, k=min(beam_size, n), dim=-1
        )  # [batch, beam, beam]
        candidates_idx = candidates_idx.unsqueeze(-1)  # [batch, beam, beam, 1]

        result_expanded = result.unsqueeze(-2).expand(
            *batch_shape, -1, min(beam_size, n), -1
        )  # [batch, beam, beam, i]
        candidates_idx = torch.cat((result_expanded, candidates_idx), dim=-1)
        candidates_idx = candidates_idx.flatten(start_dim=-3, end_dim=-2)  # [batch, beam^2, i+1]

        candidates_log_probs = (
            result_log_probs.unsqueeze(-1) + candidates_log_probs
        )  # [batch, beam, beam]
        candidates_log_probs = candidates_log_probs.flatten(start_dim=-2)  # [batch, beam^2]

        num_selected = min(beam_size, candidates_log_probs.size(-1))
        result_log_probs, topk_idx = torch.topk(
            candidates_log_probs, k=num_selected, dim=-1
        )  # [batch, beam]
        topk_idx_expanded = topk_idx.unsqueeze(-1).expand(*batch_shape, -1, i + 1)
        result = torch.gather(candidates_idx, -2, topk_idx_expanded)  # [batch, beam, i+1]

    return result.long(), result_log_probs


@torch.no_grad()
def sample_PL_beam_search(scores, beam_size):
    """
    Args:
        scores: shape [batch, n]
        beam_size: int

    Returns:
        result: shape [batch, beam_size, n]
        result_log_probs: shape [batch, beam_size]
    """
    n = scores.size(-1)
    scores_expanded = scores.unsqueeze(-2).expand(*scores.shape[:-1], n, -1)
    return sample_generalized_PL_beam_search(scores_expanded, beam_size)


@torch.no_grad()
def sample_swap(scores, x, deterministic=False, x_images=False):
    """
    Samples two elements from x using PL(scores) and swap the two elements

    Args:
        scores: logits, shape [batch_shape, n]
        x: if x_images: shape [batch_shape, n, c, h, w], else: shape [batch_shape, n]
        deterministic: bool, True means sampling the mode of the distribution

    Returns:
        x_swapped: shape: if x_images: shape [batch_shape, n, c, h, w], else: shape [batch_shape, n]
        indices: shape [batch_shape, 2]
    """
    if x_images:
        noise_shape = x.shape[:-3]
    else:
        noise_shape = x.shape

    noise = torch.rand(size=noise_shape).to(scores.device)
    noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.0)
    gumbel_noise = -torch.log(-torch.log(noise))

    _, indices = torch.topk(
        scores + (not deterministic) * gumbel_noise, 2, dim=-1
    )  # shape [batch_shape, 2]

    if x_images:
        x_swapped = utils.swap_by_idx_images(x, indices)
    else:
        x_swapped = utils.swap_by_idx(x, indices)

    return x_swapped, indices


@torch.no_grad()
def sample_lazy_swap(scores, logit_unchanged, x, deterministic=False, x_images=False):
    """
    Samples two elements from x using PL(scores) and swap the two elements

    Args:
        scores: logits, shape [batch_shape, n]
        logit_unchanged: logits, shape [batch_shape]
        x: shape [batch_shape, n, c, h, w]
        deterministic: bool, True means sampling the mode of the distribution

    Returns:
        shape [batch_shape, n]
    """
    if x_images:
        non_batch_dims = 4
    else:
        non_batch_dims = 1

    device = scores.device
    batch_shape = x.shape[:-non_batch_dims]
    rand = torch.rand(size=batch_shape, device=device)

    prob_unchanged = torch.sigmoid(logit_unchanged)  # [batch_shape]
    log_prob_unchanged = F.logsigmoid(logit_unchanged)
    log_prob_changed = -logit_unchanged + log_prob_unchanged

    x_swapped, swapped_indices = sample_swap(
        scores, x, deterministic, x_images=x_images
    )  # [batch_shape, n, c, h, w]
    identity_swap = torch.tensor(0, device=device)

    if not deterministic:
        final_sample = torch.where(
            (rand > prob_unchanged)[(...,) + (None,) * non_batch_dims], x_swapped, x
        )
        final_swapped_indices = torch.where(
            (rand > prob_unchanged).unsqueeze(-1), swapped_indices, identity_swap
        )
    else:
        log_prob_mode = log_prob_swap_indices(scores, swapped_indices)
        mask = log_prob_unchanged >= (log_prob_changed + log_prob_mode)  # [batch_shape]
        final_sample = torch.where(mask[(...,) + (None,) * non_batch_dims], x, x_swapped)
        final_swapped_indices = torch.where(mask.unsqueeze(-1), identity_swap, swapped_indices)

    return final_sample, final_swapped_indices


@torch.no_grad()
def sample_swap_with_replacement(scores, x, deterministic=False, x_images=False):
    """
    Args:
        scores: logits, shape [batch_shape, n]
        x: shape [batch_shape, n]
    """
    if deterministic:
        raise NotImplementedError
    if x_images:
        non_batch_dims = 4
    else:
        non_batch_dims = 1

    noise_shape = x.shape[:-non_batch_dims] + (2, x.size(-non_batch_dims))  # [batch_shape, 2, n]
    noise = torch.rand(size=noise_shape).to(scores.device)
    noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.0)
    gumbel_noise = -torch.log(-torch.log(noise))

    indices = torch.argmax(scores.unsqueeze(-2) + gumbel_noise, -1)  # [batch_shape, 2]
    if x_images:
        x_swapped = utils.swap_by_idx_images(x, indices)
    else:
        x_swapped = utils.swap_by_idx(x, indices)

    return x_swapped, indices


@torch.no_grad()
def sample_insertion_from_back(scores, x, deterministic=False, x_images=False):
    """
    Args:
        scores: logits, shape [batch_shape, m]
        x: if x_images: shape [batch_shape, n, c, h, w], else: shape [batch_shape, n]
        deterministic: bool, True means sampling the mode of the distribution

    Returns:
        shape [batch_shape, n]
    """
    if x_images:
        batch_shape = x.shape[:-4]
    else:
        batch_shape = x.shape[:-1]

    noise = torch.rand(*batch_shape, scores.size(-1)).to(scores.device)
    noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.0)
    gumbel_noise = -torch.log(-torch.log(noise))

    indices = torch.argmax(scores + (not deterministic) * gumbel_noise, -1)

    if x_images:
        rearranged = utils.insert_back_to_idx_images(x, indices)
    else:
        rearranged = utils.insert_back_to_idx(x, indices)

    return rearranged, indices


@torch.no_grad()
def sample_insertion_to_back(scores, x, deterministic=False, x_images=False):
    """
    Args:
        scores: logits, shape [batch_shape, n-1]
        x: if x_images: shape [batch_shape, n, c, h, w], else: shape [batch_shape, n]
        deterministic: bool, True means sampling the mode of the distribution

    Returns:
        (shape [batch_shape, n], shape [batch_shape])
    """
    if x_images:
        batch_shape = x.shape[:-4]
    else:
        batch_shape = x.shape[:-1]

    noise = torch.rand(*batch_shape, scores.size(-1)).to(scores.device)
    noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.0)
    gumbel_noise = -torch.log(-torch.log(noise))

    indices = torch.argmax(scores + (not deterministic) * gumbel_noise, -1)

    if x_images:
        rearranged = utils.insert_idx_to_back_images(x, indices)
    else:
        rearranged = utils.insert_idx_to_back(x, indices)

    return rearranged, indices


# =============================================================================
# Log Probabilities
# =============================================================================


def log_prob(scores, perm_list, tau=1.0, topk=None):
    """
    Computes log p(perm_list) where perm_list is sampled from PL(scores)
    Args:
        scores: Tensor, shape [batch_shape, n]
        perm_list: Tensor, shape [batch_shape, k], k <= n
    Returns:
        shape [batch_shape]
    """
    n = scores.size(-1)
    device = scores.device

    if topk is None:
        topk = perm_list.size(-1)

    if perm_list.size(-1) < n:
        perm_list = utils.complete_range(perm_list, n)

    perm_mat = utils.perm_list_to_mat(perm_list).detach()
    perm_s_vec = torch.matmul(
        perm_mat, scores.unsqueeze(-1)
    )  # shape [batch_shape, n, 1], [s_{pi(1)}, ..., s_{pi(n)}], has grad
    perm_s_mat = perm_s_vec.repeat(*((1,) * scores.ndim), n)  # shape [batch_shape, n, n], has grad

    mask = torch.triu(torch.ones(n, n, device=device), diagonal=1)
    perm_s_mat[..., mask == 1] = float("-inf")

    result = perm_s_vec.squeeze(-1) - tau * torch.logsumexp(perm_s_mat / tau, -2)
    result = result[..., :topk].sum(-1)

    return result


def log_prob_generalized_PL(scores, perm_list):
    """
    Args:
        scores: shape [batch, n, n]
        perm_list: shape [batch, n]

    Returns:
        shape [batch]
    """
    n = scores.size(-1)
    device = scores.device

    perm_s_mat = utils.permute_list(perm_list.unsqueeze(-2), scores)  # [batch, n, n]
    diag = torch.diagonal(perm_s_mat, dim1=-2, dim2=-1)  # [batch, n]

    mask = torch.tril(torch.ones(n, n, device=device), diagonal=-1)
    perm_s_mat[..., mask == 1] = float("-inf")

    lse = torch.logsumexp(perm_s_mat, -1)  # [batch, n]
    result = (diag - lse).sum(-1)

    return result


def log_prob_images(scores, x_tm1, x_t):
    """
    Assume x_tm1 is obtained by applying some permutation to x_t,
    i.e. x_t --perm--> x_tm1.
    Find the log prob of the permutation under PL(scores).

    Args:
        scores: shape [batch_shape, n]
        x_tm1: shape [batch_shape, n, c, h, w]
        x_t: shape [batch_shape, n, c, h, w]

    Returns:
        shape [batch_shape]
    """
    perm = utils.find_perm_images(x_t, x_tm1)
    return log_prob(scores, perm)


def log_prob_perms(scores, perm_tm1, perm_t):
    """
    Assume perm_tm1 is obtained by applying some permutation to perm_t,
    i.e. perm_t --perm--> perm_tm1.
    Find the log prob of the permutation under PL(scores).

    Args:
        scores: shape [batch_shape, n, n]
        perm_tm1: shape [batch_shape, n]
        perm_t: shape [batch_shape, n]

    Returns:
        shape [batch_shape]
    """
    perm = utils.find_perm(perm_t, perm_tm1)
    return log_prob(scores, perm)


def log_prob_images_generalized_PL(scores, x_tm1, x_t):
    """
    Assume x_tm1 is obtained by applying some permutation to x_t,
    i.e. x_t --perm--> x_tm1.
    Find the log prob of the permutation under generalized_PL(scores).

    Args:
        scores: shape [batch_shape, n, n]
        x_tm1: shape [batch_shape, n, c, h, w]
        x_t: shape [batch_shape, n, c, h, w]

    Returns:
        shape [batch_shape]
    """
    perm = utils.find_perm_images(x_t, x_tm1)
    return log_prob_generalized_PL(scores, perm)


def log_prob_perms_generalized_PL(scores, perm_tm1, perm_t):
    """
    Assume perm_tm1 is obtained by applying some permutation to perm_t,
    i.e. perm_t --perm--> perm_tm1.
    Find the log prob of the permutation under generalized_PL(scores).

    Args:
        scores: shape [batch_shape, n, n]
        perm_tm1: shape [batch_shape, n]
        perm_t: shape [batch_shape, n]

    Returns:
        shape [batch_shape]
    """
    perm = utils.find_perm(perm_t, perm_tm1)
    return log_prob_generalized_PL(scores, perm)


def log_prob_swap(scores, x_tm1, x_t, x_images=False):
    """
    Args:
        x_images: bool
        scores: shape [batch_shape, n]
        x_tm1: if x_images shape [batch_shape, n, c, h, w], else shape [batch_shape, n]
        x_t: if x_images shape [batch_shape, n, c, h, w], else shape [batch_shape, n]
        Each row of the batch dimension of x_tm1 and x_t should differ in exactly one swap

    Returns:
        shape [batch_shape]
    """
    if not x_images:
        batch_shape = x_tm1.shape[:-1]
        diff = x_tm1 != x_t
    else:
        batch_shape = x_tm1.shape[:-4]
        diff = (~torch.isclose(x_tm1, x_t)).sum(dim=(-1, -2, -3)) > 0

    diff_idx = torch.nonzero(diff, as_tuple=True)[-1]
    diff_idx = diff_idx.reshape(*batch_shape, 2)
    diff_idx_flip = torch.flip(diff_idx, [-1])

    top2_log_prob = log_prob(scores, diff_idx)
    top2_log_prob_flip = log_prob(scores, diff_idx_flip)

    top2_log_probs = torch.stack((top2_log_prob, top2_log_prob_flip))  # [2, *]
    return torch.logsumexp(top2_log_probs, 0)


def log_prob_swap_indices(scores, swapped_indices):
    """
    Args:
        scores: shape [batch_shape, n]
        swapped_indices: shape [batch_shape, 2]

    Returns:
        shape [batch_shape]
    """
    swapped_indices_flip = torch.flip(swapped_indices, [-1])

    top2_log_prob = log_prob(scores, swapped_indices)
    top2_log_prob_flip = log_prob(scores, swapped_indices_flip)

    top2_log_probs = torch.stack((top2_log_prob, top2_log_prob_flip))  # [2, *]
    return torch.logsumexp(top2_log_probs, 0)


def log_prob_lazy_swap(scores, logit_unchanged, x_tm1, x_t):
    """
    Args:
        scores: shape [batch_shape, n]
        logit_unchanged: shape [batch_shape]
        x_tm1: shape [batch_shape, n, c, h, w]
        x_t: shape [batch_shape, n, c, h, w]

    Returns:
        shape [batch_shape]
    """
    log_prob_unchanged = F.logsigmoid(logit_unchanged)
    log_prob_changed = -logit_unchanged + log_prob_unchanged

    diff = ((~torch.isclose(x_tm1, x_t)).sum(dim=(-1, -2, -3)) > 0).int()  # [batch_shape, n]
    unchanged_mask = (diff.sum(-1) == 0).detach()
    reachable_mask = (diff.sum(-1) <= 2).detach()

    _, diff_idx = torch.topk(diff, 2, dim=-1)  # [batch_shape, 2]
    diff_idx_flip = torch.flip(diff_idx, [-1])

    top2_log_prob = log_prob(scores, diff_idx)
    top2_log_prob_flip = log_prob(scores, diff_idx_flip)

    top2_log_probs = torch.stack((top2_log_prob, top2_log_prob_flip))
    log_prob_swapped = torch.logsumexp(top2_log_probs, 0) + log_prob_changed  # [batch_shape]

    result_reachable = torch.where(unchanged_mask, log_prob_unchanged, log_prob_swapped)

    result = torch.where(reachable_mask, result_reachable, float("-inf"))

    return result


def log_prob_lazy_swap_indices(scores, logit_unchanged, swapped_indices):
    """
    Args:
        scores: shape [batch_shape, n]
        logit_unchanged: shape [batch_shape]
        swapped_indices: shape [batch_shape, 2], if identity swap, then [0, 0]

    Returns:
        shape [batch_shape]
    """
    log_prob_unchanged = F.logsigmoid(logit_unchanged)
    log_prob_changed = -logit_unchanged + log_prob_unchanged

    unchanged_mask = (swapped_indices.sum(-1) == 0).detach()  # [batch_shape]
    # make the unchanged rows [0, 1] instead of [0, 0] so that utils.complete_range in log_prob works
    modify_unchanged = torch.stack(
        [torch.zeros_like(unchanged_mask), unchanged_mask], dim=-1
    )  # [batch_shape, 2]

    swapped_indices_flip = torch.flip(swapped_indices, [-1])

    top2_log_prob = log_prob(scores, swapped_indices + modify_unchanged)
    top2_log_prob_flip = log_prob(scores, swapped_indices_flip + modify_unchanged)

    top2_log_probs = torch.stack((top2_log_prob, top2_log_prob_flip))
    log_prob_swapped = torch.logsumexp(top2_log_probs, 0) + log_prob_changed  # [batch_shape]

    result = torch.where(unchanged_mask, log_prob_unchanged, log_prob_swapped)

    return result


def log_prob_swap_with_replacement(scores, x_tm1, x_t):
    """
    Args:
        scores: shape [batch_shape, n]
        x_tm1: shape [batch_shape, n, c, h, w]
        x_t: shape [batch_shape, n, c, h, w]

    Returns:
        shape [batch_shape]
    """
    log_softmax_scores = F.log_softmax(scores, dim=-1)
    log_prob_unchanged = torch.logsumexp(2 * log_softmax_scores, dim=-1)

    diff = ((~torch.isclose(x_tm1, x_t)).sum(dim=(-1, -2, -3)) > 0).int()  # [batch_shape, n]
    unchanged_mask = (diff.sum(-1) == 0).detach()

    _, diff_idx = torch.topk(diff, 2, dim=-1)  # [batch_shape, 2]
    # s = torch.gather(scores, -1, diff_idx.detach()) # [batch_shape, 2]
    # log_prob_swapped = torch.log(torch.tensor(2)) + s.sum(-1) - 2 * torch.logsumexp(scores, -1)
    log_prob_idx = torch.gather(log_softmax_scores, -1, diff_idx)  # [batch_shape, 2]
    log_prob_swapped = math.log(2) + log_prob_idx.sum(-1)

    result = torch.where(unchanged_mask, log_prob_unchanged, log_prob_swapped)

    return result


def log_prob_insertion_to_back(scores, x_tm1, x_t, x_images=False):
    """
    Args:
        scores: shape [batch_shape, n]
        x_tm1: shape [batch_shape, n]
        x_t: shape [batch_shape, n]
        Each row of the last dimension of x_tm1 and x_t should be the same or differ in exactly one insertion

    Returns:
        shape [batch_shape]
    """
    if not x_images:
        diff = (x_tm1 != x_t).int()
    else:
        diff = ((~torch.isclose(x_tm1, x_t)).sum(dim=(-1, -2, -3)) > 0).int()

    diff[..., -1] = 1  # unchanged means we chose the last element
    indices = torch.argmax(diff, -1, keepdim=True).detach()  # [N, T, bs, 1]

    s = torch.gather(scores, -1, indices).squeeze(-1)  # [batch_shape]
    lse = torch.logsumexp(scores, -1)  # [batch_shape]
    result = s - lse

    return result


def log_prob_insertion_to_back_indices(scores, indices):
    """
    Args:
        scores: shape [batch_shape, n]
        indices: shape [batch_shape]

    Returns:
        shape [batch_shape]
    """
    indices = indices.unsqueeze(-1).detach()

    s = torch.gather(scores, -1, indices).squeeze(-1)  # [batch_shape]
    lse = torch.logsumexp(scores, -1)  # [batch_shape]
    result = s - lse

    return result
