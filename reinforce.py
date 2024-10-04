import torch

import PL_distribution as PL
import riffle_shuffle as RS
import utils


def log_prob(
    scores,
    x_tm1,
    x_t,
    reinforce_N,
    ema,
    ema_rate,
    distribution="generalized_PL",
    entropy_reg_rate=0.1,
):
    """
    (\sum_{n=1}^{N} p(X_{i-1} | X_i, \sigma_i^{(n)}) \log p(\sigma_i^{(n)} | X_i)) / (\sum_{n=1}^{N} p(X_{i-1} | X_i, \sigma_i^{(n)}))

    Args:
        scores: shape [batch_shape, n, n]
        x_tm1: shape [batch_shape, n, c, h, w]
        x_t: shape [batch_shape, n, c, h, w]
        reinforce_N: int
        distribution: "PL" | "generalized_PL" | "riffle" | "swap" | "insert"
    """
    expanded_scores = scores.expand(reinforce_N, *scores.shape)
    expanded_x_t = x_t.expand(reinforce_N, *x_t.shape)

    if distribution == "generalized_PL":
        sampled_perms = PL.sample_generalized_PL(
            expanded_scores
        )  # shape [reinforce_N, batch_shape, n]
        log_prob_perm = PL.log_prob_generalized_PL(
            scores, sampled_perms
        )  # shape [reinforce_N, batch_shape]
        mean = utils.permute_image_perm_list(sampled_perms, expanded_x_t)

    elif distribution == "PL":
        sampled_perms = PL.sample(scores, reinforce_N)  # shape [reinforce_N, batch_shape, n]
        log_prob_perm = PL.log_prob(expanded_scores, sampled_perms)
        mean = utils.permute_image_perm_list(sampled_perms, expanded_x_t)

    elif distribution == "riffle":
        mean, sampled_perms = RS.sample_inverse_riffle_shuffle(expanded_scores, expanded_x_t)
        log_prob_perm = RS.log_prob_inverse_riffle_shuffle_indices(expanded_scores, sampled_perms)

    elif distribution == "insert":
        mean, sampled_perms = PL.sample_insertion_to_back(
            expanded_scores, expanded_x_t, x_images=True
        )
        log_prob_perm = PL.log_prob_insertion_to_back_indices(expanded_scores, sampled_perms)

    elif distribution == "swap":
        logits_swap, logit_unchanged = torch.split(expanded_scores, [x_t.size(-4), 1], dim=-1)
        logit_unchanged = logit_unchanged.squeeze(-1)
        mean, sampled_perms = PL.sample_lazy_swap(logits_swap, logit_unchanged, expanded_x_t)
        log_prob_perm = PL.log_prob_lazy_swap_indices(logits_swap, logit_unchanged, sampled_perms)

    else:
        raise NotImplementedError

    log_prob_given_image_and_perm = utils.log_prob_normal_dist_images(x_tm1, mean, no_const=True)

    weights = torch.softmax(log_prob_given_image_and_perm, dim=0)  # [reinforce_N, batch_shape]

    # Entropy regularization
    weights = weights - entropy_reg_rate * log_prob_perm.detach()

    ema = ema_rate * ema + (1 - ema_rate) * weights.mean()
    weights = weights - ema
    res = (weights * log_prob_perm).sum(0)  # shape [batch_shape]

    return res, ema
