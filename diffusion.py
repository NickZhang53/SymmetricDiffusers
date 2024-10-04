import torch
import torch.nn as nn

import PL_distribution as PL
import reinforce
import riffle_shuffle as RS
import utils


class DiffusionUtils(nn.Module):
    """Discrete state space diffusion process.

    Time convention: noisy data is labeled x_0, ..., x_{T-1}, and original data
    is labeled x_start (or x_{-1}). This convention differs from the papers,
    which use x_1, ..., x_T for noisy data and x_0 for original data.
    """

    def __init__(
        self,
        num_timesteps,
        sample_N,
        transition,
        latent,
        reinforce_N,
        reinforce_ema_rate,
        entropy_reg_rate,
        reverse,
        reverse_steps,
        loss,
        beam_size,
        perm_fix_first,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps  # T
        self.sample_N = sample_N  # N
        self.transition = transition
        self.reverse = reverse
        self.reverse_steps = (
            torch.arange(num_timesteps + 1).tolist() if reverse_steps == [] else reverse_steps
        )

        self.latent = latent
        self.reinforce_N = reinforce_N
        self.reinforce_ema_rate = reinforce_ema_rate
        self.ema = torch.tensor(0)

        self.loss = "log_likelihood" if loss is None else loss

        self.entropy_reg_rate = entropy_reg_rate

        self.eps = 1e-6

        self.PL_beam_size = beam_size["PL"]
        self.t_beam_size = beam_size["time"]
        assert self.t_beam_size <= self.PL_beam_size

        # if perm_fix_first is True, then the first pos of every permutation is always 0
        self.perm_fix_first = perm_fix_first

    @torch.no_grad()
    def q_sample(self, x_prev, t, x_images=False):
        """Sample from q(x_t | x_{t-1})
        (i.e. randomly swap two elements).

        Args:
          x_prev: x_{t-1}, shape [N, bs, n]
          t: timestep of the diffusion process, shape [N, bs]

        Returns:
          shape Tuple ([N, bs, n], [N, bs, n])
        """
        if x_images:
            non_batch_dims = 4
        else:
            non_batch_dims = 1

        n = x_prev.size(-non_batch_dims)
        device = x_prev.device

        if self.transition == "swap":
            uniform_logits = torch.zeros(n, device=device)
            sample_result, sampled_perm = PL.sample_swap_with_replacement(
                uniform_logits, x_prev, x_images=x_images
            )

        elif self.transition == "insert":
            uniform_logits = torch.zeros(n, device=device)
            sample_result, sampled_perm = PL.sample_insertion_from_back(
                uniform_logits, x_prev, x_images=x_images
            )

        elif self.transition == "riffle":
            sample_result, sampled_perm = RS.sample_riffle_shuffle(x_prev, x_images=x_images)

        else:
            raise NotImplementedError

        return sample_result, sampled_perm

    @torch.no_grad()
    def q_sample_seq(self, x_start, x_images=False):
        """Sample from the forward Markov chain
        (i.e. add noise to the data).

        Args:
            x_start: shape [bs, n]

        Returns:
            shape [N, T+1, bs, n]
        """
        if self.perm_fix_first:
            assert not x_images
            x_start = x_start[..., 1:]

        x_start = x_start.expand(self.sample_N, *x_start.shape)
        result = [x_start]
        for t in range(self.num_timesteps):
            x_prev = result[-1]
            x_t, sampled_perm = self.q_sample(
                x_prev, t, x_images=x_images
            )  # shape [N, bs, n, c, h, w]
            result.append(x_t)

        # apply perms[i] to result[i] to get result[i + 1]

        if x_images:
            result = torch.stack(result, -6)  # shape [N, T+1, bs, n, c, h, w]
        else:
            result = torch.stack(result, -3)  # shape [N, T+1, bs, n]

        if self.perm_fix_first:
            result = utils.add_zero_to_perm(result, add1=False)

        return result

    def p_logits(self, reverse_model, x, t, x_start):
        """
        Compute logits of p(x_{t-1} | x_t).
        """
        return reverse_model(x, t, x_start)

    def p_log_cond_prob_latent(self, scores, x_tm1, x_t):
        """
        Computes log p_{theta}(x_tm1 | x_t)

        Args:
            scores: shape [batch_shape, n]
            x_tm1: shape [batch_shape, n, c, h, w]
            x_t: shape [batch_shape, n, c, h, w]
            Each row of the last dimension of x_tm1 and x_t should differ in exactly one transition

        Returns:
            shape [batch_shape]
        """
        n = x_tm1.size(-4)

        reverse_method = self.transition if self.reverse == "original" else self.reverse

        result, new_ema = reinforce.log_prob(
            scores,
            x_tm1,
            x_t,
            self.reinforce_N,
            self.ema,
            self.reinforce_ema_rate,
            reverse_method,
            self.entropy_reg_rate,
        )
        self.ema = new_ema

        return result

    def p_log_cond_prob_images(self, scores, x_tm1, x_t):
        """
        Computes log p_{theta}(x_tm1 | x_t)

        Args:
            scores: shape [batch_shape, n]
            x_tm1: shape [batch_shape, n, c, h, w]
            x_t: shape [batch_shape, n, c, h, w]
            Each row of the last dimension of x_tm1 and x_t should differ in exactly one transition

        Returns:
            shape [batch_shape]
        """
        n = x_tm1.size(-4)

        if self.reverse == "PL":
            result = PL.log_prob_images(scores, x_tm1, x_t)

        elif self.reverse == "generalized_PL":
            result = PL.log_prob_images_generalized_PL(scores, x_tm1, x_t)

        elif self.transition == "swap":
            logits_swap, logit_unchanged = torch.split(scores, [n, 1], dim=-1)
            logit_unchanged = logit_unchanged.squeeze(-1)
            result = PL.log_prob_lazy_swap(logits_swap, logit_unchanged, x_tm1, x_t)

        elif self.transition == "insert":
            result = PL.log_prob_insertion_to_back(scores, x_tm1, x_t, x_images=True)

        elif self.transition == "riffle":
            result = RS.log_prob_inverse_riffle_shuffle(scores, x_tm1, x_t)

        else:
            raise NotImplementedError

        return result

    def p_log_cond_prob(self, scores, perm_tm1, perm_t):
        """
        Computes log p_{theta}(x_tm1 | x_t)

        Args:
            scores: shape [batch_shape, n]
            perm_tm1: shape [batch_shape, n]
            perm_t: shape [batch_shape, n]

        Returns:
            shape [batch_shape]
        """
        n = perm_tm1.size(-1)

        if self.reverse == "generalized_PL":
            result = PL.log_prob_perms_generalized_PL(scores, perm_tm1, perm_t)

        elif self.reverse == "PL":
            result = PL.log_prob_perms(scores, perm_tm1, perm_t)

        else:
            raise NotImplementedError

        return result

    # =============================================================================
    # Sampling
    # =============================================================================

    @torch.no_grad()
    def p_sample(self, reverse_model, x, t, x_start, deterministic):
        """Sample one timestep from the model p(x_{t-1} | x_t) by swapping two elements of x_t

        Args:
            x: shape [bs, n]
            t: shape [bs]
            x_start: shape [bs, n, c, h, w]

        Retunrs:
            shape [bs, n]
        """
        n = x.size(-1)
        model_logits = self.p_logits(
            reverse_model, x.unsqueeze(-2), t, x_start
        ).squeeze()  # [bs, n]

        if self.reverse == "PL":
            sample_indices = PL.sample(model_logits, 1, deterministic=deterministic).squeeze(0)
            sample_result = utils.permute_int_list(sample_indices, x)

        elif self.reverse == "generalized_PL":
            if self.perm_fix_first:
                model_logits = utils.mask_scores_pos_zero(model_logits)
            sample_indices = PL.sample_generalized_PL(model_logits, deterministic=deterministic)
            sample_result = utils.permute_int_list(sample_indices, x)

        elif self.transition == "swap":
            logits_swap, logit_unchanged = torch.split(model_logits, [n, 1], dim=-1)
            logit_unchanged = logit_unchanged.squeeze(-1)
            sample_result, sample_indices = PL.sample_lazy_swap(
                logits_swap, logit_unchanged, x, deterministic=deterministic
            )

        elif self.transition == "insert":
            sample_result, sample_indices = PL.sample_insertion_to_back(
                model_logits, x, deterministic=deterministic
            )

        elif self.transition == "riffle":
            sample_result, sample_indices = RS.sample_inverse_riffle_shuffle_perms(
                model_logits, x, deterministic=deterministic
            )

        return sample_result, sample_indices, model_logits

    @torch.no_grad()
    def p_sample_loop(self, input, reverse_model, deterministic):
        """Sampling.

        Args:
            input: shape [bs, n, c, h, w]
            reverse_model: function, reverse network

        Returns:
            x: shape [bs, n]
        """
        device = input.device
        batch = input.shape[0]
        n = input.shape[1]

        perm = torch.arange(n, device=device).expand(batch, -1)

        for i in reversed(self.reverse_steps[1:]):
            t = torch.full((batch,), i, device=device)
            perm, sample_indices, model_logits = self.p_sample(
                reverse_model, perm, t, input, deterministic
            )

        if input.dim() == 5:
            result_x = utils.permute_image_perm_list(perm, input)
        elif input.dim() == 2:
            result_x = utils.permute_int_list(perm, input)
        elif input.dim() == 3:
            result_x = utils.permute_embd(perm, input)
        else:
            raise NotImplementedError

        return result_x, perm

    @torch.no_grad()
    def p_sample_beam_search(self, input, reverse_model):
        """Sampling by beam search

        Args:
            input: original input image that we need to unscramble
                   shape [b, num_pieces**2, c, h, w]
            reverse_model: function, reverse network

        Returns:
            result_x: shape [batch, beam_size, n, c, h, w]
            result_perm: shape [batch, beam_size, n]
        """
        device = input.device
        batch = input.shape[0]
        image_shape = input.shape[1:]
        n = image_shape[0]

        if self.reverse == "generalized_PL":
            sample_distribution_beam_search = PL.sample_generalized_PL_beam_search
        elif self.reverse == "PL":
            sample_distribution_beam_search = PL.sample_PL_beam_search
        else:
            raise NotImplementedError

        # Do first step at time T
        T = torch.full((batch,), self.num_timesteps).to(device)
        identity_perm = torch.arange(n, device=device).expand(batch, 1, -1)
        model_logits = self.p_logits(reverse_model, identity_perm, T, input)  # [batch, 1, n, n]
        model_logits = model_logits.squeeze(1)
        if self.perm_fix_first:
            model_logits = utils.mask_scores_pos_zero(model_logits)

        result_perm, result_log_probs = sample_distribution_beam_search(
            model_logits, self.PL_beam_size
        )
        # result_perm: shape [batch, PL_beam, n]
        # result_log_probs: shape [batch, PL_beam]
        result_perm = result_perm[..., : self.t_beam_size, :]  # [batch, t_beam, n]
        result_log_probs = result_log_probs[..., : self.t_beam_size]  # [batch, t_beam]

        for i in reversed(self.reverse_steps[1:-1]):
            t = torch.full((batch,), i).to(device)
            model_logits = self.p_logits(
                reverse_model, result_perm, t, input
            )  # [batch, t_beam, n, n]
            if self.perm_fix_first:
                model_logits = utils.mask_scores_pos_zero(model_logits)

            candidates_perm, candidates_log_probs = sample_distribution_beam_search(
                model_logits, self.PL_beam_size
            )
            # candidates_perm: [batch, t_beam, PL_beam, n]
            # candidates_log_probs: [batch, t_beam, PL_beam]
            candidates_perm = candidates_perm[
                ..., : self.t_beam_size, :
            ]  # [batch, t_beam, t_beam, n]
            candidates_log_probs = candidates_log_probs[
                ..., : self.t_beam_size
            ]  # [batch, t_beam, t_beam]

            candidates_perm = utils.permute_int_list(
                candidates_perm, result_perm.unsqueeze(-2)
            )  # [batch, t_beam, t_beam, n]
            candidates_log_probs = (
                result_log_probs.unsqueeze(-1) + candidates_log_probs
            )  # [batch, t_beam, t_beam]

            candidates_perm = candidates_perm.flatten(
                start_dim=-3, end_dim=-2
            )  # [batch, t_beam^2, n]
            candidates_log_probs = candidates_log_probs.flatten(start_dim=-2)  # [batch, t_beam^2]

            num_selected = min(self.t_beam_size, candidates_log_probs.size(-1))
            result_log_probs, topk_idx = torch.topk(
                candidates_log_probs, k=num_selected, dim=-1
            )  # [batch, t_beam]
            topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, n)
            result_perm = torch.gather(candidates_perm, -2, topk_idx_expanded)

        if self.perm_fix_first:  # TSP
            tsp_eval = utils.TSPEvaluator(
                input.unsqueeze(1).expand(-1, result_perm.size(-2), -1, -1)
            )
            tour_len = tsp_eval.evaluate(result_perm)  # [batch, t_beam]
            min_tour_len_idx = torch.argmin(tour_len, -1)  # [batch]
            min_tour_len_idx = min_tour_len_idx[..., None, None].expand(-1, -1, n)  # [batch, 1, n]
            result_perm = torch.gather(result_perm, -2, min_tour_len_idx).squeeze(-2)  # [batch, n]
        else:
            result_perm = result_perm[:, 0, ...]  # [batch, n]

        if input.dim() == 5:
            result_x = utils.permute_image_perm_list(result_perm, input)
        elif input.dim() == 2:
            result_x = utils.permute_int_list(result_perm, input)
        elif input.dim() == 3:
            result_x = utils.permute_embd(result_perm, input)
        else:
            raise NotImplementedError

        return result_x, result_perm

    # =============================================================================
    # Training losses
    # =============================================================================

    def training_loss_log_likelihood(self, x_start, reverse_model):
        """Training loss calculation.
        Compute E_{q(x_{0:T-1} | x_start)} [ - \sum_{t=0}^{T-1} log p_{theta}(x_{t-1} | x_t) ]

        Args:
            x_start: true data, shape [bs, n, c, h, w]
            reverse_model: function, reverse network
        """
        device = x_start.device
        n = x_start.size(1)

        identity_perm = torch.arange(n, device=device).expand(x_start.size(0), -1)
        perm_seq = self.q_sample_seq(identity_perm)  # shape [N, T+1, bs, n]
        perm_seq = perm_seq[:, self.reverse_steps, ...]
        perm_seq_no_start = perm_seq[:, 1:, ...]  # shape [N, T, bs, n]
        perm_seq_no_end = perm_seq[:, :-1, ...]  # shape [N, T, bs, n]

        t = torch.tensor(self.reverse_steps[1:], device=device).unsqueeze(-1)

        scores = self.p_logits(
            reverse_model, perm_seq_no_start, t, x_start
        )  # shape [N, T, bs, n, n]
        if self.perm_fix_first:
            scores = utils.mask_scores_pos_zero(scores)

        if self.reverse in ["PL", "generalized_PL"] and not self.latent:
            p_log_probs = self.p_log_cond_prob(
                scores, perm_tm1=perm_seq_no_end, perm_t=perm_seq_no_start
            )  # [N, T, bs]
        else:
            x_seq = utils.permute_image_perm_list(perm_seq, x_start)
            x_seq_no_start = x_seq[:, 1:, ...]  # shape [N, T, bs, n, c, h, w]
            x_seq_no_end = x_seq[:, :-1, ...]  # shape [N, T, bs, n, c, h, w]
            if self.latent:
                p_log_probs = self.p_log_cond_prob_latent(
                    scores, x_tm1=x_seq_no_end, x_t=x_seq_no_start
                )  # [N, T, bs]
            else:
                p_log_probs = self.p_log_cond_prob_images(
                    scores, x_tm1=x_seq_no_end, x_t=x_seq_no_start
                )  # [N, T, bs]

        loss = -p_log_probs.mean(-2)  # [N, bs]
        loss = loss.mean()

        return loss

    def training_loss_log_likelihood_randt(self, x_start, reverse_model):
        """
        Args:
            x_start: true data, shape [b, n, c, h, w]
            reverse_model: function, reverse network
        """
        assert self.transition == "riffle"

        device = x_start.device
        n = x_start.size(1)
        batch_size = x_start.size(0)

        timesteps = torch.tensor(self.reverse_steps, device=device).expand(batch_size, -1)
        rand_t_idx = torch.randint(
            low=1, high=timesteps.size(-1), size=(batch_size, 1), device=device
        )
        rand_t = torch.gather(timesteps, -1, rand_t_idx).squeeze(-1)  # [batch]
        rand_t_prev = torch.gather(timesteps, -1, rand_t_idx - 1).squeeze(-1)

        identity_perm = torch.arange(n, device=device).expand(self.sample_N, batch_size, -1)
        perm_start_prev, _ = RS.sample_riffle_shuffle(identity_perm, a=2**rand_t_prev)
        perm_start_now, perm_prev_now = RS.sample_riffle_shuffle(
            perm_start_prev, a=2 ** (rand_t - rand_t_prev)
        )

        scores = self.p_logits(reverse_model, perm_start_now, rand_t, x_start)  # shape [N, b, n, n]

        perm_now_prev = torch.argsort(perm_prev_now)

        if self.reverse == "generalized_PL":
            loss = -PL.log_prob_generalized_PL(scores, perm_now_prev)
        elif self.reverse == "PL":
            loss = -PL.log_prob(scores, perm_now_prev)
        else:
            raise NotImplementedError

        loss = loss.mean()

        return loss

    def training_losses(self, x_start, reverse_model):
        if self.loss == "log_likelihood":
            return self.training_loss_log_likelihood(x_start, reverse_model)

        elif self.loss == "log_likelihood_randt":
            return self.training_loss_log_likelihood_randt(x_start, reverse_model)

        else:
            raise NotImplementedError
