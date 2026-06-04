import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
import warnings
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches

from core.nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many
import copy
import random


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        # return super(AttentionModelFixed, self).__getitem__(key)
        return self[key]


class AttentionHCVRPActor(nn.Module):
    """Unified HCVRP attention actor; behavior selected by ``variant`` and ``n_paths``."""

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 obj,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 n_paths=None,
                 variant='base'):
        super(AttentionHCVRPActor, self).__init__()
        valid = ('base', 'explicit', 'implicit', 'decoder', 'multi')
        if variant not in valid:
            raise ValueError('variant must be one of {}, got {!r}'.format(valid, variant))
        self._variant = variant
        self._tw_in_precompute = variant in ('decoder', 'multi')

        # print("init hyperparam attention")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.obj = obj
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.is_hcvrp = problem.NAME == 'hcvrp'
        self.feed_forward_hidden = 512

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        if variant == 'multi':
            self.n_paths = int(n_paths if n_paths is not None else 5)
        else:
            self.n_paths = 1

        if not self.is_hcvrp:
            raise NotImplementedError('AttentionHCVRPActor is implemented for HCVRP only')

        # print("Check if problem hcvrp")
        # Problem specific context parameters (placeholder and step context dimension)
        step_context_dim = embedding_dim + 1
        num_veh = 3
        if variant == 'implicit':
            node_dim = 2 + num_veh
        else:
            node_dim = 2 + num_veh + 24
        node_veh = 9
        ff_fallback = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.FF_veh = nn.Sequential(
            nn.Linear(node_veh, self.embedding_dim),
            nn.Linear(self.embedding_dim, self.feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(self.feed_forward_hidden, self.embedding_dim)
        ) if self.feed_forward_hidden > 0 else ff_fallback

        self.FF_tour = nn.Sequential(
            nn.Linear(num_veh * self.embedding_dim, self.embedding_dim),
            nn.Linear(self.embedding_dim, self.feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(self.feed_forward_hidden, self.embedding_dim)
        ) if self.feed_forward_hidden > 0 else nn.Linear(self.embedding_dim, self.embedding_dim)
        self.select_embed = nn.Linear(self.embedding_dim * 2, num_veh)

        # Special embedding projection for depot node
        self.init_embed_depot = nn.Linear(2, embedding_dim)
        self.init_embed_ret = nn.Linear(2 * embedding_dim, embedding_dim)

        # print("init node embedding")
        self.init_embed = nn.Linear(node_dim, embedding_dim)  # node_embedding
        time_window_dim_out = 32
        if self._tw_in_precompute:
            self.init_embed_time_window = nn.Linear(24, time_window_dim_out)

        node_proj_in = embedding_dim + (time_window_dim_out if self._tw_in_precompute else 0)
        step_proj_in = step_context_dim + (time_window_dim_out if self._tw_in_precompute else 0)

        # print("embedder graph att")
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        assert embedding_dim % n_heads == 0
        if variant == 'multi':
            self.project_node_embeddings = nn.ModuleList([
                nn.Linear(node_proj_in, 3 * embedding_dim, bias=False) for _ in range(self.n_paths)])
            self.project_fixed_context = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim, bias=False) for _ in range(self.n_paths)])
            self.project_step_context = nn.ModuleList([
                nn.Linear(step_proj_in, embedding_dim, bias=False) for _ in range(self.n_paths)])
            self.project_out = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim, bias=False) for _ in range(self.n_paths)])
        else:
            # Plain Linear modules preserve legacy state_dict keys (not ModuleList.0.*)
            self.project_node_embeddings = nn.Linear(node_proj_in, 3 * embedding_dim, bias=False)
            self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.project_step_context = nn.Linear(step_proj_in, embedding_dim, bias=False)
            self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    @staticmethod
    def _projection(module, path_index):
        return module[path_index] if isinstance(module, nn.ModuleList) else module

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    @staticmethod
    def _pad_time_dim_tensors(tensors, pad_dim=1, fill_value=0.0):
        """Pad each tensor along time dim `pad_dim` to the max length in the list."""
        if not tensors:
            return tensors
        max_len = max(t.size(pad_dim) for t in tensors)
        out = []
        for t in tensors:
            cur = t.size(pad_dim)
            if cur < max_len:
                pad_shape = list(t.shape)
                pad_shape[pad_dim] = max_len - cur
                pad = torch.full(
                    pad_shape, fill_value, device=t.device, dtype=t.dtype
                )
                t = torch.cat([t, pad], dim=pad_dim)
            out.append(t)
        return out

    def _compute_first_step_path_logits(self, embeddings, input):
        """Per-decoder first-step log_p over customers (depot stripped) for KL."""
        states = [self.problem.make_state(input) for _ in range(self.n_paths)]
        outputs = []
        sequences = []
        tour_1, tour_2, tour_3 = [], [], []
        veh_list = []
        for i in range(self.n_paths):
            fixed = self._precompute(embeddings, input, path_index=i)
            veh, _ = self.select_veh(
                input, states[i], sequences, embeddings, self.obj,
                veh_list, tour_1, tour_2, tour_3
            )
            log_p, _ = self._get_log_p(fixed, states[i], veh=veh, path_index=i)
            if self.is_hcvrp:
                outputs.append(log_p[:, 0, 1:])
            else:
                outputs.append(log_p[:, 0, :])
            outputs[-1] = torch.maximum(
                outputs[-1],
                torch.full_like(outputs[-1], -1e9)
            )
        return outputs

    def _kl_between_paths(self, path_logits):
        """path_logits: list of [batch, n_nodes] log-probs (same support). Returns scalar KL penalty tensor."""
        if len(path_logits) < 2:
            return path_logits[0].new_zeros(())
        kl_terms = []
        for i in range(len(path_logits)):
            for j in range(len(path_logits)):
                if i == j:
                    continue
                kl_terms.append(
                    torch.sum(torch.exp(path_logits[i]) * (path_logits[i] - path_logits[j]), dim=-1).mean()
                )
        return torch.stack(kl_terms, 0).mean()

    def forward(self, input, opts=None, bi_val=None, baseline=None, return_pi=False, return_kl=False, **kwargs):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        if self.checkpoint_encoder:
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            embeddings, _ = self.embedder(self._init_embed(input))

        if self._variant == 'multi':
            kl_coef = float(getattr(opts, 'kl_loss', 0) or 0) if opts is not None else 0.0
            kl_mean = embeddings.new_zeros(())
            if self.n_paths > 1 and kl_coef > 0:
                path_logits = self._compute_first_step_path_logits(embeddings, input)
                kl_mean = self._kl_between_paths(path_logits)

            if self.training and self.n_paths > 1:
                costs_p, ll_p, llv_p, pi = self._decode_all_paths_training(input, embeddings)
                cost_min = costs_p.min(dim=0).values
                best_idx = costs_p.argmin(dim=0)
                b_idx = torch.arange(costs_p.size(1), device=costs_p.device)
                ll_b = ll_p[best_idx, b_idx]
                llv_b = llv_p[best_idx, b_idx]
                core = torch.stack((costs_p, ll_p, llv_p), dim=0)
                kl_plane = kl_mean.view(1, 1, 1).expand(1, costs_p.size(0), costs_p.size(1))
                pack = torch.cat((core, kl_plane), dim=0)
                if return_pi:
                    return cost_min, ll_b, llv_b, pi
                if return_kl:
                    return cost_min, ll_b, llv_b, -kl_coef * kl_mean
                return cost_min, ll_b, llv_b, pack

            _log_p, log_p_veh, pi, veh_list, t1, t2, t3 = self._inner_eval(input, embeddings)
            cost, mask = self.problem.get_costs(input, self.obj, pi, veh_list, t1, t2, t3)
            ll, ll_veh = self._calc_log_likelihood(_log_p, log_p_veh, pi, mask, veh_list)
            if return_pi:
                return cost, ll, ll_veh, pi
            if return_kl:
                return cost, ll, ll_veh, -kl_coef * kl_mean
            return cost, ll, ll_veh

        _log_p, log_p_veh, pi, veh_list, t1, t2, t3 = self._inner(input, embeddings)
        cost, mask = self.problem.get_costs(input, self.obj, pi, veh_list, t1, t2, t3)
        ll, ll_veh = self._calc_log_likelihood(_log_p, log_p_veh, pi, mask, veh_list)
        if return_pi:
            return cost, ll, ll_veh, pi

        return cost, ll, ll_veh

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        tw_in = self._tw_in_precompute
        return CachedLookup(
            self._precompute(embeddings, input if tw_in else None, path_index=0))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) / ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, _log_p_veh, a, mask, veh_list):  # a is pi
        log_p = _log_p.gather(2, a.long().unsqueeze(-1)).squeeze(-1)
        log_p_veh = _log_p_veh.gather(2, veh_list.long().unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0
            log_p_veh[mask] = 0
        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"
        assert (log_p_veh > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1), log_p_veh.sum(1)  # [batch_size]

    def _init_embed(self, input):
        if self.is_hcvrp:
            dev = input['loc'].device
            demand = torch.tensor(
                [(input['demand'] / input['capacity'][0:1, veh]).tolist() for veh in
                 range(input['capacity'].size(-1))],
                device=dev,
                dtype=input['demand'].dtype,
            ).transpose(0, 1).transpose(1, 2)
            if self._variant == 'implicit':
                cust = torch.cat((input['loc'], demand), -1)
            else:
                time_window = input['time_window']
                cust = torch.cat((input['loc'], demand, time_window), -1)
            return torch.cat(
                (
                    self.init_embed_depot(input['depot'])[:, None, :],
                    self.init_embed(cust),
                ),
                1
            )

    def select_veh(self, input, state, sequences, embeddings, obj, veh_list, tour_1, tour_2, tour_3):
        current_node = state.get_current_node()  # [batch_size]
        tour_dis = state.lengths  # [batch_size, num_veh]
        if self._variant == 'explicit':
            SPEED = [1, 1, 1]
            if obj == 'min-sum':
                SPEED = [1/4, 1/5, 1/6]
        else:
            if obj == 'min-max':
                SPEED = [1, 1, 1]
            if obj == 'min-sum':
                SPEED = [1/4, 1/5, 1/6]

        batch_size, _, embed_dim = embeddings.size()
        _, num_veh = current_node.size()

        if sequences:
            tour_1 = torch.stack(tour_1, -1)  # [batch_size, tour_len]
            tour_2 = torch.stack(tour_2, -1)
            tour_3 = torch.stack(tour_3, -1)

            tour_con_1 = torch.gather(
                embeddings,  # [batch_size, graph_size, embed_dim]
                1,
                (tour_1.clone())[..., None].contiguous()  # [batch_size, tour_len]
                    .expand(batch_size, tour_1.size(-1), embed_dim)
            ).view(batch_size, tour_1.size(-1), embed_dim)  # [batch_size, tour_len, embed_dim]
            tour_con_2 = torch.gather(
                embeddings,  # [batch_size, graph_size, embed_dim]
                1,
                (tour_2.clone())[..., None].contiguous()  # [batch_size, tour_len]
                    .expand(batch_size, tour_2.size(-1), embed_dim)
            ).view(batch_size, tour_2.size(-1), embed_dim)
            tour_con_3 = torch.gather(  # [batch_size, tour_len, embed_dim]
                embeddings,  # [batch_size, graph_size, embed_dim]
                1,
                (tour_3.clone())[..., None].contiguous()  # [batch_size, tour_len]
                    .expand(batch_size, tour_3.size(-1), embed_dim)
            ).view(batch_size, tour_3.size(-1), embed_dim)

            mean_tour = torch.cat(  # [batch_size, 3*embed_dim]
                (
                    torch.max(tour_con_1, dim=1)[0],
                    torch.max(tour_con_2, dim=1)[0],
                    torch.max(tour_con_3, dim=1)[0]
                ),
                1,
            )  # [batch_size, embed_dim]

            current_loc = state.coords.gather(  # [batch_size, graph_size, 2]
                1,
                (current_node.clone())[..., None].contiguous()
                    .expand_as(state.coords[:, 0:num_veh, :])
            ).transpose(0, 1)  # [num_veh, batch_size, 2]

            veh_context = torch.cat(
                (
                    tour_dis[:, 0].unsqueeze(-1) / SPEED[0],
                    current_loc[0, :],  # [batch_size, 2])
                    tour_dis[:, 1].unsqueeze(-1) / SPEED[1],
                    current_loc[1, :],
                    tour_dis[:, 2].unsqueeze(-1) / SPEED[2],
                    current_loc[2, :]
                ),
                -1
            )
        else:
            current_loc = state.coords.gather(  # [batch_size, graph_size, 2]
                1,
                (current_node.clone())[..., None].contiguous()
                    .expand_as(state.coords[:, 0:num_veh, :])
            ).transpose(0, 1)  # [batch_size, 2]

            mean_tour = torch.zeros(
                batch_size, 3 * embed_dim, device=embeddings.device, dtype=embeddings.dtype
            )
            veh_context = torch.cat(  # [batch_size, num_veh, 5]
                (
                    tour_dis[:, 0].unsqueeze(-1) / SPEED[0],  # [batch_size, num_veh]
                    current_loc[0, :],  # [batch_size, 2])
                    tour_dis[:, 1].unsqueeze(-1) / SPEED[1],
                    current_loc[1, :],
                    tour_dis[:, 2].unsqueeze(-1) / SPEED[2],
                    current_loc[2, :]
                ),
                -1
            )

        veh_context = self.FF_veh(veh_context)
        tour_context = self.FF_tour(mean_tour)
        context = torch.cat((veh_context, tour_context), -1).view(batch_size, self.embedding_dim * 2)

        log_veh = F.log_softmax(self.select_embed(context), dim=1)
        # print(self.select_embed(context).shape)
        # print(log_veh)
        # raise ValueError
        if self.decode_type == "greedy":
            veh = torch.max(F.softmax(self.select_embed(context), dim=1), dim=1)[1]
        elif self.decode_type == "sampling":
            veh = F.softmax(self.select_embed(context), dim=1).multinomial(1).squeeze(-1)

        return veh, log_veh


    def _decode_all_paths_training(self, input, embeddings):
        """Decode each path; return stacked costs / log-likelihoods [n_paths, batch] and last path pi."""
        state = [self.problem.make_state(input) for _ in range(self.n_paths)]
        cost_rows, ll_rows, llv_rows = [], [], []
        last_pi = None
        for i in range(self.n_paths):
            current_node = state[i].get_current_node()
            batch_size, num_veh = current_node.size()

            outputs = []
            outputs_veh = []
            sequences = []
            tour_1 = []
            tour_2 = []
            tour_3 = []

            fixed = self._precompute(embeddings, input, path_index=i)

            veh_steps = []
            while not (self.shrink_size is None and state[i].all_finished()):
                veh, log_p_veh = self.select_veh(
                    input, state[i], sequences, embeddings, self.obj,
                    veh_steps, tour_1, tour_2, tour_3
                )
                veh_steps.append(veh)
                if self.shrink_size is not None:
                    unfinished = torch.nonzero(state[i].get_finished() == 0)
                    if len(unfinished) == 0:
                        break
                    unfinished = unfinished[:, 0]
                    if 16 <= len(unfinished) <= state[i].ids.size(0) - self.shrink_size:
                        state[i] = state[i][unfinished]
                        fixed = fixed[unfinished]
                log_p, mask = self._get_log_p(fixed, state[i], veh=veh, path_index=i)

                selected = self._select_node(
                    log_p.exp()[:, 0, :], mask[:, 0, :], state, veh, sequences, path_index=i
                )

                state[i] = state[i].update(selected, veh)

                if self.shrink_size is not None and state[i].ids.size(0) < batch_size:
                    log_p_, selected_ = log_p, selected
                    log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                    selected = selected_.new_zeros(batch_size)

                    log_p[state[i].ids[:, 0]] = log_p_
                    selected[state[i].ids[:, 0]] = selected_

                outputs.append(log_p[:, 0, :])
                outputs_veh.append(log_p_veh)

                sequences.append(selected[torch.arange(batch_size, device=selected.device), veh])
                tour_1.append(selected[:, 0])
                tour_2.append(selected[:, 1])
                tour_3.append(selected[:, 2])

            veh_list = torch.stack(veh_steps, dim=1)
            _log_p = torch.stack(outputs, 1)
            log_p_veh = torch.stack(outputs_veh, 1)
            pi = torch.stack(sequences, -1)
            tour_1 = torch.stack(tour_1, -1)
            tour_2 = torch.stack(tour_2, -1)
            tour_3 = torch.stack(tour_3, -1)

            cost, mask = self.problem.get_costs(
                input, self.obj, pi, veh_list, tour_1, tour_2, tour_3
            )
            ll, ll_veh = self._calc_log_likelihood(_log_p, log_p_veh, pi, mask, veh_list)
            cost_rows.append(cost)
            ll_rows.append(ll)
            llv_rows.append(ll_veh)
            last_pi = pi

        costs_p = torch.stack(cost_rows, dim=0)
        ll_p = torch.stack(ll_rows, dim=0)
        llv_p = torch.stack(llv_rows, dim=0)
        return costs_p, ll_p, llv_p, last_pi

    def _inner_eval(self, input, embeddings, bl_val=None, baseline=None):
        """Greedy / sampling eval: decode all paths, pick best cost per batch row. Returns 7-tuple for sample_many."""
        state = [self.problem.make_state(input) for _ in range(self.n_paths)]
        costs = []
        list_log_p = []
        list_log_p_veh = []
        list_pi = []
        list_vehicle = []
        list_tour_1 = []
        list_tour_2 = []
        list_tour_3 = []
        for i in range(self.n_paths):
            current_node = state[i].get_current_node()
            batch_size, num_veh = current_node.size()

            outputs = []
            outputs_veh = []
            sequences = []
            tour_1 = []
            tour_2 = []
            tour_3 = []

            fixed = self._precompute(embeddings, input, path_index=i)

            veh_hist = []
            while not (self.shrink_size is None and state[i].all_finished()):
                veh, log_p_veh = self.select_veh(
                    input, state[i], sequences, embeddings, self.obj,
                    veh_hist, tour_1, tour_2, tour_3
                )
                veh_hist.append(veh)
                if self.shrink_size is not None:
                    unfinished = torch.nonzero(state[i].get_finished() == 0)
                    if len(unfinished) == 0:
                        break
                    unfinished = unfinished[:, 0]
                    if 16 <= len(unfinished) <= state[i].ids.size(0) - self.shrink_size:
                        state[i] = state[i][unfinished]
                        fixed = fixed[unfinished]
                log_p, mask = self._get_log_p(fixed, state[i], veh=veh, path_index=i)

                selected = self._select_node(
                    log_p.exp()[:, 0, :], mask[:, 0, :], state, veh, sequences, path_index=i
                )

                state[i] = state[i].update(selected, veh)

                if self.shrink_size is not None and state[i].ids.size(0) < batch_size:
                    log_p_, selected_ = log_p, selected
                    log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                    selected = selected_.new_zeros(batch_size)

                    log_p[state[i].ids[:, 0]] = log_p_
                    selected[state[i].ids[:, 0]] = selected_

                outputs.append(log_p[:, 0, :])
                outputs_veh.append(log_p_veh)

                sequences.append(selected[torch.arange(batch_size, device=selected.device), veh])
                tour_1.append(selected[:, 0])
                tour_2.append(selected[:, 1])
                tour_3.append(selected[:, 2])

            veh_list = torch.stack(veh_hist, dim=1)
            _log_p = torch.stack(outputs, 1)
            log_p_veh = torch.stack(outputs_veh, 1)
            pi = torch.stack(sequences, -1)
            tour_1 = torch.stack(tour_1, -1)
            tour_2 = torch.stack(tour_2, -1)
            tour_3 = torch.stack(tour_3, -1)

            cost, mask = self.problem.get_costs(
                input, self.obj, pi, veh_list, tour_1, tour_2, tour_3
            )
            costs.append(cost.detach())
            list_vehicle.append(veh_list)
            list_log_p.append(_log_p)
            list_log_p_veh.append(log_p_veh)
            list_pi.append(pi)
            list_tour_1.append(tour_1)
            list_tour_2.append(tour_2)
            list_tour_3.append(tour_3)

        costs_t = torch.stack(costs, dim=0)
        best_idx = costs_t.argmin(dim=0)
        batch_idx = torch.arange(best_idx.size(0), device=best_idx.device)

        list_log_p = self._pad_time_dim_tensors(list_log_p, pad_dim=1, fill_value=0.0)
        list_log_p_veh = self._pad_time_dim_tensors(list_log_p_veh, pad_dim=1, fill_value=0.0)
        list_pi = self._pad_time_dim_tensors(list_pi, pad_dim=1, fill_value=0)
        list_vehicle = self._pad_time_dim_tensors(list_vehicle, pad_dim=1, fill_value=0)
        list_tour_1 = self._pad_time_dim_tensors(list_tour_1, pad_dim=1, fill_value=0)
        list_tour_2 = self._pad_time_dim_tensors(list_tour_2, pad_dim=1, fill_value=0)
        list_tour_3 = self._pad_time_dim_tensors(list_tour_3, pad_dim=1, fill_value=0)

        log_p = torch.stack(list_log_p, dim=0)
        log_p_veh = torch.stack(list_log_p_veh, dim=0)
        pi = torch.stack(list_pi, dim=0)
        veh_list = torch.stack(list_vehicle, dim=0)
        tour_1 = torch.stack(list_tour_1, dim=0)
        tour_2 = torch.stack(list_tour_2, dim=0)
        tour_3 = torch.stack(list_tour_3, dim=0)

        best_log_p = log_p[best_idx, batch_idx]
        best_log_p_veh = log_p_veh[best_idx, batch_idx]
        best_pi = pi[best_idx, batch_idx]
        best_veh_list = veh_list[best_idx, batch_idx]
        best_tour_1 = tour_1[best_idx, batch_idx]
        best_tour_2 = tour_2[best_idx, batch_idx]
        best_tour_3 = tour_3[best_idx, batch_idx]

        return best_log_p, best_log_p_veh, best_pi, best_veh_list, best_tour_1, best_tour_2, best_tour_3

    def _inner(self, input, embeddings):
        state = self.problem.make_state(input)
        current_node = state.get_current_node()
        batch_size, num_veh = current_node.size()

        outputs = []
        outputs_veh = []
        sequences = []
        tour_1 = []
        tour_2 = []
        tour_3 = []

        pre_inp = input if self._tw_in_precompute else None
        fixed = self._precompute(embeddings, pre_inp, path_index=0)

        veh_list = []
        while not (self.shrink_size is None and state.all_finished()):
            veh, log_p_veh = self.select_veh(
                input, state, sequences, embeddings, self.obj, veh_list, tour_1, tour_2, tour_3)
            veh_list.append(veh.tolist())
            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    state = state[unfinished]
                    fixed = fixed[unfinished]
            log_p, mask = self._get_log_p(fixed, state, veh=veh, path_index=0)

            selected = self._select_node(
                log_p.exp()[:, 0, :], mask[:, 0, :], state, veh, sequences, path_index=0)

            state = state.update(selected, veh)

            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            outputs.append(log_p[:, 0, :])
            outputs_veh.append(log_p_veh)

            sequences.append(selected[torch.arange(batch_size), veh])
            tour_1.append(selected[:, 0])
            tour_2.append(selected[:, 1])
            tour_3.append(selected[:, 2])

        veh_list = torch.tensor(
            veh_list, device=embeddings.device, dtype=torch.long
        ).transpose(0, 1)
        return (torch.stack(outputs, 1), torch.stack(outputs_veh, 1),
                torch.stack(sequences, -1), veh_list,
                torch.stack(tour_1, -1), torch.stack(tour_2, -1), torch.stack(tour_3, -1))

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        # print('input', input)

        inner_fn = self._inner_eval if self._variant == 'multi' else self._inner
        return sample_many(
            lambda input: inner_fn(*input),
            lambda input, pi, veh_list, tour_1, tour_2, tour_3: self.problem.get_costs(input[0], self.obj, pi, veh_list, tour_1, tour_2, tour_3),  # Don't need embeddings as input to get_costs
            (input, self.embedder(self._init_embed(input))[0]),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask, state, veh, sequences, path_index=0):
        assert (probs == probs).all(), "Probs should not contain any nans"

        st = state[path_index] if isinstance(state, list) else state
        selected = st.get_current_node().clone()
        batch_size, _ = st.get_current_node().size()

        if self.decode_type == "greedy":
            _, selected[torch.arange(batch_size), veh] = probs.max(1)
            assert not mask.gather(-1,selected[torch.arange(batch_size), veh].unsqueeze(-1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected[torch.arange(batch_size), veh] = probs.multinomial(1).squeeze(
                1)  # [batch_size]

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(-1, selected[torch.arange(batch_size), veh].unsqueeze(-1)).data.any():
                warnings.warn('Sampled infeasible action; resampling.', UserWarning)
                selected[torch.arange(batch_size), veh] = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, input=None, num_steps=1, path_index=0):
        graph_embed = embeddings.mean(1)
        p_fix = self._projection(self.project_fixed_context, path_index)
        fixed_context = p_fix(graph_embed)[:, None, :]

        if self._tw_in_precompute:
            time_windows = input['time_window']
            time_windows = torch.cat((
                torch.ones(
                    (time_windows.size(0), 1, time_windows.size(-1)),
                    device=time_windows.device),
                time_windows), dim=1)
            time_windows = self.init_embed_time_window(time_windows)
            emb = torch.cat((embeddings, time_windows), dim=-1)
        else:
            emb = embeddings

        p_node = self._projection(self.project_node_embeddings, path_index)
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            p_node(emb[:, None, :, :]).chunk(3, dim=-1)

        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(emb, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True, path_index=0):
        bsz = state.get_current_node().size(0)
        dev = state.get_current_node().device
        veh = torch.zeros(bsz, dtype=torch.long, device=dev)
        log_p, _ = self._get_log_p(fixed, state, veh=veh, path_index=path_index, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, veh, path_index=0, normalize=True):
        p_step = self._projection(self.project_step_context, path_index)
        query = fixed.context_node_projected + \
                p_step(self._get_parallel_step_context(fixed.node_embeddings, state, veh))

        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        mask = state.get_mask(veh)

        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, veh, path_index)

        if normalize:
            log_p = F.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, veh, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)

        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = (state.get_current_node()).clone()
        batch_size, num_veh = current_node.size()
        num_steps = 1

        if self.is_hcvrp:
            # Embedding of previous node + remaining capacity
            if from_depot:
                # 1st dimension is node idx, but we do not squeeze it since we want to insert step dimension
                # i.e. we actually want embeddings[:, 0, :][:, None, :] which is equivalent
                return torch.cat(  # [batch_size, num_veh, embed_dim+1]
                    (
                        embeddings[:, 0:1, :].expand(batch_size, num_veh, embeddings.size(-1)),
                        # used capacity is 0 after visiting depot
                        torch.tensor(
                            self.problem.VEHICLE_CAPACITY,
                            device=state.used_capacity.device,
                            dtype=state.used_capacity.dtype,
                        )[None, :, None] - torch.zeros_like(state.used_capacity[:, :, None])
                    ),
                    -1
                )
            else:
                # Check if the tensor for vehicle capacity is on the GPU
                # print(torch.tensor(self.problem.VEHICLE_CAPACITY).cuda().device)

                # Check if the tensor for used capacity is on the GPU
                # print(state.used_capacity.device)

                # Check if the tensor for the range is on the GPU
                # print(torch.arange(batch_size).device)
                return torch.cat(  # [batch_size, num_veh, embed_dim+1]
                    (
                        torch.gather(
                            embeddings,  # [batch_size, graph_size, embed_dim]
                            1,
                            (current_node[torch.arange(batch_size), veh]).contiguous()
                                .view(batch_size, num_steps, 1)
                                .expand(batch_size, num_steps, embeddings.size(-1))
                        ).view(batch_size, num_steps, embeddings.size(-1)),  # [batch_size, num_step, embed_dim]
                        (torch.tensor(
                            self.problem.VEHICLE_CAPACITY,
                            device=state.used_capacity.device,
                            dtype=state.used_capacity.dtype,
                        )[None, veh] - state.used_capacity[torch.arange(batch_size, device=veh.device), veh]
                         ).transpose(0, 1).unsqueeze(-1)
                    ),
                    -1
                )

        if self.is_pdvrp:
            # Embedding of previous node + remaining capacity
            if from_depot:
                # 1st dimension is node idx, but we do not squeeze it since we want to insert step dimension
                # i.e. we actually want embeddings[:, 0, :][:, None, :] which is equivalent
                return torch.cat(  # [batch_size, num_steps, 2*embed_dim] step_contex_dim
                    (
                        embeddings[:, 0:1, :].expand(batch_size, num_steps, embeddings.size(-1)),
                        torch.gather(
                            embeddings,
                            1,
                            current_node.contiguous().view(batch_size, num_steps, 1)
                                .expand(batch_size, num_steps, embeddings.size(-1))
                            # [batch_size, num_steps, embed_dim]
                        ).view(batch_size, num_steps, embeddings.size(-1))
                    ),
                    -1
                )
            else:
                return torch.gather(
                    embeddings,
                    1,
                    current_node.contiguous()
                        .view(batch_size, num_steps, 1)
                        .expand(batch_size, num_steps, embeddings.size(-1))
                ).view(batch_size, num_steps, embeddings.size(-1))

        if self.is_vrp:
            # Embedding of previous node + remaining capacity
            if from_depot:
                # 1st dimension is node idx, but we do not squeeze it since we want to insert step dimension
                # i.e. we actually want embeddings[:, 0, :][:, None, :] which is equivalent
                return torch.cat(
                    (
                        embeddings[:, 0:1, :].expand(batch_size, num_steps, embeddings.size(-1)),
                        # used capacity is 0 after visiting depot
                        self.problem.VEHICLE_CAPACITY - torch.zeros_like(state.used_capacity[:, :, None])
                    ),
                    -1
                )
            else:
                return torch.cat(
                    (
                        torch.gather(
                            embeddings,  # [batch_size, graph_size, embed_dim]
                            1,
                            current_node.contiguous()
                                .view(batch_size, num_steps, 1)
                                .expand(batch_size, num_steps, embeddings.size(-1))
                        ).view(batch_size, num_steps, embeddings.size(-1)),  # [batch_size, num_steps, embed_dim]
                        self.problem.VEHICLE_CAPACITY - state.used_capacity[:, :, None]
                    ),
                    -1
                )

        elif self.is_orienteering or self.is_pctsp:
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous()
                            .view(batch_size, num_steps, 1)
                            .expand(batch_size, num_steps, embeddings.size(-1))
                    ).view(batch_size, num_steps, embeddings.size(-1)),
                    (
                        state.get_remaining_length()[:, :, None]
                        if self.is_orienteering
                        else state.get_remaining_prize_to_collect()[:, :, None]
                    )
                ),
                -1
            )

        else:  # TSP

            if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
                if state.i.item() == 0:
                    # First and only step, ignore prev_a (this is a placeholder)
                    return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
                else:
                    return embeddings.gather(
                        1,
                        torch.cat((state.first_a, current_node), 1)[:, :, None].expand(batch_size, 2,
                                                                                       embeddings.size(-1))
                    ).view(batch_size, 1, -1)
            # More than one step, assume always starting with first
            embeddings_per_step = embeddings.gather(
                1,
                current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
            )
            return torch.cat((
                # First step placeholder, cat in dim 1 (time steps)
                self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
                # Second step, concatenate embedding of first with embedding of current/previous (in dim 2, context dim)
                torch.cat((
                    embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                    embeddings_per_step
                ), 2)
            ), 1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, veh, path_index):
        batch_size, num_step, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads  # query and K both have key_size

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_step, 1, key_size)
        glimpse_Q = query.view(batch_size, num_step, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_step, 1, graph_size)
        # glimpse_K (n_heads, batch_size, 1, graph_size, key_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

        if self.mask_inner:  # True
            assert self.mask_logits, "Cannot mask inner without masking logits"  # True
            # mask: # [batch_size, num_veh, graph_size]
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf  # nask visited nodes and nodes cannot be visited

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_step, 1, val_size)
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_step, 1, embedding_dim)
        p_out = self._projection(self.project_out, path_index)
        glimpse = p_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_step, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        final_Q = glimpse
        # logits_K, (batch_size, 1, graph_size, embed_dim)
        # Batch matrix multiplication to compute logits (batch_size, num_step, graph_size)
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:  # 10
            # print*(F.tanh(logits))
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:  # True
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)  # glimpse[batch_size, num_veh, embed_dim]

    def _get_attention_node_data(self, fixed, state):

        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=1):  # v: [batch_size, 1, graph_size+1, embed_dim]
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
                .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
                .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, embed_dim)
        )


class AttentionModel(AttentionHCVRPActor):
    def __init__(self, embedding_dim, hidden_dim, obj, problem, n_encode_layers=2,
                 tanh_clipping=10., mask_inner=True, mask_logits=True, normalization='batch',
                 n_heads=8, checkpoint_encoder=False, shrink_size=None, n_paths=None):
        super(AttentionModel, self).__init__(
            embedding_dim, hidden_dim, obj, problem,
            n_encode_layers=n_encode_layers,
            tanh_clipping=tanh_clipping,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            normalization=normalization,
            n_heads=n_heads,
            checkpoint_encoder=checkpoint_encoder,
            shrink_size=shrink_size,
            n_paths=1,
            variant='base')


class AttentionModelExplicit(AttentionHCVRPActor):
    def __init__(self, embedding_dim, hidden_dim, obj, problem, n_encode_layers=2,
                 tanh_clipping=10., mask_inner=True, mask_logits=True, normalization='batch',
                 n_heads=8, checkpoint_encoder=False, shrink_size=None, n_paths=None):
        super(AttentionModelExplicit, self).__init__(
            embedding_dim, hidden_dim, obj, problem,
            n_encode_layers=n_encode_layers,
            tanh_clipping=tanh_clipping,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            normalization=normalization,
            n_heads=n_heads,
            checkpoint_encoder=checkpoint_encoder,
            shrink_size=shrink_size,
            n_paths=1,
            variant='explicit')


class AttentionModelImplicit(AttentionHCVRPActor):
    def __init__(self, embedding_dim, hidden_dim, obj, problem, n_encode_layers=2,
                 tanh_clipping=10., mask_inner=True, mask_logits=True, normalization='batch',
                 n_heads=8, checkpoint_encoder=False, shrink_size=None, n_paths=None):
        super(AttentionModelImplicit, self).__init__(
            embedding_dim, hidden_dim, obj, problem,
            n_encode_layers=n_encode_layers,
            tanh_clipping=tanh_clipping,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            normalization=normalization,
            n_heads=n_heads,
            checkpoint_encoder=checkpoint_encoder,
            shrink_size=shrink_size,
            n_paths=1,
            variant='implicit')


class AttentionModelDecoder(AttentionHCVRPActor):
    def __init__(self, embedding_dim, hidden_dim, obj, problem, n_encode_layers=2,
                 tanh_clipping=10., mask_inner=True, mask_logits=True, normalization='batch',
                 n_heads=8, checkpoint_encoder=False, shrink_size=None, n_paths=None):
        super(AttentionModelDecoder, self).__init__(
            embedding_dim, hidden_dim, obj, problem,
            n_encode_layers=n_encode_layers,
            tanh_clipping=tanh_clipping,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            normalization=normalization,
            n_heads=n_heads,
            checkpoint_encoder=checkpoint_encoder,
            shrink_size=shrink_size,
            n_paths=1,
            variant='decoder')


class AttentionModelMultiDecoder(AttentionHCVRPActor):
    def __init__(self, embedding_dim, hidden_dim, obj, problem, n_encode_layers=2,
                 tanh_clipping=10., mask_inner=True, mask_logits=True, normalization='batch',
                 n_heads=8, checkpoint_encoder=False, shrink_size=None, n_paths=None):
        super(AttentionModelMultiDecoder, self).__init__(
            embedding_dim, hidden_dim, obj, problem,
            n_encode_layers=n_encode_layers,
            tanh_clipping=tanh_clipping,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            normalization=normalization,
            n_heads=n_heads,
            checkpoint_encoder=checkpoint_encoder,
            shrink_size=shrink_size,
            n_paths=n_paths if n_paths is not None else 5,
            variant='multi')
