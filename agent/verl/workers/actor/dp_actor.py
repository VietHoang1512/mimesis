# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.activation_offload import suspend_activation_offload
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

SDPO_TOPK = int(os.environ.get("SDPO_TOPK", "20"))
SDPO_NORM = os.environ.get("SDPO_NORM", "masked")
SDAR_BETA = float(os.environ.get("SDPO_GATE_BETA", "5.0"))

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if float(self.config.get("sdpo_coef", 0.0)) != 0.0:
            def _dropouts(cfg, prefix=""):
                items = vars(cfg).items() if hasattr(cfg, "__dict__") else []
                for k, v in items:
                    if hasattr(v, "__dict__") and k.endswith("_config"):
                        yield from _dropouts(v, f"{prefix}{k}.")
                    elif "dropout" in k and isinstance(v, (int, float)) and v:
                        yield f"{prefix}{k}", v
            hf_cfg = getattr(actor_module, "config", None)
            live = dict(_dropouts(hf_cfg)) if hf_cfg is not None else {}
            if live:
                raise ValueError(
                    f"sdpo_coef != 0 requires dropout disabled, found {live}. "
                    "Two forwards of the same weights would draw different masks "
                    "and the SDPO KL would measure noise.")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        self._sdpo_student_topk = None
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    self._sdpo_student_topk = None
                    _tk = None
                    if "sdpo_topk_idx" in micro_batch.keys():
                        _tk = micro_batch["sdpo_topk_idx"]
                    if _tk is not None:
                        assert not self.use_ulysses_sp, (
                            "SDPO distill does not support ulysses sp>1: logits_rmpad "
                            "is sharded and the gather would need the same "
                            "gather_outpus_and_unpad treatment as log_probs.")
                        kk = _tk.size(-1)
                        _full = torch.zeros((batch_size, seqlen, kk),
                                            dtype=torch.long, device=_tk.device)
                        _s = seqlen - response_length - 1
                        _full[:, _s:_s + response_length, :] = _tk.long()
                        _rm = _full.view(-1, kk)[indices]
                        self._sdpo_student_topk = (
                            torch.gather(logits_rmpad, -1, _rm)
                            - torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                        ).float()
                        del _full, _rm

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy or self._sdpo_student_topk is not None:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if self._sdpo_student_topk is not None:
                    self._sdpo_student_topk = pad_input(
                        hidden_states=self._sdpo_student_topk,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )[:, -response_length - 1 : -1, :]

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _sdpo_teacher_logprobs(self, micro_batch, temperature) -> torch.Tensor:
        ids = micro_batch["sdpo_teacher_input_ids"].long()
        attn = micro_batch["sdpo_teacher_attention_mask"]
        index_map = micro_batch["sdpo_index_map"].long()
        row = micro_batch["sdpo_row"].long()
        sdpo_mask_bool = micro_batch["sdpo_mask"].bool()
        want_topk = self.config.get("sdpo_estimator", "ref") == "distill"
        bsz, K, W = ids.shape
        flat_ids, flat_attn = ids.reshape(bsz * K, W), attn.reshape(bsz * K, W)

        _pm_all = None
        if "sdpo_placebo_map" in micro_batch.keys():
            _pm_all = micro_batch["sdpo_placebo_map"].long()
        _need_src = index_map if _pm_all is None else torch.minimum(index_map, torch.where(_pm_all > 0, _pm_all, index_map))
        need = _need_src[sdpo_mask_bool] if sdpo_mask_bool.any() else None
        first = int(need.min()) - 1 if need is not None and need.numel() else W - 2
        first = max(min(first, W - 2), 0)
        keep = W - first
        off = W - keep

        was_training = self.actor_module.training
        _CH = max(1, int(os.environ.get("SDPO_TEACHER_CHUNK", "4")))
        _nrows = flat_ids.size(0)
        _nch = (_nrows + _CH - 1) // _CH
        if torch.distributed.is_initialized():
            _t = torch.tensor([_nch], device=flat_ids.device)
            torch.distributed.all_reduce(_t, op=torch.distributed.ReduceOp.MAX)
            _nch = int(_t.item())
        _pieces, _tv_p, _ti_p = [], [], []
        with suspend_activation_offload():
            for _ci in range(_nch):
                _c0 = _ci * _CH
                _real = _c0 < _nrows
                if not _real:
                    _c0 = max(0, _nrows - _CH)
                _ids_c = flat_ids[_c0:_c0 + _CH]
                _att_c = flat_attn[_c0:_c0 + _CH]
                position_ids = torch.clamp(torch.cumsum(_att_c, dim=1) - 1, min=0)
                with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                    try:
                        logits = self.actor_module(
                            input_ids=_ids_c, attention_mask=_att_c,
                            position_ids=position_ids, use_cache=False,
                            logits_to_keep=keep,
                        ).logits
                    except TypeError:
                        try:
                            logits = self.actor_module(
                                input_ids=_ids_c, attention_mask=_att_c,
                                position_ids=position_ids, use_cache=False,
                                num_logits_to_keep=keep,
                            ).logits
                        except TypeError:
                            logits = self.actor_module(
                                input_ids=_ids_c, attention_mask=_att_c,
                                position_ids=position_ids, use_cache=False).logits
                            logits = logits[:, -keep:, :]
                    logits = logits.div_(temperature)
                    _lp_c = logprobs_from_logits(
                        logits[:, :-1, :], _ids_c[:, off + 1:]).float()
                    if _real:
                        _pieces.append(_lp_c)
                    if want_topk and _real:
                        _lg = logits[:, :-1, :]
                        _tv, _tix = torch.topk(_lg, min(SDPO_TOPK, _lg.size(-1)), dim=-1)
                        _tv_p.append((_tv - torch.logsumexp(_lg, dim=-1, keepdim=True)).float())
                        _ti_p.append(_tix)
                        del _lg, _tv, _tix
                del logits
        shifted = torch.cat(_pieces, dim=0)
        if want_topk:
            _tlp = torch.cat(_tv_p, dim=0)
            _ti = torch.cat(_ti_p, dim=0)
        del _pieces, _tv_p, _ti_p
        shifted = shifted.view(bsz, K, keep - 1)

        assert self.actor_module.training == was_training

        at = torch.clamp(index_map - off - 1, min=0, max=keep - 2)
        b = torch.arange(bsz, device=ids.device).unsqueeze(1).expand_as(at)
        rc = torch.clamp(row, 0, K - 1)
        teacher_lp = shifted[b, rc, at]

        pmap = None
        if "sdpo_placebo_map" in micro_batch.keys():
            pmap = micro_batch["sdpo_placebo_map"]
        placebo_lp = None
        if pmap is not None and os.environ.get("SDPO_DEBIAS", "1") == "1":
            prow = micro_batch["sdpo_placebo_row"].long()
            pok = prow >= 0
            pat = torch.clamp(pmap.long() - off - 1, min=0, max=keep - 2)
            _lp = shifted[b, torch.clamp(prow, 0, K - 1), pat]
            placebo_lp = torch.where(pok, _lp, torch.full_like(_lp, float("nan")))
        if not want_topk:
            return teacher_lp, None, None, placebo_lp
        kk = _tlp.size(-1)
        _tlp = _tlp.view(bsz, K, keep - 1, kk)
        _ti = _ti.view(bsz, K, keep - 1, kk)
        return teacher_lp, _ti[b, rc, at], _tlp[b, rc, at], placebo_lp

    def _sdpo_probe_grad_dtypes(self):
        """Report which parameters carry which grad dtype, once."""
        if getattr(self, "_dtype_probed", False):
            return
        self._dtype_probed = True
        from collections import Counter
        counts, examples = Counter(), {}
        n_none = sum(1 for _, prm in self.actor_module.named_parameters() if prm.grad is None)
        print(f"[SDPO] params with NO grad: {n_none}")
        for n, prm in self.actor_module.named_parameters():
            if prm.grad is None:
                print(f"[SDPO]   no-grad: {n}")
            if prm.grad is not None:
                d = str(prm.grad.dtype)
                counts[d] += 1
                examples.setdefault(d, []).append(n)
        print(f"[SDPO] grad dtypes: {dict(counts)}")
        for d, names in examples.items():
            print(f"[SDPO]   {d}: {len(names)} params, e.g. {names[:3]}")

    def _optimizer_step(self):
        if getattr(self, "_sdpo_active", False) or os.environ.get("SDPO_PROBE_GRADS") == "1":
            self._sdpo_probe_grad_dtypes()
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()
        self._sdpo_step = getattr(self, "_sdpo_step", 0) + 1

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        sdpo_keys = ["sdpo_teacher_input_ids", "sdpo_teacher_attention_mask",
                     "sdpo_index_map", "sdpo_row", "sdpo_mask"]
        sdpo_opt_keys = [k for k in ("sdpo_placebo_map", "sdpo_placebo_row")
                         if k in data.batch.keys()]
        self._sdpo_active = (float(self.config.get("sdpo_coef", 0.0)) != 0.0
                             and all(k in data.batch.keys() for k in sdpo_keys))
        if self._sdpo_active:
            select_keys += sdpo_keys + sdpo_opt_keys
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    _sdpo_teacher_lp = None
                    _sdpo_teacher_topk = None
                    _sdpo_plac_lp = None
                    if getattr(self, "_sdpo_active", False):
                        _any = data["sdpo_mask"].sum()
                        if torch.distributed.is_initialized():
                            torch.distributed.all_reduce(
                                _any, op=torch.distributed.ReduceOp.MAX)
                        if _any > 0:
                            with torch.no_grad():
                                _sdpo_teacher_lp, _tidx, _tlp, _sdpo_plac_lp = self._sdpo_teacher_logprobs(data, temperature)
                            if _tidx is not None:
                                data["sdpo_topk_idx"] = _tidx
                                _sdpo_teacher_topk = _tlp

                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
                    _sdpo_student_topk = self._sdpo_student_topk

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if getattr(self, "_sdpo_active", False):
                        sdpo_mask = data["sdpo_mask"].to(log_prob.dtype)
                        if not getattr(self, "_mask_probed", False):
                            self._mask_probed = True
                            print(f"[SDPO LOSS] mask.sum={sdpo_mask.sum().item():.0f} "
                                  f"shape={tuple(sdpo_mask.shape)} "
                                  f"resp_len={log_prob.shape[-1]} "
                                  f"teacher_rows={tuple(data['sdpo_teacher_input_ids'].shape)}")
                        if True:
                            if _sdpo_teacher_lp is None:
                                teacher_lp = torch.zeros_like(log_prob)
                                sdpo_mask = sdpo_mask * 0
                            else:
                                teacher_lp = _sdpo_teacher_lp
                            teacher_lp = teacher_lp.to(log_prob.dtype)

                            _warm = int(os.environ.get("SDPO_WARMUP_STEPS", "0"))
                            if _warm > 0 and getattr(self, "_sdpo_step", 0) < _warm:
                                sdpo_mask = sdpo_mask * 0
                            _agate = float(os.environ.get("SDPO_ADV_GATE", "0"))
                            if _agate > 0:
                                _a = advantages.abs().detach()
                                sdpo_mask = sdpo_mask * (_a <= _agate).to(sdpo_mask.dtype)
                            _dbg_raw = None
                            if _sdpo_plac_lp is not None:
                                _p = _sdpo_plac_lp.to(log_prob.dtype)
                                _p = torch.where(torch.isnan(_p), log_prob.detach(), _p)
                                _dbg_raw = (teacher_lp - log_prob).detach()
                                teacher_lp = log_prob.detach() + (teacher_lp - _p)
                            if os.environ.get("SDPO_NULL_PROBE") == "1" and not getattr(self, "_null_probed", False):
                                _m = sdpo_mask.bool()
                                if _m.any():
                                    self._null_probed = True
                                    _diff = (teacher_lp - log_prob)[_m].abs()
                                    print(f"[SDPO NULL PROBE] n={_m.sum().item()} "
                                          f"max|teacher-student|={_diff.max().item():.6f} "
                                          f"mean={_diff.mean().item():.6f} "
                                          f"frac>1e-2={(_diff > 1e-2).float().mean().item():.4f}  "
                                          f"(all must be ~0; nonzero => row/index/pad/forward bug)")
                            estimator = self.config.get("sdpo_estimator", "ref")
                            kld = None
                            if estimator == "sdar":
                                _t = teacher_lp.detach()
                                _delta = _t - log_prob.detach()
                                _gate = torch.sigmoid(SDAR_BETA * _delta).detach()
                                _gkl = _gate * (_t - log_prob)
                                sdpo_loss = agg_loss(loss_mat=_gkl,
                                                     loss_mask=sdpo_mask,
                                                     loss_agg_mode="token-mean")
                                kld = None
                                with torch.no_grad():
                                    _ms = sdpo_mask.sum().clamp(min=1)
                                    metrics["sdar/gate_mean"] = (
                                        (_gate * sdpo_mask).sum() / _ms).item()
                                    metrics["sdar/gate_active_ratio"] = (
                                        ((_gate > 0.5).float() * sdpo_mask).sum() / _ms).item()
                                    metrics["sdar/teacher_gap_mean"] = (
                                        (_delta * sdpo_mask).sum() / _ms).item()
                            elif estimator == "distill":
                                if _sdpo_teacher_topk is None:
                                    sdpo_loss = (log_prob * 0.0).sum()
                                elif _sdpo_student_topk is None:
                                    raise RuntimeError(
                                        "SDPO distill: teacher produced top-K indices but the "
                                        "student forward produced none. use_remove_padding="
                                        f"{self.use_remove_padding} (distill needs it True).")
                                else:
                                    t = _sdpo_teacher_topk.to(log_prob.dtype)
                                    s = _sdpo_student_topk.to(log_prob.dtype)
                                    _lim = 1.0 - 1e-4
                                    t_tail = torch.log1p(-t.exp().sum(-1).clamp(max=_lim))
                                    s_tail = torch.log1p(-s.exp().sum(-1).clamp(max=_lim))
                                    t = torch.cat([t, t_tail.unsqueeze(-1)], dim=-1)
                                    s = torch.cat([s, s_tail.unsqueeze(-1)], dim=-1)
                                    _kl = (s.exp() * (s - t)).sum(-1)
                                    if SDPO_NORM == "full":
                                        _n = response_mask.sum().clamp(min=1.0)
                                    else:
                                        _n = sdpo_mask.sum().clamp(min=1.0)
                                    sdpo_loss = (_kl * sdpo_mask).sum() / _n
                            elif estimator == "ref":
                                _d = (teacher_lp - log_prob).detach()
                                _pt = -(_d * log_prob) * sdpo_mask
                                if SDPO_NORM == "full":
                                    _len = response_mask.sum(dim=1).clamp(min=1.0)
                                else:
                                    _len = sdpo_mask.sum(dim=1).clamp(min=1.0)
                                sdpo_loss = (_pt.sum(dim=1) / _len).mean()
                                kld = None
                            elif estimator == "k3":
                                kld = kl_penalty(logprob=log_prob, ref_logprob=teacher_lp,
                                                 kl_penalty="low_var_kl")
                            else:
                                kld = ((log_prob - teacher_lp).detach() + 1.0) * log_prob
                            if kld is not None:
                                sdpo_loss = agg_loss(loss_mat=kld, loss_mask=sdpo_mask,
                                                     loss_agg_mode=loss_agg_mode)
                            sdpo_loss = torch.nan_to_num(sdpo_loss)
                            if os.environ.get("SDPO_ONLY") == "1":
                                self._sdpo_only_seen = getattr(self, "_sdpo_only_seen", 0)
                                self._sdpo_only_tot = getattr(self, "_sdpo_only_tot", 0) + 1
                                if float(sdpo_mask.sum()) > 0.0:
                                    self._sdpo_only_seen += 1
                                policy_loss = sdpo_loss * float(self.config.sdpo_coef)
                            else:
                                policy_loss = policy_loss + sdpo_loss * float(self.config.sdpo_coef)

                            with torch.no_grad():
                                if _dbg_raw is not None:
                                    _m = sdpo_mask.bool()
                                    if _m.any():
                                        _r = _dbg_raw[_m].abs().mean()
                                        _dbias = (teacher_lp - log_prob).detach()[_m].abs().mean()
                                        metrics["actor/sdpo_delta_raw_absmean"] = _r.item()
                                        metrics["actor/sdpo_delta_debiased_absmean"] = _dbias.item()
                                        metrics["actor/sdpo_placebo_share"] = (
                                            1.0 - (_dbias / _r.clamp(min=1e-6))).item()
                                d = (log_prob - teacher_lp)[sdpo_mask.bool()]
                                if d.numel() == 0:
                                    d = torch.zeros(1, device=log_prob.device)
                                metrics["actor/sdpo_loss"] = sdpo_loss.detach().item()
                                metrics["actor/sdpo_delta_mean"] = d.mean().item()
                                metrics["actor/sdpo_delta_absmean"] = d.abs().mean().item()
                                metrics["actor/sdpo_delta_p95"] = torch.quantile(
                                    d.abs().float(), 0.95).item()
                                metrics["actor/sdpo_frac_tokens"] = (
                                    sdpo_mask.sum() / sdpo_mask.numel()).item()

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        if os.environ.get("SDPO_ONLY") == "1":
            seen = getattr(self, "_sdpo_only_seen", 0)
            tot = getattr(self, "_sdpo_only_tot", 0)
            self._sdpo_only_seen = self._sdpo_only_tot = 0
            if tot and seen == 0:
                raise RuntimeError(
                    f"SDPO_ONLY=1 but NO micro-batch in this step had any hinted token "
                    f"(0/{tot}). Total loss is 0 and nothing trains. Coverage has "
                    f"collapsed -- check privileged fetches and traj-carrying-a-hint.")
            metrics["actor/sdpo_only_active_mb"] = seen / max(tot, 1)
        return metrics
