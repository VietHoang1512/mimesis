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

# K for SDPO full_distillation. 20 is the reference default
# (online_sdpo_updater_config.py:26, distillation_topk).
SDPO_TOPK = int(os.environ.get("SDPO_TOPK", "20"))
# "masked" (current) | "full" (reference semantics). See the normalisation
# note in the `ref` branch of the loss below.
SDPO_NORM = os.environ.get("SDPO_NORM", "masked")
# Sigmoid gate sharpness for the SDAR estimator. 5.0 is the paper default
# (sdar_utils.py:18); 0.0 reproduces their naive GRPO+OPSD baseline (gate==0.5).
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

        # SDPO compares two forwards of the SAME weights under different
        # contexts, and update_policy() calls .train() first. Nonzero dropout
        # would give the two forwards different masks, so the KL would measure
        # dropout noise rather than the hint's effect -- no error, just a
        # meaningless term. Recurse: Qwen3.5 and the VL models nest this at
        # text_config.attention_dropout, and a flat read reports "no dropout
        # key" on exactly the checkpoints most likely to be trained.
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
        # Cleared on every call: a stale tensor from the previous micro-batch
        # would be silently reused by the loss and would have the wrong shape
        # only when batch sizes happen to differ.
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

                    # SDPO full_distillation: student log-probs at the teacher's
                    # top-K token ids, for the same positions.
                    #
                    # Computed HERE, from the logits the main forward already
                    # produced, rather than from a second forward over the same
                    # prefixes -- that would roughly double the graded compute
                    # for a distribution this pass already has.
                    #
                    # gather(logits) - logsumexp(logits) instead of
                    # log_softmax(...).gather(...): log_softmax would materialise
                    # a second (total_nnz, vocab) tensor, ~14 GB at this vocab
                    # and sequence length. logsumexp collapses vocab to a column.
                    #
                    # Must run BEFORE logprobs_from_logits, which is called with
                    # inplace_backward and rewrites logits_rmpad during backward.
                    self._sdpo_student_topk = None
                    # NOT micro_batch.get("sdpo_topk_idx"): this function is also
                    # called from compute_log_prob, where micro_batch is a
                    # TensorDict holding only input_ids/attention_mask/
                    # position_ids/responses -- and TensorDict.get RAISES on a
                    # missing key rather than returning None the way dict.get
                    # does. Membership-test first so both call sites work; a
                    # plain .get here died on the very first log-prob pass.
                    _tk = None
                    if "sdpo_topk_idx" in micro_batch.keys():
                        _tk = micro_batch["sdpo_topk_idx"]
                    if _tk is not None:
                        assert not self.use_ulysses_sp, (
                            "SDPO distill does not support ulysses sp>1: logits_rmpad "
                            "is sharded and the gather would need the same "
                            "gather_outpus_and_unpad treatment as log_probs.")
                        kk = _tk.size(-1)
                        # Response position j lives at full-sequence position
                        # seqlen-response_length-1+j, the same offset the
                        # response slice below undoes.
                        _full = torch.zeros((batch_size, seqlen, kk),
                                            dtype=torch.long, device=_tk.device)
                        _s = seqlen - response_length - 1
                        _full[:, _s:_s + response_length, :] = _tk.long()
                        _rm = _full.view(-1, kk)[indices]        # (total_nnz, k)
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
                # Same unpad->pad round trip as log_probs, hidden dim k instead
                # of 1, so the response slice below lines up identically.
                if self._sdpo_student_topk is not None:
                    self._sdpo_student_topk = pad_input(
                        hidden_states=self._sdpo_student_topk,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )[:, -response_length - 1 : -1, :]     # (bsz, resp_len, k)

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
        """log pi(y_i | H_t + hint_t), aligned to the student's response positions.

        The hindsight sequence contains the student's own response tokens, moved
        later by however many hint tokens were spliced in ahead of them;
        sdpo_index_map records where each one landed. So one forward over the
        hindsight sequence yields every teacher log-prob required, and no
        generation is involved.

        Scores the WHOLE sequence and then indexes, rather than gathering logits
        at the mapped positions first. Gathering first would materialise a
        second (bsz, response_len, vocab) tensor -- ~2.5 GB at this vocab size --
        whereas logprobs_from_logits collapses vocab immediately and leaves a
        (bsz, T-1) tensor to index into. Peak memory then matches the student
        forward, which already materialises full logits.

        The log-prob of the token at teacher position m is produced by the logit
        at m-1, hence the shift.

        Deliberately does NOT toggle the module to eval. Doing so dodged the
        activation-offload counter (activation_offload.py:464 bypasses the
        handler when not training), but flipping an FSDP module's training flag
        between forward and backward leaves some mixed-precision handles
        emitting fp32 grads while the rest stay bf16, and clip_grad_norm_ then
        refuses the batch (fully_sharded_data_parallel.py:2122).
        Activation offloading is instead disabled for the whole run whenever the
        SDPO term is active; the shipped launchers set
        enable_activation_offload=False unconditionally, so the GRPO control and
        the SDPO arm stay comparable (scripts/run_sdpo.sh). Dropout is covered
        by the assert in __init__ rather than by eval mode.
        """
        ids = micro_batch["sdpo_teacher_input_ids"].long()        # (bsz, K, W)
        attn = micro_batch["sdpo_teacher_attention_mask"]
        index_map = micro_batch["sdpo_index_map"].long()          # (bsz, R)
        row = micro_batch["sdpo_row"].long()                      # (bsz, R)
        sdpo_mask_bool = micro_batch["sdpo_mask"].bool()
        want_topk = self.config.get("sdpo_estimator", "ref") == "distill"
        bsz, K, W = ids.shape
        flat_ids, flat_attn = ids.reshape(bsz * K, W), attn.reshape(bsz * K, W)

        # ONE batched forward, mirroring the reference, which calls
        # _token_logps_of_given_y once over the whole batch
        # (offline_sdpo_trainer.py:204-209) rather than looping.
        #
        # The rows are LEFT-padded, so y_t is the tail of every row at the same
        # offset and a single logits_to_keep serves them all. The previous
        # per-row loop existed only because right padding put y_t in the middle
        # of each row; it issued bsz*K full FSDP parameter all-gathers for an 8B
        # model plus a GPU->CPU sync per row, and the NCCL watchdog killed the
        # step. Before that, the same loop's data-dependent `continue`
        # deadlocked ranks outright. No loop, no per-row sync, and no
        # data-dependent control flow around a forward: the collective count is
        # now fixed at one regardless of what the batch contains.
        _pm_all = None
        if "sdpo_placebo_map" in micro_batch.keys():
            _pm_all = micro_batch["sdpo_placebo_map"].long()
        # `keep` must cover the EARLIEST mapped position of either gather; a
        # placebo row has a different left-pad, so its index can land before the
        # real one and would otherwise be clamped onto the wrong token.
        _need_src = index_map if _pm_all is None else torch.minimum(index_map, torch.where(_pm_all > 0, _pm_all, index_map))
        need = _need_src[sdpo_mask_bool] if sdpo_mask_bool.any() else None
        first = int(need.min()) - 1 if need is not None and need.numel() else W - 2
        first = max(min(first, W - 2), 0)
        keep = W - first                       # covers every mapped position
        off = W - keep

        was_training = self.actor_module.training
        # CHUNKED over rows. One forward over all bsz*K rows is what made K=8
        # OOM (11.98 GiB at modeling_qwen3.py:90), and "SDPO on every turn"
        # needs K of order the trajectory length, not 3. Rows are independent,
        # so chunking costs time, not correctness, and bounds peak memory at
        # SDPO_TEACHER_CHUNK rows regardless of K.
        #
        # Rank-invariant by construction: bsz and K come from config, identical
        # on every rank, so every rank issues the same number of forwards. A
        # data-dependent chunk count here would deadlock FSDP the way the old
        # per-row `continue` did.
        _CH = max(1, int(os.environ.get("SDPO_TEACHER_CHUNK", "4")))
        _nrows = flat_ids.size(0)
        _nch = (_nrows + _CH - 1) // _CH
        # RANK-INVARIANT CHUNK COUNT. With use_dynamic_bsz the micro-batch size
        # differs per rank (rearrange_micro_batches splits by token budget), so
        # bsz*K -- and therefore the number of chunks -- is NOT the same
        # everywhere. Ranks then issue different numbers of collectives and the
        # job hangs until the NCCL watchdog kills it: exactly what happened to
        # an early multi-rank run (_ALLGATHER_BASE, Timeout=1800000ms). The
        # comment above claimed invariance "by construction" because bsz and K
        # come from config; K does, bsz does not.
        #
        # all_reduce(MAX) makes the loop length a property of the whole batch.
        # Ranks that run out of rows repeat the final chunk and discard it, so
        # every rank issues identical collectives. Same fix as the `continue`
        # that deadlocked ranks earlier.
        if torch.distributed.is_initialized():
            _t = torch.tensor([_nch], device=flat_ids.device)
            torch.distributed.all_reduce(_t, op=torch.distributed.ReduceOp.MAX)
            _nch = int(_t.item())
        _pieces, _tv_p, _ti_p = [], [], []
        with suspend_activation_offload():
            for _ci in range(_nch):
                _c0 = _ci * _CH
                _real = _c0 < _nrows
                if not _real:                      # padding forward, discarded
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
                    # bf16 in place, exactly as _forward_micro_batch does.
                    # Casting here made the teacher the only fp32 consumer of
                    # lm_head and split the root FSDP flat param's grad dtype
                    # from the layers'.
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
        teacher_lp = shifted[b, rc, at]                            # (bsz, R)

        # Placebo control variate. The same y_t scored under an UNRELATED hint,
        # from rows packed in the second half of the same block, so this costs
        # no extra forward -- only a second gather.
        #
        # Inserting ANY extra user turn moves the teacher's conditional by
        # ~0.187 nats/token, measured identical across three unrelated hint
        # designs. That component carries no hindsight information and prompt
        # wording cannot remove it. Subtracting it leaves
        #     (teacher_real - student) - (teacher_placebo - student)
        #   =  teacher_real - teacher_placebo
        # so the student cancels and what remains is the part of the signal that
        # depends on WHICH hint was given.
        pmap = None
        if "sdpo_placebo_map" in micro_batch.keys():
            pmap = micro_batch["sdpo_placebo_map"]
        placebo_lp = None
        if pmap is not None and os.environ.get("SDPO_DEBIAS", "1") == "1":
            prow = micro_batch["sdpo_placebo_row"].long()
            pok = prow >= 0                       # -1 == no placebo for this token
            pat = torch.clamp(pmap.long() - off - 1, min=0, max=keep - 2)
            _lp = shifted[b, torch.clamp(prow, 0, K - 1), pat]
            # NaN where there is no placebo. The correct fallback is the
            # STUDENT's log-prob -- delta must degrade to (teacher - student),
            # the raw signal -- but log_prob is not in scope here, so mark the
            # slot and resolve it in the loss. Falling back to teacher_lp
            # instead makes (teacher - placebo) zero, which does not disable the
            # debias, it disables SDPO -- runs using that fallback all read
            # delta 0.000.
            placebo_lp = torch.where(pok, _lp, torch.full_like(_lp, float("nan")))
        if not want_topk:
            return teacher_lp, None, None, placebo_lp
        kk = _tlp.size(-1)
        _tlp = _tlp.view(bsz, K, keep - 1, kk)
        _ti = _ti.view(bsz, K, keep - 1, kk)
        return teacher_lp, _ti[b, rc, at], _tlp[b, rc, at], placebo_lp

    def _sdpo_probe_grad_dtypes(self):
        """Report which parameters carry which grad dtype, once.

        FSDP's clip_grad_norm_ refuses a mixed-dtype batch but does not say
        WHICH parameters disagree. The two likelier explanations are ruled out
        -- the loss is provably uniform bf16, and the eval/train toggle is gone
        -- so the parameter names are the evidence that separates what remains.
        """
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
        # SDPO_PROBE_GRADS=1 fires the probe even with SDPO off, which is the
        # control this needed all along: if the baseline leaves the root flat
        # param WITHOUT a grad, clip_grad_norm_ normally sees only the 36 bf16
        # layers and passes -- meaning the teacher forward is what creates the
        # fp32 outlier, not what changes its dtype. Those are different bugs
        # with different fixes, and the probe is what tells them apart.
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
        # select_keys is an allowlist, so omitting a key here is a silent no-op
        # rather than an error -- the SDPO tensors would simply never reach the
        # micro-batch and the term would quietly evaluate to nothing.
        sdpo_keys = ["sdpo_teacher_input_ids", "sdpo_teacher_attention_mask",
                     "sdpo_index_map", "sdpo_row", "sdpo_mask"]
        # Placebo control variate. Kept OUT of sdpo_keys so a batch produced by
        # a rollout without them still activates SDPO instead of silently
        # disabling it; added to the allowlist only when actually present.
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
                    # Teacher forward FIRST, before the student's.
                    #
                    # It used to sit between the student forward and
                    # loss.backward(), which meant a second top-level call into
                    # the FSDP root module inside an open forward->backward
                    # pair: the root reshards on its post-forward hook and its
                    # gradient then came back fp32 while all 36 layer flat
                    # params stayed bf16, so clip_grad_norm_ rejected the batch
                    # (the probe pinned it to exactly one param,
                    # `_fsdp_wrapped_module._flat_param`, which holds
                    # embed_tokens, the final norm and lm_head).
                    # Running it before leaves the student's pair contiguous.
                    # Skip the teacher forward when NO row in this micro-batch
                    # carries a hint -- but decide that identically on every
                    # rank, or the forward counts diverge and FSDP deadlocks.
                    # all_reduce(MAX) makes the branch a function of
                    # the whole batch, not this rank's slice.
                    #
                    # Necessary because the forward is not free even when its
                    # output is masked away: an early run ran it unconditionally
                    # with sdpo_frac_tokens=0.000 and sdpo_loss=0.000 -- so it
                    # contributed nothing to the loss -- yet reward collapsed to
                    # 0.000, while its twin (identical rollout, sdpo_coef=0,
                    # hence no teacher forward) scored 0.062/0.141.
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
                                # Hand the teacher's support to the student
                                # forward, which gathers its own log-probs at
                                # the same ids. Rank-invariant: every rank got
                                # here through the same all_reduce'd branch.
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

                    # ---- SDPO: L = L_RLVR + alpha * KL(pi(.|H) || pi(.|H+hint))
                    #
                    # A separate additive term, never an advantage: nothing here
                    # touches the GRPO path, so L_RLVR is unchanged by
                    # construction rather than by careful masking.
                    #
                    # The teacher forward runs HERE, inside the micro-batch
                    # loop, not precomputed alongside old_log_probs.
                    # train_batch_size / ppo_mini_batch_size is 8 optimizer steps
                    # per rollout batch, so a precomputed teacher would make this
                    # KL measure the hindsight effect PLUS however far the policy
                    # had drifted within the batch. With ppo_epochs=1 each row is
                    # visited once, so in-loop costs the same total tokens.
                    # (A precomputed teacher is correct when the term is
                    # injected into advantages instead -- there it is detached
                    # anyway and PPO's ratio absorbs the drift. A loss term has
                    # no such correction.)
                    if getattr(self, "_sdpo_active", False):
                        sdpo_mask = data["sdpo_mask"].to(log_prob.dtype)
                        if not getattr(self, "_mask_probed", False):
                            self._mask_probed = True
                            print(f"[SDPO LOSS] mask.sum={sdpo_mask.sum().item():.0f} "
                                  f"shape={tuple(sdpo_mask.shape)} "
                                  f"resp_len={log_prob.shape[-1]} "
                                  f"teacher_rows={tuple(data['sdpo_teacher_input_ids'].shape)}")
                        # NO `if sdpo_mask.sum() > 0` here. Whether a micro-batch
                        # contains any hinted turn is data, and it differs per
                        # rank, so gating the teacher forward and the loss term
                        # on it makes ranks build different backward graphs --
                        # which surfaced as ranks reporting different grad dtypes
                        # ({fp32:1, bf16:36} on some, {fp32:37} on others) and
                        # clip_grad_norm_ rejecting the batch. Same
                        # mistake as the `continue` that deadlocked ranks: any
                        # branch around a forward inside update_policy has to be
                        # rank-invariant. Always run it; the mask makes the
                        # contribution zero when there is nothing to score.
                        if True:
                            # Zeros, not None: the mask below already makes the
                            # contribution nil, but execution still falls through
                            # to .to(log_prob.dtype) and every arithmetic line
                            # after it. Returning None there crashed with an
                            # AttributeError on the first micro-batch that
                            # had no hinted row.
                            if _sdpo_teacher_lp is None:
                                teacher_lp = torch.zeros_like(log_prob)
                                sdpo_mask = sdpo_mask * 0        # nothing to score
                            else:
                                teacher_lp = _sdpo_teacher_lp
                            # Match log_prob's dtype. The teacher path
                            # accumulates in fp32 so chunks concatenate
                            # consistently, but leaving it fp32 here promotes
                            # policy_loss to fp32 while every other term is
                            # bf16, and FSDP's clip_grad_norm_ then refuses the
                            # mixed-dtype gradients (fully_sharded_data_parallel
                            # .py:2122).
                            teacher_lp = teacher_lp.to(log_prob.dtype)

                            # --- composition gates ---------------------------
                            # The reference runs SDPO as the SOLE objective: no
                            # reward, 15 interactions, LoRA. Here it rides on a
                            # GRPO signal that already takes the policy from
                            # 0.01 to 0.87, and the two push the SAME token by
                            # different criteria -- reward vs the teacher's
                            # opinion. Where they disagree the sum is worse than
                            # either, which is what Ditto's SDPO+GRPO ablation
                            # found and what every monotone-in-alpha curve here
                            # has shown.
                            #
                            # WARMUP: hold SDPO off while the policy is still
                            # learning tool-call FORMAT. The hint prompt
                            # deliberately refuses to comment on format, so
                            # before that is solved the term is orthogonal noise
                            # applied exactly when the reward gradient is
                            # steepest.
                            _warm = int(os.environ.get("SDPO_WARMUP_STEPS", "0"))
                            if _warm > 0 and getattr(self, "_sdpo_step", 0) < _warm:
                                sdpo_mask = sdpo_mask * 0
                            # ADV GATE: let the reward lead where it has an
                            # opinion. |advantage| near zero means GRPO is flat
                            # on this token and hindsight is the only signal
                            # available; large |advantage| means the reward
                            # already knows and SDPO can only fight it.
                            _agate = float(os.environ.get("SDPO_ADV_GATE", "0"))
                            if _agate > 0:
                                _a = advantages.abs().detach()
                                sdpo_mask = sdpo_mask * (_a <= _agate).to(sdpo_mask.dtype)
                            # -------------------------------------------------
                            # Debias with the placebo control variate. The loss
                            # below is written in terms of `teacher_lp`, so the
                            # cleanest injection point is here: replace the
                            # teacher's log-prob with
                            #     log_prob + (teacher_real - teacher_placebo)
                            # which makes every downstream delta
                            # (teacher_lp - log_prob) equal the DEBIASED signal
                            # while leaving the loss expressions untouched.
                            # With no placebo the two gathers coincide, the
                            # correction is identically 0, and this is a no-op.
                            _dbg_raw = None
                            if _sdpo_plac_lp is not None:
                                _p = _sdpo_plac_lp.to(log_prob.dtype)
                                # NaN slots have no placebo -> fall back to the
                                # student, so the delta degrades to the RAW
                                # signal rather than to zero.
                                _p = torch.where(torch.isnan(_p), log_prob.detach(), _p)
                                _dbg_raw = (teacher_lp - log_prob).detach()
                                teacher_lp = log_prob.detach() + (teacher_lp - _p)
                            # SDPO_NULL_PROBE=1: the teacher row was built with
                            # no hint block, so it is a pure prefix of the
                            # student's sequence and teacher_lp must equal
                            # log_prob on every selected token. This is the only
                            # check that exercises row construction, padding,
                            # position_ids, index_map, the keep/offset
                            # arithmetic and the forward TOGETHER, on the real
                            # model. Prints rather than asserts so one run
                            # reports the size of the error instead of dying on
                            # the first token of float noise.
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
                            # Set BEFORE the branch, not inside each arm. The
                            # `distill` arm computes sdpo_loss directly and left
                            # kld unbound, so the `if kld is not None` below
                            # raised UnboundLocalError on the first optimizer
                            # step. Initialising here means a future
                            # estimator cannot reintroduce that.
                            kld = None
                            if estimator == "sdar":
                                # SDAR (arXiv 2605.15155), sdar_utils.py:42-52.
                                #
                                # The gate is the entire point. Weight is
                                # sigma(beta*delta) in (0,1) -- ALWAYS POSITIVE
                                # -- so a token the teacher dislikes is
                                # attenuated toward zero, never pushed down.
                                #
                                # The `ref` loss weights by raw delta, which is
                                # signed and unbounded: measured mean -0.22,
                                # min -14.9. Most tokens therefore got a
                                # NEGATIVE weight, and the term actively
                                # suppressed the model's own outputs with up to
                                # 14.9x strength. That is exactly the
                                # "negative teacher rejections" instability the
                                # SDAR abstract exists to fix, and it accounts
                                # for every result so far: more coverage worse,
                                # higher alpha worse, SDPO-only collapsing.
                                #
                                # Note the naive GRPO+OPSD baseline is
                                # beta=0 -> gate == 0.5 constant. Even that
                                # weakest baseline beats plain GRPO in SDAR's
                                # Table 1 (81.2 vs 75.0 ALFWorld). The `ref`
                                # loss was never GRPO+OPSD; it was weaker than
                                # that weakest arm.
                                _t = teacher_lp.detach()
                                _delta = _t - log_prob.detach()
                                _gate = torch.sigmoid(SDAR_BETA * _delta).detach()
                                # grad flows through log_prob only
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
                                # full_distillation: the reference's SHIPPED
                                # online default (online_sdpo_updater.py:731-737,
                                # config:25, eval_online_sdpo.sh:50) -- not the
                                # scalar-signal loss, which is the `ref` branch
                                # below and is what offline_sdpo uses.
                                #
                                # Why it matters here: -(d * log pi) reduces to
                                # c_i * grad log pi(y_i), a scalar times the
                                # gradient of a token the model ALREADY emitted.
                                # It can only reweight y_t. Matching the
                                # teacher's top-K distribution per position can
                                # move mass onto tokens y_t never contained --
                                # which in a goal-directed gym is the whole
                                # correction, since the fix is usually a
                                # DIFFERENT action rather than the same action
                                # made likelier.
                                if _sdpo_teacher_topk is None:
                                    # Legitimate: no hinted row in this
                                    # micro-batch, so there is nothing to
                                    # distil. Keep the graph, zero the value.
                                    sdpo_loss = (log_prob * 0.0).sum()
                                elif _sdpo_student_topk is None:
                                    # NOT legitimate: the teacher handed over
                                    # indices and the student forward did not
                                    # gather at them. Only the remove-padding
                                    # branch of _forward_micro_batch implements
                                    # the gather, so this fires if rmpad is off.
                                    # Raise rather than fall back to a zero
                                    # loss: a silent no-op here looks exactly
                                    # like "SDPO does not help" on the reward
                                    # curve, which is the single most expensive
                                    # failure mode in this project.
                                    raise RuntimeError(
                                        "SDPO distill: teacher produced top-K indices but the "
                                        "student forward produced none. use_remove_padding="
                                        f"{self.use_remove_padding} (distill needs it True).")
                                else:
                                    t = _sdpo_teacher_topk.to(log_prob.dtype)
                                    s = _sdpo_student_topk.to(log_prob.dtype)
                                    # Tail bucket (config:27, implemented :747-753):
                                    # top-K alone is not a distribution, and
                                    # without the remainder the KL ignores how
                                    # much mass each policy puts OUTSIDE the K.
                                    # clamp keeps log1p finite when the top-K
                                    # already covers ~all mass, which is common
                                    # on confident format tokens.
                                    _lim = 1.0 - 1e-4
                                    t_tail = torch.log1p(-t.exp().sum(-1).clamp(max=_lim))
                                    s_tail = torch.log1p(-s.exp().sum(-1).clamp(max=_lim))
                                    t = torch.cat([t, t_tail.unsqueeze(-1)], dim=-1)
                                    s = torch.cat([s, s_tail.unsqueeze(-1)], dim=-1)
                                    # KL(student || teacher) -- reverse KL, the
                                    # direction SDPO wants: sum_i p_s log(p_s/p_t).
                                    # Gradient flows through s only; t came from a
                                    # no_grad forward.
                                    _kl = (s.exp() * (s - t)).sum(-1)
                                    # .sum()/total_tokens, as the reference
                                    # normalises full_distillation -- NOT the
                                    # per-sequence mean that simple_signal uses.
                                    if SDPO_NORM == "full":
                                        _n = response_mask.sum().clamp(min=1.0)
                                    else:
                                        _n = sdpo_mask.sum().clamp(min=1.0)
                                    sdpo_loss = (_kl * sdpo_mask).sum() / _n
                            elif estimator == "ref":
                                # The reference implementation's loss, verbatim
                                # (offline_sdpo_trainer.py:222-228):
                                #     diff = (logps_xo - logps_x).detach()
                                #     per_token = -(diff * logps_x) * mask
                                #     loss = (per_token.sum(1) / lengths).mean()
                                # Neither of the two earlier estimators tried
                                # here matched it.
                                # k3 computes exp(D) - D - 1, a different
                                # function; "surrogate" added a +1 the reference
                                # does not have, and was unclamped so it blew up.
                                # Note the normalisation too: per-SEQUENCE by
                                # token count, then mean over sequences -- not
                                # verl's token-mean, which weights long turns
                                # more and is not what the paper optimises.
                                _d = (teacher_lp - log_prob).detach()
                                _pt = -(_d * log_prob) * sdpo_mask
                                # Normalisation note. The reference divides by the
                                # FULL completion length, which for it equals
                                # the masked count (its mask is everything).
                                # Here the mask covers ~9.5% of tokens, so the
                                # two differ: dividing by the masked count
                                # up-weights rows with few hinted tokens, so a
                                # row with one hinted turn counts as much as a
                                # row with nine. NORM=full weights each row by
                                # how much hindsight it actually carries.
                                if SDPO_NORM == "full":
                                    _len = response_mask.sum(dim=1).clamp(min=1.0)
                                else:
                                    _len = sdpo_mask.sum(dim=1).clamp(min=1.0)
                                sdpo_loss = (_pt.sum(dim=1) / _len).mean()
                                kld = None
                            elif estimator == "k3":
                                # Schulman's low-variance estimator. Non-negative
                                # and cheap, but assumes pi_s ~= pi_t and verl
                                # clamps it to +/-10; an imperative hint can make
                                # the hindsight conditional much sharper than the
                                # paper's implicit user message, in which case
                                # tokens pin to the clamp and the loss becomes a
                                # hinge. Watch actor/sdpo_delta_p95 and switch.
                                kld = kl_penalty(logprob=log_prob, ref_logprob=teacher_lp,
                                                 kl_penalty="low_var_kl")
                            else:
                                # The reference implementation's reverse-KL
                                # surrogate (offline_sdpo_trainer.py:519):
                                # grad KL = E[(log pi_s - log pi_t + 1) grad log pi_s].
                                # Unclamped, so it degrades gracefully when the
                                # two policies are far apart.
                                kld = ((log_prob - teacher_lp).detach() + 1.0) * log_prob
                            if kld is not None:
                                sdpo_loss = agg_loss(loss_mat=kld, loss_mask=sdpo_mask,
                                                     loss_agg_mode=loss_agg_mode)
                            # An all-zero mask divides by zero in agg_loss; keep
                            # the graph and zero the value instead of branching.
                            sdpo_loss = torch.nan_to_num(sdpo_loss)
                            # SDPO_ONLY=1 -- the paper's actual configuration.
                            # They never combine SDPO with a reward: no RL term,
                            # no advantages, SDPO is the whole objective
                            # (main_online_sdpo.py:44-45 passes a reward stub
                            # returning zeros, beta=0.0, no ref model).
                            #
                            # This is the test that separates "the SDPO
                            # gradient is uninformative" from "it is
                            # informative but cannot compete with GRPO on 0.3%
                            # of tokens". If the policy learns anything at all
                            # here, the signal is real and the problem is
                            # composition/dilution.
                            if os.environ.get("SDPO_ONLY") == "1":
                                # SDPO is the ENTIRE objective here, so a zero
                                # loss means zero gradient and the model simply
                                # never trains. An early run did exactly that
                                # for 22 steps -- sdpo_loss 0.000, grad_norm
                                # 0.000, score frozen at the SFT init -- and
                                # that 0.000 is easily misread as "the objective
                                # cannot drive learning". Coverage had collapsed
                                # (0-3 of 64 trajectories hinted, 96% of
                                # privileged fetches unscoped). Fail loudly.
                                # Count, do NOT raise here. An individual
                                # micro-batch with no hinted trajectory is
                                # NORMAL -- at MICRO_BSZ=2 and ~50% coverage
                                # roughly a quarter of them are empty. Raising
                                # per micro-batch killed a perfectly healthy
                                # warm-started run. The pathology worth
                                # catching is a WHOLE STEP with no signal, which
                                # is checked after the loop.
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
                                        # >1 means the placebo carried more than
                                        # the real hint did: the subtraction is
                                        # removing signal, not noise.
                                        metrics["actor/sdpo_placebo_share"] = (
                                            1.0 - (_dbias / _r.clamp(min=1e-6))).item()
                                d = (log_prob - teacher_lp)[sdpo_mask.bool()]
                                if d.numel() == 0:
                                    d = torch.zeros(1, device=log_prob.device)
                                metrics["actor/sdpo_loss"] = sdpo_loss.detach().item()
                                metrics["actor/sdpo_delta_mean"] = d.mean().item()
                                # |delta| percentile picks the estimator and is
                                # the first thing to read: near zero means the
                                # hint moved nothing and the term is dead weight.
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
                # No micro-batch in the entire step carried a hinted token, so
                # the loss was 0 everywhere and nothing trained. That is the
                # cold-start spiral an early run died of silently for 22 steps.
                raise RuntimeError(
                    f"SDPO_ONLY=1 but NO micro-batch in this step had any hinted token "
                    f"(0/{tot}). Total loss is 0 and nothing trains. Coverage has "
                    f"collapsed -- check privileged fetches and traj-carrying-a-hint.")
            metrics["actor/sdpo_only_active_mb"] = seen / max(tot, 1)
        return metrics
