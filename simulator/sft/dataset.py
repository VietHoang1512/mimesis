# User-simulator SFT dataset for verl's sft_trainer.
#
# Wired via data.custom_cls.path=sft/dataset.py + data.custom_cls.name=OdysSimSFTDataset.
# Returns tensors matching what verl's sft_loss expects for pad_mode=no_padding:
#   {input_ids: (L,) long, position_ids: (L,) long, loss_mask: (L,) long}
#
# The role-swap / per-source processing lives in sft/sft_data.py::process.

import bisect
import glob as _glob
import os
from collections import OrderedDict

import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _wrap_tokenizer(tokenizer):
    """Kept for backward compatibility. TURNOFF_THINK is handled inside
    sft.sft_data._apply_chat_template, so no monkey-patching is needed here —
    patching tokenizer.apply_chat_template would swallow the TemplateError
    that Qwen3.5 raises for user-less prefixes and short-circuit verl's
    dummy-user fallback."""
    return tokenizer


def parse_files(files_arg) -> list[tuple[str, float]]:
    """Parse train_files / val_files into [(path, ratio), ...].

    Accepts a Hydra list, a whitespace-separated string, or a single string.
    Ratio suffix syntax: "/data/a.parquet:1.5" → oversample 1.5×.
    Glob expansion: "*" and "?" are expanded; the ratio applies to each shard.
    """
    if isinstance(files_arg, str):
        tokens = files_arg.strip().split()
    elif isinstance(files_arg, list | ListConfig):
        tokens = list(files_arg)
    else:
        tokens = [files_arg]

    result: list[tuple[str, float]] = []
    for token in tokens:
        token = str(token)
        parts = token.rsplit(":", 1)
        try:
            ratio = float(parts[-1])
            path = parts[0]
        except (ValueError, IndexError):
            ratio = 1.0
            path = token
        if "*" in path or "?" in path:
            matched = sorted(_glob.glob(path))
            if not matched:
                raise FileNotFoundError(f"Glob matched no parquet files: {path}")
            for p in matched:
                result.append((p, ratio))
        else:
            result.append((path, ratio))
    return result


class _ShardIndex:
    """Row-group index for a single parquet shard.

    Holds only ParquetFile metadata (cheap: schema + row-group offsets) and
    loads one row group at a time on demand. Row groups are cached LRU with a
    small capacity so total per-worker memory stays bounded regardless of how
    many shards or files the dataset spans.
    """

    def __init__(self, path: str, cache_size: int = 1):
        self.path = path
        # Lazy: only open the file when we actually need to read.
        self._pf: pq.ParquetFile | None = None
        self._rg_offsets: list[int] | None = None
        self._n_rows: int | None = None
        # rg_idx -> RecordBatch (single-chunk); OrderedDict for LRU semantics.
        self._rg_cache: OrderedDict[int, "pq.pyarrow.RecordBatch"] = OrderedDict()
        self._cache_size = max(1, cache_size)

    def _open(self):
        if self._pf is not None:
            return
        self._pf = pq.ParquetFile(self.path, memory_map=True)
        offsets = [0]
        meta = self._pf.metadata
        for i in range(meta.num_row_groups):
            offsets.append(offsets[-1] + meta.row_group(i).num_rows)
        self._rg_offsets = offsets
        self._n_rows = offsets[-1]

    @property
    def n_rows(self) -> int:
        if self._n_rows is None:
            # Fast path: read only the metadata footer.
            self._n_rows = pq.read_metadata(self.path).num_rows
        return self._n_rows

    def get_row(self, row_idx: int) -> dict:
        self._open()
        assert self._rg_offsets is not None
        rg_idx = bisect.bisect_right(self._rg_offsets, row_idx) - 1
        local_idx = row_idx - self._rg_offsets[rg_idx]

        batch = self._rg_cache.get(rg_idx)
        if batch is None:
            # Read this row group as a single Table then combine to one chunk.
            table = self._pf.read_row_group(rg_idx)
            batch = table.combine_chunks().to_batches(max_chunksize=table.num_rows)[0]
            self._rg_cache[rg_idx] = batch
            if len(self._rg_cache) > self._cache_size:
                self._rg_cache.popitem(last=False)
        else:
            self._rg_cache.move_to_end(rg_idx)

        raw = batch.slice(local_idx, 1).to_pydict()
        return {k: v[0] for k, v in raw.items()}


class OdysSimSFTDataset(Dataset):
    """Multi-source SFT dataset that plugs into verl's sft_trainer.

    Reads per-source SFT data (mid-training mix), routes each row through
    sft/sft_data.py::process to get an old-style
    {input_ids, responses, response_mask} dict, then flattens to the
    {input_ids, position_ids, loss_mask} schema used by verl's sft_loss.

    Memory model: only one parquet row group per shard is resident at any
    time (LRU, cache_size=1 by default). Init reads only footer metadata for
    each shard, so opening 600+ shards costs seconds and a few MB, not GB.

    Config knobs (all read from data.* in the hydra config):
      - seed                    (int, default 42) — shuffle / oversample RNG
      - max_length              (int, from sft_trainer_engine.yaml)
      - max_prompt_length       (int, optional; passed to _tokenize_chat)
      - max_response_length     (int, optional; passed to _tokenize_chat)
      - truncation              ("error"|"left"|"right", default "right")
      - strip_system            (bool, default False)
      - row_group_cache_size    (int, default 1) — LRU capacity per shard
    """

    def __init__(
        self,
        parquet_files,
        tokenizer,
        config=None,
        processor=None,
        max_samples: int = -1,
    ):
        from sft.sft_data import process

        self._process = process
        self.tokenizer = _wrap_tokenizer(tokenizer)
        self.processor = processor
        self.config = config

        data_cfg = config if config is not None else {}
        seed = int(data_cfg.get("seed", 42)) if data_cfg else 42
        self.max_length = int(data_cfg.get("max_length", 0)) if data_cfg else 0
        self.truncation = data_cfg.get("truncation", "right") if data_cfg else "right"
        cache_size = int(data_cfg.get("row_group_cache_size", 1)) if data_cfg else 1

        file_ratios = parse_files(parquet_files)
        rng = np.random.default_rng(seed)

        # Metadata-only pass: cheap even for hundreds of shards.
        self._shards: list[_ShardIndex] = [_ShardIndex(p, cache_size=cache_size) for p, _ in file_ratios]
        ratios = [r for _, r in file_ratios]

        index_map: list[tuple[int, int]] = []
        for src_idx, shard in enumerate(self._shards):
            ratio = ratios[src_idx]
            n_rows = shard.n_rows
            n = int(n_rows * ratio)
            replace = ratio > 1.0
            idxs = rng.choice(n_rows, size=n, replace=replace)
            index_map.extend((src_idx, int(i)) for i in idxs)

        perm = rng.permutation(len(index_map))
        self._index_map: list[tuple[int, int]] = [index_map[i] for i in perm]

        if max_samples is not None and max_samples > 0 and max_samples < len(self._index_map):
            self._index_map = self._index_map[:max_samples]

        print(f"[OdysSimSFTDataset] shards={len(self._shards)} samples={len(self._index_map)}")

    def __len__(self):
        return len(self._index_map)

    def _get_row(self, idx: int) -> dict:
        src_idx, row_idx = self._index_map[idx]
        return self._shards[src_idx].get_row(row_idx)

    def _truncate(self, input_ids: torch.Tensor, loss_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.max_length <= 0 or input_ids.shape[0] <= self.max_length:
            return input_ids, loss_mask
        if self.truncation == "error":
            raise ValueError(f"sequence length {input_ids.shape[0]} exceeds max_length {self.max_length}")
        if self.truncation == "left":
            return input_ids[-self.max_length :], loss_mask[-self.max_length :]
        return input_ids[: self.max_length], loss_mask[: self.max_length]

    def __getitem__(self, idx: int) -> dict:
        row = self._get_row(idx)
        d = self._process(row, self.tokenizer, config=self.config)

        input_ids = d["input_ids"].to(dtype=torch.long)
        responses = d["responses"].to(dtype=torch.long)
        response_mask = d["response_mask"].to(dtype=torch.long)

        seq_len = input_ids.shape[0]
        resp_len = responses.shape[0]
        prompt_len = seq_len - resp_len
        loss_mask = torch.zeros(seq_len, dtype=torch.long)
        loss_mask[prompt_len:] = response_mask

        input_ids, loss_mask = self._truncate(input_ids, loss_mask)
        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
