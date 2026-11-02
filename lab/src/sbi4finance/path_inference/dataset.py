"""Bounded-memory offline path training from immutable native bank shards.

Only a first-N training subset is selected. Every epoch visits each selected
trajectory once, in shuffled shard order and shuffled within-shard row order.
Batch access/prefixes are deterministic in (seed, epoch, batch index), including
framework shape probes. No simulation occurs during fitting.
"""

from collections import OrderedDict
import hashlib
from pathlib import Path

import keras
import numpy as np

from . import bank, inference


def training_count(manifest, limit=None):
    available = manifest["split_counts"]["train"]
    count = available if limit is None else inference.integer(limit, "training limit")
    if not 1 <= count <= available:
        raise ValueError("training limit must be positive and no larger than the bank's train split")
    return count


def planned_optimizer_steps(count, batch_size, epochs):
    count = inference.integer(count, "training count")
    batch_size = inference.integer(batch_size, "batch size")
    return ((count + batch_size - 1) // batch_size) * inference.integer(epochs, "epochs")


def subset_identity(root, manifest, count):
    """Bind first-N IDs and their source shards without loading observations."""
    count = training_count(manifest, count)
    records = []
    for record in manifest["splits"]["train"]:
        if record["start"] >= count:
            break
        records.append({"path": record["path"], "sha256": record["sha256"], "start": record["start"],
                        "selected_count": min(record["count"], count - record["start"])})
    prefix = f"{manifest['task_sha256'][:16]}:{manifest['root_seed']}:train:"
    digest = hashlib.sha256()
    for index in range(count):
        digest.update(f"{prefix}{index}\n".encode())
    identity = {"kind": "canonical_training_prefix_v1", "selection": "first_N_independent_train_trajectories",
                "count": count, "index_start": 0, "index_stop_exclusive": count,
                "task_sha256": manifest["task_sha256"], "root_seed": manifest["root_seed"],
                "bank_manifest_sha256": inference.sha256(Path(root) / "manifest.json"),
                "trajectory_id_encoding": "ascending canonical UTF-8 IDs, each followed by LF",
                "trajectory_id_sha256": digest.hexdigest(), "source_shards": records}
    identity["subset_sha256"] = inference.fingerprint(identity)
    return identity


def streaming_memory_projection(manifest, count, batch_size, validation_cases, lengths, cache_shards=2):
    count = training_count(manifest, count)
    cache_shards = inference.integer(cache_shards, "cache shards")
    task = manifest["task"]
    steps, targets = task["n_observations"], len(task["parameter_names"])
    selected = [r for r in manifest["splits"]["train"] if r["start"] < count]
    # UTF-32 canonical ID storage is included alongside float64 native arrays.
    longest_id = len(f"{manifest['task_sha256'][:16]}:{manifest['root_seed']}:validation:{max(count, validation_cases)}")
    raw_row_bytes = 8 * (steps + targets) + 4 * longest_id
    shard_bytes = max(r["count"] for r in selected) * raw_row_bytes
    neural_row_bytes = steps * 9 + (targets + 2) * 4
    valid_count = min(validation_cases, manifest["split_counts"]["validation"])
    validation_bytes = valid_count * len(lengths) * neural_row_bytes
    batch_bytes = batch_size * (raw_row_bytes + neural_row_bytes)
    order_bytes = count * 8
    # Include one transient loaded shard, full batch workspaces, validation
    # concatenation copies and a fixed host runtime allowance. No bank-wide
    # float64 or token array is part of this projection.
    peak = ((cache_shards + 1) * shard_bytes + 16 * batch_bytes + 4 * validation_bytes
            + 2 * valid_count * raw_row_bytes + 4 * order_bytes + 1024 ** 3)
    return {"mode": "streaming_sharded_offline", "training_input_bytes": batch_size * neural_row_bytes,
            "raw_training_arrays_bytes": count * raw_row_bytes, "training_order_bytes": order_bytes,
            "max_native_shard_bytes": shard_bytes, "cache_shards": cache_shards,
            "cache_budget_bytes": cache_shards * shard_bytes, "validation_input_bytes": validation_bytes,
            "estimated_peak_host_bytes": peak, "host_guard_bytes": 12 * 1024 ** 3,
            "estimate_note": "bounded native shard cache plus batch tokens; no whole-bank arrays; GPU/compilation memory requires the separate production-shape readiness check"}


class ShardReader:
    """LRU native shards, checked on first read and guarded by file stat after it."""

    def __init__(self, root, manifest=None, *, cache_shards=2):
        self.root = Path(root)
        self.manifest = bank.load_manifest(root) if manifest is None else manifest
        self.cache_shards = inference.integer(cache_shards, "cache shards")
        self.cache = OrderedDict()
        self.verified = {}
        self.loads = self.hash_verifications = self.cache_hits = 0
        self.peak_cache_shards = self.peak_cache_bytes = 0

    @staticmethod
    def _signature(path):
        status = path.stat()
        return status.st_size, status.st_mtime_ns, status.st_ino

    def load(self, split, record):
        if split not in ("train", "validation"):
            raise ValueError("offline training reader never opens the reserved test split")
        key = record["path"]
        path = self.root / key
        signature = self._signature(path)
        if key in self.verified and signature != self.verified[key]:
            raise ValueError(f"{key}: immutable shard changed after verification")
        if key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        while len(self.cache) >= self.cache_shards:
            self.cache.popitem(last=False)
        if key not in self.verified:
            arrays = bank._load_part(self.root, self.manifest, split, record)
            self.hash_verifications += 1
            if self._signature(path) != signature:
                raise ValueError(f"{key}: shard changed while being verified")
            self.verified[key] = signature
        else:
            with np.load(path, allow_pickle=False) as saved:
                arrays = dict(saved)
            if self._signature(path) != signature:
                raise ValueError(f"{key}: shard changed while being read")
        self.loads += 1
        self.cache[key] = arrays
        self.peak_cache_shards = max(self.peak_cache_shards, len(self.cache))
        self.peak_cache_bytes = max(self.peak_cache_bytes,
                                    sum(value.nbytes for a in self.cache.values() for value in a.values()))
        return arrays

    def statistics(self):
        return {"shard_loads": self.loads, "hash_verifications": self.hash_verifications,
                "cache_hits": self.cache_hits, "verified_shards": len(self.verified),
                "cache_shards_limit": self.cache_shards, "peak_cache_shards": self.peak_cache_shards,
                "peak_cache_array_bytes": self.peak_cache_bytes,
                "integrity": "SHA-256 and native schema/IDs verified once per shard; file size/mtime/inode checked on every later access"}


def load_subset(root, split, count, *, manifest=None):
    """Read only the first requested rows and required shards, never a full split."""
    manifest = bank.load_manifest(root) if manifest is None else manifest
    if split not in ("train", "validation"):
        raise ValueError("streamed offline training never opens the reserved test split")
    count = min(inference.integer(count, "subset count"), manifest["split_counts"][split])
    if not count:
        raise ValueError(f"{split} split is empty")
    reader = ShardReader(root, manifest, cache_shards=1)
    pieces = {"observations": [], "parameters": [], "trajectory_id": []}
    for record in manifest["splits"][split]:
        if record["start"] >= count:
            break
        arrays = reader.load(split, record)
        take = min(record["count"], count - record["start"])
        for key in pieces:
            pieces[key].append(arrays[key][:take].copy())
    return {**{key: np.concatenate(value) for key, value in pieces.items()}, "task": manifest["task"]}


class ShardedPathDataset(keras.utils.PyDataset):
    """Batched observed returns/labels; only the supplied adapter routes labels."""

    def __init__(self, root, *, adapter, batch_size=128, training_limit=None,
                 seed=0, prefix_policy="uniform", cache_shards=2):
        # No worker processes: they would copy caches and prefix state. Framework
        # signature probes can request batches repeatedly without changing them.
        super().__init__(workers=1, use_multiprocessing=False, max_queue_size=1)
        self.root = Path(root)
        self.manifest = bank.load_manifest(root)
        self.task = self.manifest["task"]
        self.num_samples = training_count(self.manifest, training_limit)
        self.batch_size = inference.integer(batch_size, "batch size")
        self.seed = inference.integer(seed, "dataset seed", minimum=0)
        if prefix_policy not in inference.PREFIX_POLICIES:
            raise ValueError("unsupported prefix policy")
        self.prefix_policy, self.adapter = prefix_policy, adapter
        self.records = [r for r in self.manifest["splits"]["train"] if r["start"] < self.num_samples]
        self.ends = np.array([r["start"] + r["count"] for r in self.records], dtype=np.int64)
        self.reader = ShardReader(root, self.manifest, cache_shards=cache_shards)
        self.epoch = 0
        self.exposure = np.zeros(self.task["n_observations"] + 1, dtype=np.int64)
        self.requested_batches = 0
        self._make_order()

    def _rng(self, domain, index=0):
        return np.random.default_rng(np.random.SeedSequence(
            [self.seed & 0xffffffff, self.seed >> 32, self.epoch, domain, index]))

    def _make_order(self):
        order = np.empty(self.num_samples, dtype=np.int64)
        position = 0
        for index in self._rng(0).permutation(len(self.records)):
            record = self.records[int(index)]
            count = min(record["count"], self.num_samples - record["start"])
            rows = self._rng(1, int(index)).permutation(count) + record["start"]
            order[position:position + count] = rows
            position += count
        self.order = order

    @property
    def num_batches(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __len__(self):
        return self.num_batches

    def batch_indices(self, item):
        if item < 0:
            item += self.num_batches
        if not 0 <= item < self.num_batches:
            raise IndexError("streamed path batch index out of range")
        return self.order[item * self.batch_size:min((item + 1) * self.batch_size, self.num_samples)].copy()

    def raw_batch(self, indices):
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices) or np.any(indices < 0) or np.any(indices >= self.num_samples):
            raise ValueError("selected training indices must lie inside first-N subset")
        shape = (len(indices), self.task["n_observations"], 1)
        result = {"observations": np.empty(shape, dtype=np.float64),
                  "parameters": np.empty((len(indices), len(self.task["parameter_names"])), dtype=np.float64)}
        shard_ids = np.searchsorted(self.ends, indices, side="right")
        for shard_id in np.unique(shard_ids):
            selected = np.flatnonzero(shard_ids == shard_id)
            record = self.records[int(shard_id)]
            values = self.reader.load("train", record)
            local = indices[selected] - record["start"]
            for key in result:
                result[key][selected] = values[key][local]
        return result

    def __getitem__(self, item):
        if item < 0:
            item += self.num_batches
        indices = self.batch_indices(item)
        lengths = inference.draw_prefix_lengths(self.task, len(indices), self._rng(2, int(item)),
                                               policy=self.prefix_policy)
        batch = inference.training_data(self.raw_batch(indices), self.task, lengths=lengths)
        self.exposure += np.bincount(lengths, minlength=len(self.exposure))
        self.requested_batches += 1
        return self.adapter(batch) if self.adapter is not None else batch

    def on_epoch_end(self):
        self.epoch += 1
        self._make_order()

    def statistics(self):
        return {**self.reader.statistics(), "dataset_epoch_counter": self.epoch,
                "requested_batches_including_framework_probes": self.requested_batches,
                "order_bytes": self.order.nbytes,
                "shuffle": "deterministic shuffled shards and shuffled rows within each shard per epoch; framework batch shuffle disabled",
                "prefix_seed_rule": "SeedSequence(dataset_seed_low32,high32,epoch,domain=2,batch_index)"}
