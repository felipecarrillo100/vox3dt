"""Bounded-memory external sort/merge for voxel aggregates.

The rest of the pipeline works with (key, colour_sum, count) triples: `key` is
a packed grid index (see `voxelize.brick_key`), `colour_sum` a per-channel
integer numerator, `count` a divisor (point count, or downsample weight one
level up). `RunSpiller` accepts a stream of such triples and keeps at most
`budget_bytes` of them combined in RAM at a time, spilling the rest to sorted,
unique-by-key run files under a temp directory. `merge_stream` reads any
number of those run files back as one globally sorted, globally unique stream,
without requiring more than a small, constant number of them open and buffered
at once.

Peak RAM is therefore independent of how many total records were spilled --
only of `budget_bytes` and the merge fan-in/buffer size below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: One aggregate record: packed key, per-channel colour-numerator sum, divisor.
AGG = np.dtype([("key", "<i8"), ("sum", "<i8", (3,)), ("cnt", "<i8")])

_MERGE_EVERY = 8  # matches the original in-RAM implementation's batch size
K_MAX = 32  # max run files merged in one pass before recursing
RUN_BUFFER_RECORDS = 131_072


def _pack(keys: np.ndarray, sums: np.ndarray, cnts: np.ndarray) -> np.ndarray:
    out = np.empty(len(keys), dtype=AGG)
    out["key"] = keys
    out["sum"] = sums
    out["cnt"] = cnts
    return out


def combine(parts: list[np.ndarray]) -> np.ndarray:
    """Merge a list of AGG arrays into one sorted array, unique by key.

    Always re-sorts and re-dedupes, even for a single input array: a lone
    part is not guaranteed to already be sorted (e.g. a `RunSpiller` that
    received only one `.add()` call before `finish()` holds a single raw,
    unsorted part).
    """
    cat = np.concatenate(parts) if len(parts) > 1 else parts[0]
    order = np.argsort(cat["key"], kind="stable")
    cat = cat[order]
    uniq_keys, start = np.unique(cat["key"], return_index=True)
    out = np.empty(len(uniq_keys), dtype=AGG)
    out["key"] = uniq_keys
    out["sum"] = np.add.reduceat(cat["sum"], start, axis=0)
    out["cnt"] = np.add.reduceat(cat["cnt"], start)
    return out


@dataclass
class Spilled:
    """Either an in-RAM AGG array, or a list of sorted/unique run files on disk."""

    ram: np.ndarray | None
    runs: list[Path] = field(default_factory=list)
    n_records: int = 0
    denom: int = 1

    def is_ram(self) -> bool:
        return self.ram is not None


class RunSpiller:
    """Accumulates AGG records, spilling sorted/unique runs once a size budget is hit."""

    def __init__(self, temp_dir: Path, budget_bytes: int, denom: int = 1):
        self.temp_dir = temp_dir
        self.budget_bytes = budget_bytes
        self.denom = denom
        self._parts: list[np.ndarray] = []
        self._run_paths: list[Path] = []
        self._n_run = 0
        self._n_records = 0

    def add(self, keys: np.ndarray, sums: np.ndarray, cnts: np.ndarray) -> None:
        if len(keys) == 0:
            return
        self._parts.append(_pack(keys, sums, cnts))
        if len(self._parts) >= _MERGE_EVERY:
            self._parts = [combine(self._parts)]
        self._maybe_spill()

    def _maybe_spill(self) -> None:
        """Check pending size against budget after *every* add, not just the
        every-8-calls collapse above -- a spiller fed few, large `.add()`
        calls (e.g. one huge chunk, or one call per tile at a coarse pyramid
        level) must still be checked, or `--memory-budget` silently does
        nothing for it."""
        if not self._parts:
            return
        total = sum(p.nbytes for p in self._parts)
        if total >= self.budget_bytes:
            self._write_run(combine(self._parts))
            self._parts = []

    def _write_run(self, arr: np.ndarray) -> None:
        if len(arr) == 0:
            return
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        path = self.temp_dir / f"run_{self._n_run:05d}.bin"
        arr.tofile(path)
        self._run_paths.append(path)
        self._n_run += 1
        self._n_records += len(arr)

    def finish(self) -> Spilled:
        combined = combine(self._parts) if self._parts else np.empty(0, dtype=AGG)
        if not self._run_paths:
            return Spilled(ram=combined, n_records=len(combined), denom=self.denom)
        if len(combined):
            self._write_run(combined)
        return Spilled(ram=None, runs=list(self._run_paths), n_records=self._n_records, denom=self.denom)


def _read_block(f, n: int) -> np.ndarray:
    data = f.read(n * AGG.itemsize)
    if not data:
        return np.empty(0, dtype=AGG)
    return np.frombuffer(data, dtype=AGG)


def _merge_runs(run_paths: list[Path], buf_records: int):
    """k-way vectorized merge over `run_paths` (each sorted, unique by key)."""
    if not run_paths:
        return
    files = [open(p, "rb") for p in run_paths]
    try:
        buffers = [_read_block(f, buf_records) for f in files]
        live = [i for i in range(len(files)) if len(buffers[i])]
        while live:
            if len(live) == 1:
                i = live[0]
                while len(buffers[i]):
                    yield buffers[i]
                    buffers[i] = _read_block(files[i], buf_records)
                break
            horizon = min(int(buffers[i]["key"][-1]) for i in live)
            takes = []
            new_live = []
            for i in live:
                n = int(np.searchsorted(buffers[i]["key"], horizon, side="right"))
                if n:
                    takes.append(buffers[i][:n])
                remainder = buffers[i][n:]
                if len(remainder) == 0:
                    remainder = _read_block(files[i], buf_records)
                buffers[i] = remainder
                if len(buffers[i]):
                    new_live.append(i)
            live = new_live
            if takes:
                yield combine(takes) if len(takes) > 1 else takes[0]
    finally:
        for f in files:
            f.close()


def _write_merged(run_paths: list[Path], out_path: Path, buf_records: int) -> None:
    with open(out_path, "wb") as out:
        for block in _merge_runs(run_paths, buf_records):
            block.tofile(out)


def _merge_multi_pass(run_paths: list[Path], temp_dir: Path, k_max: int, buf_records: int):
    pass_no = 0
    while len(run_paths) > k_max:
        new_paths = []
        for gi in range(0, len(run_paths), k_max):
            group = run_paths[gi : gi + k_max]
            out_path = temp_dir / f"m{pass_no}_{gi // k_max:05d}.bin"
            _write_merged(group, out_path, buf_records)
            new_paths.append(out_path)
        for p in run_paths:
            p.unlink(missing_ok=True)
        run_paths = new_paths
        pass_no += 1
    yield from _merge_runs(run_paths, buf_records)


def merge_stream(spilled: Spilled, temp_dir: Path, k_max: int = K_MAX, buf_records: int = RUN_BUFFER_RECORDS):
    """Yield sorted, globally-unique AGG blocks covering all of `spilled`."""
    if spilled.is_ram():
        arr = spilled.ram
        for i in range(0, len(arr), buf_records):
            yield arr[i : i + buf_records]
        return
    if len(spilled.runs) > k_max:
        yield from _merge_multi_pass(list(spilled.runs), temp_dir, k_max, buf_records)
    else:
        yield from _merge_runs(spilled.runs, buf_records)
