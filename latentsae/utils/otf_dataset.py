"""On-the-fly ColBERT → SAE dataset.

Streams chunk text from parquet, embeds with a pylate ColBERT model, and
yields batched token vectors ready for an SAE trainer to consume. A
background thread runs the ColBERT forward asynchronously so the main
training loop never blocks waiting on data.

Usage:
    ds = OnTheFlyColBERTDataset(
        parquet_dirs=[
            "/data/chunks/fineweb-edu-sample-10BT-chunked-500/train",
            "/data/chunks/RedPajama-Data-V2-sample-10B-chunked-500/train",
            "/data/chunks/pile-uncopyrighted-chunked-500/train",
        ],
        model_id="mixedbread-ai/mxbai-edge-colbert-v0-32m",
        batch_size=8192,
        chunks_per_encode=64,
        device="cuda",
    )
    for batch in ds.take(200_000_000):      # tokens, not chunks
        out = sae(batch)
        ...

Design notes
------------
- Double-buffering is implicit through a bounded queue: the producer
  encodes ColBERT batches and `put()`s them; the consumer `get()`s and
  re-slices into SAE-sized batches. ColBERT is ~2x faster than the SAE
  on GSV so the queue stays saturated, and SAE never waits.
- Parquets across multiple domains are interleaved chunk-by-chunk with
  a seeded shuffle. No disk lives needed past the parquet text, and the
  ColBERT output never touches disk.
- The producer holds the ColBERT model on the same CUDA device as the
  SAE. Small mxbai-32m (~64 MB fp16) so the share is fine.
- Cleanup: call `stop()` explicitly, or let `__del__` do it. The worker
  thread is a daemon so a hard exit doesn't leak.
"""
from __future__ import annotations

import itertools
import queue
import random
import threading
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset


class OnTheFlyColBERTDataset(IterableDataset):
    def __init__(
        self,
        parquet_dirs: list[str | Path],
        model_id: str = "mixedbread-ai/mxbai-edge-colbert-v0-32m",
        batch_size: int = 8192,
        chunks_per_encode: int = 64,
        device: str = "cuda",
        queue_size: int = 4,
        seed: int = 42,
        encode_max_length: int = 512,
        domain_weights: Optional[list[float]] = None,
        shuffle_buffer_size: int = 0,
        replay_factor: float = 1.0,
        on_device: bool = False,
    ):
        # Group parquets by domain (one domain per parquet_dir). Sampling
        # picks a domain by `domain_weights`, then a random parquet within
        # it. Default: uniform over domains (each gets equal chunks).
        self.domain_paths: list[list[Path]] = []
        for d in parquet_dirs:
            p = Path(d)
            files = sorted(p.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"no parquets under {p}")
            self.domain_paths.append(files)
        if domain_weights is None:
            self.domain_weights = [1.0] * len(self.domain_paths)
        else:
            if len(domain_weights) != len(self.domain_paths):
                raise ValueError(
                    f"domain_weights length {len(domain_weights)} must match "
                    f"parquet_dirs length {len(self.domain_paths)}")
            self.domain_weights = [float(w) for w in domain_weights]
        total_w = sum(self.domain_weights)
        self.domain_probs = [w / total_w for w in self.domain_weights]
        # Flat list of all paths (for bookkeeping / display only)
        self.parquet_paths: list[Path] = [p for ps in self.domain_paths for p in ps]
        self.model_id = model_id
        self.batch_size = batch_size
        self.chunks_per_encode = chunks_per_encode
        self.device = device
        self.queue_size = queue_size
        self.seed = seed
        self.encode_max_length = encode_max_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.replay_factor = float(replay_factor)
        # When True, encode path keeps tensors on GPU throughout:
        #   producer: model.encode(convert_to_numpy=False) → fp16 GPU tensor
        #   queue: fp16 CUDA tensor (thread-safe across Python threads)
        #   consumer: accumulate in a GPU-resident shuffle buffer, sample in-place,
        #             yield fp32 CUDA tensor (zero CPU↔GPU roundtrip per batch).
        self.on_device = bool(on_device)
        # Lazy producer state (CUDA-init on first iter so main-thread CUDA
        # context isn't initialized in __init__ before DataLoader fork).
        self._q: Optional[queue.Queue] = None
        self._stop: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._tokens_produced = 0
        self._start_time: Optional[float] = None

    # ---- Producer ----

    def _start_producer(self):
        # Import inside so workers don't trigger pylate/torch at import time
        from pylate import models

        self._q = queue.Queue(maxsize=self.queue_size)
        self._stop = threading.Event()
        self._tokens_produced = 0
        self._start_time = time.monotonic()

        def worker():
            try:
                model = models.ColBERT(model_name_or_path=self.model_id,
                                       device=self.device)
                rng = random.Random(self.seed)
                domain_indices = list(range(len(self.domain_paths)))
                # Per-domain parquet-file queue (shuffled, refilled on exhaustion).
                per_domain_parquet_queue = [[] for _ in self.domain_paths]
                # Per-domain in-memory buffer of shuffled text chunks +
                # current read cursor. Keeping one open per domain so each
                # encode-batch can pull from a freshly-picked domain rather
                # than draining an entire parquet before switching.
                domain_buf: list[list[str]] = [[] for _ in self.domain_paths]
                domain_pos: list[int] = [0] * len(self.domain_paths)

                def next_path_for_domain(di: int) -> Path:
                    if not per_domain_parquet_queue[di]:
                        per_domain_parquet_queue[di] = list(self.domain_paths[di])
                        rng.shuffle(per_domain_parquet_queue[di])
                    return per_domain_parquet_queue[di].pop()

                def refill_domain(di: int) -> bool:
                    path = next_path_for_domain(di)
                    try:
                        df = pd.read_parquet(path, columns=["chunk_text"])
                    except Exception as exc:
                        print(f"[otf] parquet read error {path}: {exc}")
                        return False
                    texts = df["chunk_text"].tolist()
                    rng.shuffle(texts)
                    domain_buf[di] = texts
                    domain_pos[di] = 0
                    return True

                while not self._stop.is_set():
                    di = rng.choices(domain_indices, weights=self.domain_probs, k=1)[0]
                    # Ensure the chosen domain has at least one full encode
                    # batch remaining; refill from the next parquet otherwise.
                    while len(domain_buf[di]) - domain_pos[di] < self.chunks_per_encode:
                        if not refill_domain(di):
                            break
                    if self._stop.is_set():
                        break
                    start = domain_pos[di]
                    batch = domain_buf[di][start:start + self.chunks_per_encode]
                    domain_pos[di] = start + self.chunks_per_encode
                    batch = [t if t and t.strip() else " " for t in batch]
                    try:
                        emb_list = model.encode(
                            batch,
                            batch_size=self.chunks_per_encode,
                            show_progress_bar=False,
                            is_query=False,
                            convert_to_numpy=not self.on_device,
                        )
                    except Exception as exc:
                        print(f"[otf] encode error: {exc}")
                        continue
                    # CPU path: emb_list is a list of per-chunk fp32 numpy
                    # arrays. Cast to fp16 and concatenate into one numpy array
                    # before enqueueing.
                    # GPU path: emb_list is a list of fp32 CUDA tensors. Cast
                    # to fp16 and concatenate on-device; enqueue a CUDA tensor.
                    if self.on_device:
                        if not emb_list:
                            continue
                        # Each element is a 2D fp32 CUDA tensor (n_tokens_i, 64)
                        flat = torch.cat([e.to(dtype=torch.float16) for e in emb_list], dim=0)
                        self._tokens_produced += int(flat.shape[0])
                        self._q.put(flat)
                    else:
                        arrs = []
                        for e in emb_list:
                            if hasattr(e, "cpu"):
                                e = e.cpu().numpy()
                            arrs.append(np.asarray(e, dtype=np.float16))
                        if not arrs:
                            continue
                        flat = np.concatenate(arrs, axis=0)
                        self._tokens_produced += flat.shape[0]
                        self._q.put(flat)
            except Exception as exc:
                # Signal shutdown to consumer so it doesn't hang forever
                import traceback
                print(f"[otf] producer crashed: {exc}")
                traceback.print_exc()
                self._stop.set()
                self._q.put(None)  # poison pill

        self._thread = threading.Thread(target=worker, daemon=True,
                                        name="otf-colbert-producer")
        self._thread.start()

    # ---- Consumer ----

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self._thread is None:
            self._start_producer()
        if self.shuffle_buffer_size > 0:
            yield from self._iter_shuffled()
        else:
            yield from self._iter_fifo()

    def _iter_fifo(self) -> Iterator[torch.Tensor]:
        """Original consumer — concatenate queue entries in FIFO, slice
        to batch_size. Each batch is tokens from ~batch_size/tokens_per_chunk
        consecutive chunks (~22 at cpe=64) — small per-batch chunk diversity.

        on_device=True variant: queue entries are CUDA fp16 tensors; we
        concatenate, slice, and cast in-place on GPU — no CPU hop.
        """
        if self.on_device:
            carry = torch.empty((0, 64), dtype=torch.float16, device=self.device)
            while not (self._stop and self._stop.is_set()):
                while carry.shape[0] < self.batch_size:
                    chunk = self._q.get()
                    if chunk is None:
                        return
                    carry = torch.cat([carry, chunk], dim=0) if carry.shape[0] > 0 else chunk
                batch = carry[: self.batch_size]
                carry = carry[self.batch_size:]
                yield batch.to(dtype=torch.float32)
            return
        carry = np.empty((0, 64), dtype=np.float16)
        while not (self._stop and self._stop.is_set()):
            while carry.shape[0] < self.batch_size:
                chunk = self._q.get()
                if chunk is None:
                    return
                carry = np.concatenate([carry, chunk], axis=0) if carry.shape[0] > 0 else chunk
            batch = carry[: self.batch_size]
            carry = carry[self.batch_size:]
            yield torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32))

    def _iter_shuffled(self) -> Iterator[torch.Tensor]:
        """Reservoir-style shuffle buffer.

        Maintain a fixed-capacity buffer of token vectors. New encode
        batches append (filling phase) or evict random rows (steady
        state). Each SAE batch = random sample from buffer — mixes
        tokens from thousands of chunks per batch instead of ~22,
        approaching disk-backed behavior.

        Yield rate matches producer rate: the consumer yields floor(
        new_tokens_since_last_yield / batch_size) batches per iteration.
        A token's expected number of samplings is
        (buffer_size / producer_rate) × (batch_size / buffer_size) ×
        (consumer_rate) ≈ 1 when consumer matches producer — same
        regime as disk-backed (each token seen ~once).
        """
        buf_size = int(self.shuffle_buffer_size)
        if self.on_device:
            # GPU-resident reservoir buffer. ~500 MB VRAM at buf_size=4M.
            buf = torch.empty((buf_size, 64), dtype=torch.float16, device=self.device)
            fill = 0
            pending = 0
            gen = torch.Generator(device=self.device); gen.manual_seed(self.seed + 1)
            while not (self._stop and self._stop.is_set()):
                chunk = self._q.get()
                if chunk is None:
                    return
                n = int(chunk.shape[0])
                pending += n * self.replay_factor
                if fill < buf_size:
                    take = min(n, buf_size - fill)
                    buf[fill:fill + take] = chunk[:take]
                    fill += take
                    remaining = chunk[take:] if n > take else None
                else:
                    remaining = chunk
                if remaining is not None and remaining.shape[0] > 0:
                    idx = torch.randint(0, buf_size, (remaining.shape[0],),
                                        device=self.device, generator=gen)
                    buf[idx] = remaining
                while pending >= self.batch_size and fill >= self.batch_size:
                    sample_idx = torch.randint(0, fill, (self.batch_size,),
                                               device=self.device, generator=gen)
                    batch = buf[sample_idx]
                    pending -= self.batch_size
                    yield batch.to(dtype=torch.float32)
            return
        buf = np.empty((buf_size, 64), dtype=np.float16)
        fill = 0
        pending = 0  # tokens added but not yet accounted for in yields
        rng = np.random.default_rng(self.seed + 1)
        while not (self._stop and self._stop.is_set()):
            chunk = self._q.get()
            if chunk is None:
                return
            n = int(chunk.shape[0])
            # replay_factor > 1 multiplies the "pending yield budget" so
            # each pulled encode batch produces >1 SAE batches — every
            # token ends up sampled ~replay_factor times on average.
            pending += n * self.replay_factor
            # Filling phase: linearly append to end of buffer
            if fill < buf_size:
                take = min(n, buf_size - fill)
                buf[fill:fill + take] = chunk[:take]
                fill += take
                remaining = chunk[take:] if n > take else None
            else:
                remaining = chunk
            # Steady-state eviction: overwrite random positions with leftover rows
            if remaining is not None and remaining.shape[0] > 0:
                idx = rng.integers(0, buf_size, size=remaining.shape[0])
                buf[idx] = remaining
            # Emit as many SAE batches as the producer has "paid for"
            # in new tokens. Keeps consumer rate ≈ producer rate so each
            # token is expected to be sampled ~1× while it lives in the
            # buffer (matching disk-backed single-epoch semantics).
            while pending >= self.batch_size and fill >= self.batch_size:
                sample_idx = rng.integers(0, fill, size=self.batch_size)
                batch = buf[sample_idx]
                pending -= self.batch_size
                yield torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32))

    def take(self, n_tokens: int) -> Iterator[torch.Tensor]:
        """Convenience: yield batches until ~n_tokens have been produced."""
        n_batches = (n_tokens + self.batch_size - 1) // self.batch_size
        it = iter(self)
        for i, batch in enumerate(it):
            if i >= n_batches:
                break
            yield batch

    def stats(self) -> dict:
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        return {
            "tokens_produced": self._tokens_produced,
            "queue_depth": self._q.qsize() if self._q else 0,
            "tokens_per_s": self._tokens_produced / elapsed if elapsed > 0 else 0,
            "elapsed_s": elapsed,
        }

    def stop(self):
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
