"""Multi-producer OTF probe.

Tests whether running N parallel ColBERT producer threads (each with its
own model instance) beats the single-thread throughput ceiling.
"""
import argparse
import queue
import random
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class MultiProducerOTF:
    def __init__(self, parquet_dirs, batch_size=8192, chunks_per_encode=128,
                 n_producers=2, queue_size=8, device="cuda",
                 model_id="mixedbread-ai/mxbai-edge-colbert-v0-32m",
                 seed=42):
        self.domain_paths = []
        for d in parquet_dirs:
            paths = sorted(Path(d).glob("*.parquet"))
            self.domain_paths.append(paths)
        self.batch_size = batch_size
        self.chunks_per_encode = chunks_per_encode
        self.n_producers = n_producers
        self.queue_size = queue_size
        self.device = device
        self.model_id = model_id
        self.seed = seed
        self._q = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._tokens = 0
        self._tok_lock = threading.Lock()
        self._threads = []
        self._start = None

    def _make_worker(self, worker_id: int):
        from pylate import models
        model = models.ColBERT(model_name_or_path=self.model_id, device=self.device)
        rng = random.Random(self.seed + worker_id * 1000)
        domain_ix = list(range(len(self.domain_paths)))
        per_domain_q = [[] for _ in self.domain_paths]

        def next_path(di):
            if not per_domain_q[di]:
                per_domain_q[di] = list(self.domain_paths[di])
                rng.shuffle(per_domain_q[di])
            return per_domain_q[di].pop()

        def run():
            while not self._stop.is_set():
                di = rng.choice(domain_ix)
                path = next_path(di)
                try:
                    df = pd.read_parquet(path, columns=["chunk_text"])
                except Exception as e:
                    print(f"[prod{worker_id}] parquet err {path}: {e}")
                    continue
                texts = df["chunk_text"].tolist()
                rng.shuffle(texts)
                for i in range(0, len(texts), self.chunks_per_encode):
                    if self._stop.is_set():
                        break
                    batch = [t if t and t.strip() else " "
                             for t in texts[i:i + self.chunks_per_encode]]
                    try:
                        emb_list = model.encode(
                            batch, batch_size=self.chunks_per_encode,
                            show_progress_bar=False, is_query=False)
                    except Exception as e:
                        print(f"[prod{worker_id}] encode err: {e}")
                        continue
                    arrs = []
                    for e in emb_list:
                        if hasattr(e, "cpu"):
                            e = e.cpu().numpy()
                        arrs.append(np.asarray(e, dtype=np.float16))
                    if not arrs:
                        continue
                    flat = np.concatenate(arrs, axis=0)
                    with self._tok_lock:
                        self._tokens += flat.shape[0]
                    self._q.put(flat)

        return run

    def start(self):
        self._start = time.monotonic()
        for i in range(self.n_producers):
            t = threading.Thread(target=self._make_worker(i), daemon=True,
                                 name=f"otf-prod-{i}")
            t.start()
            self._threads.append(t)

    def iter_batches(self):
        carry = np.empty((0, 64), dtype=np.float16)
        while not self._stop.is_set():
            while carry.shape[0] < self.batch_size:
                chunk = self._q.get()
                if chunk is None:
                    return
                carry = np.concatenate([carry, chunk], axis=0) if carry.shape[0] > 0 else chunk
            batch = carry[:self.batch_size]
            carry = carry[self.batch_size:]
            yield torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32))

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-producers", type=int, default=2)
    ap.add_argument("--chunks-per-encode", type=int, default=128)
    ap.add_argument("--queue-size", type=int, default=8)
    ap.add_argument("--duration", type=float, default=25.0)
    ap.add_argument("--warmup", type=float, default=10.0)
    args = ap.parse_args()

    parquet_dirs = [
        "/data/chunks/fineweb-edu-sample-10BT-chunked-500/train",
        "/data/chunks/RedPajama-Data-V2-sample-10B-chunked-500/train",
        "/data/chunks/pile-uncopyrighted-chunked-500/train",
    ]
    m = MultiProducerOTF(parquet_dirs,
                         chunks_per_encode=args.chunks_per_encode,
                         n_producers=args.n_producers,
                         queue_size=args.queue_size)
    m.start()
    it = m.iter_batches()

    # warmup
    t_warm = time.monotonic() + args.warmup
    n_warm = 0
    while time.monotonic() < t_warm:
        next(it); n_warm += 1
    tok_at_warm = m._tokens
    t0 = time.monotonic()
    t_end = t0 + args.duration
    n = 0
    q_depths = []
    while time.monotonic() < t_end:
        next(it); n += 1
        if n % 50 == 0:
            q_depths.append(m._q.qsize())
    wall = time.monotonic() - t0
    tok_made = m._tokens - tok_at_warm
    print(f"n_producers={args.n_producers} cpe={args.chunks_per_encode} "
          f"qs={args.queue_size} batches={n} tok/s={tok_made/wall:,.0f} "
          f"batch/s={n/wall:.2f} q_avg={sum(q_depths)/max(1,len(q_depths)):.1f} "
          f"q_max={max(q_depths) if q_depths else 0}")
    m.stop()


if __name__ == "__main__":
    main()
