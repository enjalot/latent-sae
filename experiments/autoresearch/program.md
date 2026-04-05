# SAE Autoresearch Program

You are an autonomous ML researcher improving a Sparse Autoencoder (SAE) for sentence embeddings. You will run experiments in a loop, each completing in under 5 minutes.

## The Setup

- `train.py` — **YOU EDIT THIS.** It contains SAE_CONFIG, TRAIN_CONFIG, and the training procedure.
- `prepare.py` — **DO NOT EDIT.** It contains the locked evaluation suite.
- `../../latentsae/sae.py` — **YOU CAN EDIT THIS** for architecture changes (new encode methods, etc).
- `../../latentsae/utils/config.py` — **YOU CAN EDIT THIS** to add new config fields.
- `../../latentsae/trainer.py` — **YOU CAN EDIT THIS** for training loop changes.

## The Metric

Composite score (higher = better):
```
score = CLINC150_accuracy * 0.4 + SciFact_nDCG@10 * 0.3 + (1 - MMCS) * 0.3
```

- CLINC150: 150-class intent classification via logistic regression on sparse features
- SciFact: retrieval quality via cosine similarity on sparse features
- MMCS: decoder weight redundancy (lower = more distinct features)
- Also reported: FVU, active features, gaps to raw embeddings

Current best (4x k=128): **composite = 0.747** (CLINC150=0.796, SciFact=0.621, MMCS=0.193)

## The Loop

```
LOOP FOREVER:
1. Look at git log and results.tsv to understand what's been tried
2. Come up with ONE specific idea to test
3. Edit train.py (and optionally sae.py/config.py/trainer.py)
4. git add -A && git commit -m "description of change"
5. Run: python experiments/autoresearch/train.py \
     --data-dir /embeddings/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train \
     --d-in 384 --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
     --device mps > run.log 2>&1
6. Read results: grep "^composite_score:\|^clinc150_sparse:\|^scifact_sparse:\|^mmcs:\|^fvu:" run.log
7. If grep is empty → crashed. Read: tail -50 run.log. Fix and retry.
8. Append to results.tsv: commit, composite_score, clinc150, scifact, mmcs, fvu, status, description
9. If composite_score improved → KEEP (advance branch)
10. If composite_score same or worse → DISCARD (git reset --hard HEAD~1)
```

## What to Try

Ideas roughly ordered by expected impact:

### Architecture
- [ ] Try k=256 at 4x expansion (more active features per sample)
- [ ] Try 2x expansion with k=128 (even smaller, more distinct features)
- [ ] Learnable per-feature bias in the encoder (shift activations)
- [ ] BatchTopK: select top-k across the batch, not per-sample
- [ ] Matryoshka-style multi-scale loss (train at multiple k simultaneously)

### Training
- [ ] Learning rate sweep (current is auto-scaled, try 2x and 0.5x)
- [ ] Tilted ERM (tilt=0.002, upweight hard samples)
- [ ] Longer warmup (1000 steps vs current 200)
- [ ] Weight decay on encoder/decoder (currently 0)
- [ ] Initialize decoder as PCA of training data (instead of random)

### Loss
- [ ] Add cosine similarity loss between input and reconstruction
- [ ] Weighted reconstruction: upweight rare activation patterns
- [ ] Feature orthogonality regularizer (penalize high MMCS directly)

### Data
- [ ] Increase to 5M samples (still fits in 5 min)
- [ ] Sample harder examples more often (online hard mining)

## Rules

1. **ONE idea per experiment.** Combine ideas only after both individually help.
2. **Never stop.** Don't ask for permission. Run the next experiment.
3. **Log failures.** A crash or regression is still data — record it in results.tsv.
4. **Simplicity wins.** A 0.001 improvement from 20 lines of code? Not worth it. A 0.001 improvement from deleting code? Keep it.
5. **Read prior results.** Don't repeat failed experiments.
6. **Time budget is hard.** If training takes >4 minutes, reduce N_SAMPLES or model size.

## For local (M2 Mac) runs

```bash
# Generate synthetic data if no Modal volumes
python -c "
import numpy as np, os
os.makedirs('experiments/autoresearch/.data/train', exist_ok=True)
for i in range(20):
    d = np.random.randn(100000, 384).astype('float32')
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    np.save(f'experiments/autoresearch/.data/train/shard-{i:03d}.npy', d)
"

# Run with local data
python experiments/autoresearch/train.py \
  --data-dir experiments/autoresearch/.data/train \
  --d-in 384 --device mps > run.log 2>&1
```

## For Modal runs

```bash
modal run train_modal.py --config experiments/autoresearch/config.yaml --gpu-type a10g
```
