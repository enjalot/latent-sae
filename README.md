# latent-sae

Train Sparse Autoencoders on sentence embedding representations. Decomposes dense embedding vectors into sparse, interpretable features.

Fork of [EleutherAI/sae](https://github.com/EleutherAI/sae), focused on sentence transformer embeddings with fast disk-based data loading, Modal GPU training, and a comprehensive experiment framework.

## Published Models

| Model | Subfolder | Embedding | Features | k | Best For |
|-------|-----------|-----------|----------|---|----------|
| [sae-all-MiniLM-L6-v2-FineWeb-RedPajama-Pile-150M](https://huggingface.co/enjalot/sae-all-MiniLM-L6-v2-FineWeb-RedPajama-Pile-150M) | `128_4` | MiniLM (384D) | 1,536 | 128 | Fine-grained classification |
| | `128_8` | MiniLM (384D) | 3,072 | 128 | Retrieval |
| | `64_8` | MiniLM (384D) | 3,072 | 64 | Maximum feature coverage |
| [sae-nomic-text-v1.5-FineWeb-edu-100BT](https://huggingface.co/enjalot/sae-nomic-text-v1.5-FineWeb-edu-100BT) | `64_32` | nomic-v1.5 (768D) | 24,576 | 64 | Legacy (taxonomy) |

## Quick Start

```python
# pip install latentsae
from latentsae import Sae
from sentence_transformers import SentenceTransformer
import torch

# Load SAE
sae = Sae.load_from_hub("enjalot/sae-all-MiniLM-L6-v2-FineWeb-RedPajama-Pile-150M", "128_4")

# Embed text
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(["Hello world", "Sparse autoencoders decompose embeddings"],
                          convert_to_tensor=True, normalize_embeddings=True)

# Extract sparse features
features = sae.encode(embeddings)
print(features.top_indices)  # which features activated
print(features.top_acts)     # how strongly
```

## Training

### With YAML config (recommended)

```bash
# Single run on Modal A10G
modal run train_modal.py --config experiments/configs/minilm_30M_3source.yaml --gpu-type a10g

# Parameter sweep (parallel)
modal run train_modal.py --config experiments/configs/arch_sweep_base.yaml \
  --sweep experiments/configs/arch_sweep_type.yaml --gpu-type a10g

# Local training on M2 Mac
python -m experiments.run_experiment experiments/configs/smoke_test.yaml --device mps
```

### With CLI args (quick experiments)

```bash
modal run train_modal.py --batch-size 1024 --k 128 --expansion-factor 4 --gpu-type a10g
python train_local.py --batch_size 512 --k 64 --expansion_factor 8
```

## Experiment Framework

YAML-driven experiment system with config hashing, cartesian sweeps, and WandB integration. See [experiments/](experiments/) for configs and results.

```bash
# Dry-run a sweep to see what would execute
python -m experiments.run_experiment experiments/configs/arch_sweep_base.yaml \
  --sweep experiments/configs/arch_sweep_type.yaml --dry-run

# Compare results
python -m experiments.compare_results experiments/results/

# Evaluate a trained SAE
python -m experiments.eval_probes --sae-path checkpoints/sae_topk_128_4.xxx \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 --suite hard
```

### Evaluation Suite

| Task | Type | What it tests |
|------|------|---------------|
| AG News | 4-class classification | Coarse topic features |
| SST-2 | 2-class classification | Sentiment features |
| BANKING77 | 77-class classification | Fine-grained intent features |
| CLINC150 | 150-class classification | Very fine-grained (hardest) |
| STS-B | Similarity (spearman) | Continuous semantic structure |
| SciFact | Retrieval (nDCG@10) | Information preservation |
| MMCS | Feature quality | Decoder weight redundancy |

## Architecture

Supported SAE types: **TopK** (recommended), Gated, JumpReLU, LISTA.

Training features: auxk dead feature revival, k-annealing, tilted ERM, decoder decorrelation loss, fire rate penalty, mixed precision (AMP), cosine LR schedule.

### Key Research Findings

- **Expansion factor 4-8x** is optimal for embeddings (not 32x like LLM layers)
- **k=128** dramatically beats k=64 on fine-grained tasks (CLINC150: 79.6% vs 64.6%)
- **30M diverse samples** retains 97% of 150M quality at 6.4x lower cost
- **3-source data mix** matters more than any training regularization
- **A10G at batch_size=1024** is the cost-optimal GPU config ($2.60/100M samples)

## Data Preparation

Training data (pre-computed embeddings) is prepared in [latent-data-modal](https://github.com/enjalot/latent-data-modal). See the [Latent Taxonomy methodology](https://enjalot.github.io/latent-taxonomy/articles/about) for details.

## Part of the latent-* ecosystem

- [latent-scope](https://github.com/enjalot/latent-scope) — Interactive dataset exploration
- [latent-taxonomy](https://github.com/enjalot/latent-taxonomy) — SAE feature visualization
- [latent-basemap](https://github.com/enjalot/latent-basemap) — Parametric UMAP for consistent 2D layouts
- [latent-data-modal](https://github.com/enjalot/latent-data-modal) — Data pipelines on Modal
