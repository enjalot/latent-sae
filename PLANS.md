# Future Plans

## LISTA Encoder (Not Yet Implemented)

### What is LISTA?

LISTA (Learned ISTA) is a neural approximation to iterative sparse coding from
Gregor & LeCun (2010). Instead of the standard SAE encoder `ReLU(W_enc @ x + b)`,
LISTA adds a **lateral inhibition matrix** S that lets features compete with each
other *before* the sparsity step:

```
z = ReLU(W_enc @ x + b)           # standard SAE encoder
z = ReLU(W_enc @ x + S @ z + b)   # LISTA: one step of learned recurrence
```

The matrix S (num_latents × num_latents) captures correlations between features.
If feature i fires, it can suppress correlated feature j through a negative S[i,j],
reducing redundancy. This is one iteration of an unrolled sparse coding algorithm.

### Why it might help for embedding SAEs

Sentence embeddings pack many overlapping concepts into a relatively low-dimensional
space (768d for nomic, 384d for MiniLM). Features are likely correlated — e.g. "about
science" and "about physics" share variance. The lateral inhibition in LISTA gives
features a chance to sort out who should represent what, rather than relying purely on
top-k to do this implicitly.

### Implementation plan

1. **Add SaeType.LISTA** to the config enum

2. **New parameters in SaeConfig:**
   - `lista_eta: float = 0.1` — step size for the LISTA recurrence (controls how
     much the lateral inhibition matrix influences the output; the chanind research
     found eta=0.3 worked on synthetic data but broke on LLMs, so start conservative)
   - `lista_steps: int = 1` — number of LISTA iterations (1 is standard, >1 is
     deeper unrolling but more compute)

3. **Architecture changes in Sae.__init__:**
   ```python
   if cfg.sae_type == SaeType.LISTA:
       self.encoder = nn.Linear(d_in, num_latents)
       # S is the lateral inhibition matrix — learned during training
       # Initialize to negative of encoder weight correlations so correlated
       # features suppress each other from the start
       self.S = nn.Linear(num_latents, num_latents, bias=False)
       # Initialize S ≈ I - eta * W_enc @ W_enc.T (approximate ISTA step)
       with torch.no_grad():
           W = self.encoder.weight.data  # [num_latents, d_in]
           self.S.weight.data = torch.eye(num_latents) - eta * (W @ W.T)
   ```

4. **Encoder forward:**
   ```python
   def encode_lista(self, x):
       sae_in = x - self.b_dec
       z = F.relu(self.encoder(sae_in))  # initial encoding
       for _ in range(self.cfg.lista_steps):
           z = F.relu(self.encoder(sae_in) + self.S(z))  # lateral inhibition
       return self.select_topk(z)  # still use top-k for final sparsity
   ```

5. **Memory concern:** The S matrix is num_latents × num_latents. At expansion_factor=32
   with d_in=768, that's 24576² ≈ 600M parameters — way too large. Solutions:
   - **Low-rank S:** `S = A @ B` where A is [num_latents, rank] and B is [rank, num_latents].
     Rank 256-512 should capture the major correlations.
   - **Block-diagonal S:** Group features and only allow inhibition within groups.
   - **Sparse S:** Only learn inhibition for the top-k most correlated feature pairs.

   The low-rank approach is simplest and most likely what we'd try first:
   ```python
   self.S_A = nn.Linear(num_latents, lista_rank, bias=False)
   self.S_B = nn.Linear(lista_rank, num_latents, bias=False)
   # Forward: z = relu(W_enc @ x + S_A(S_B(z)))
   ```

6. **Risks and unknowns:**
   - The chanind research found LISTA with high eta breaks on LLM hidden states.
     Embedding spaces are different (lower-dimensional, normalized) so results may differ.
   - The extra parameters and compute per forward pass may not be worth the quality gain.
   - Need careful evaluation (see below) to know if it actually helps.

### Decision criteria for trying LISTA

Implement LISTA if:
- Baseline evaluation (see below) shows significant room for improvement beyond what
  k-annealing and tilted ERM provide
- The low-rank variant is tractable memory-wise for your expansion factor
- You're willing to do a sweep over eta and rank


## Evaluation Plan for Embedding SAEs

### The core question

How do we know if one SAE is better than another for embedding representations?
FVU (reconstruction loss) alone is insufficient — a lower FVU doesn't necessarily
mean the SAE learned more interpretable or useful features.

### Evaluation strategy: Downstream probe tasks

Train lightweight probes (logistic regression or small MLPs) on the SAE's sparse
feature activations and compare against probes on the raw embeddings. If the SAE
features are good, probes on SAE activations should approach or match raw-embedding
probe performance despite being much sparser.

#### Step 1: Choose probe datasets

Pick classification/retrieval tasks where your embedding model performs well. These
should span different semantic levels:

| Task Type | Example Datasets | What it tests |
|-----------|-----------------|---------------|
| Topic classification | 20 Newsgroups, AG News, DBpedia | Coarse semantic features |
| Sentiment | SST-2, Amazon Reviews | Fine-grained semantic features |
| Semantic similarity | STS Benchmark, SICK | Continuous semantic structure |
| Retrieval | MTEB subsets (retrieval split) | Practical downstream utility |
| Clustering | Tweet clustering, Reddit communities | Feature compositionality |

Start with **3 diverse tasks**: one topic (AG News), one sentiment (SST-2), one
similarity (STS-B). These are small, fast, and well-understood.

#### Step 2: Create the evaluation pipeline

```
For each SAE checkpoint:
  1. Encode eval dataset through the embedding model → raw embeddings
  2. Encode raw embeddings through SAE → sparse activations [batch, k] or dense [batch, num_latents]
  3. Train logistic regression probe on:
     a) Raw embeddings (ceiling — what's the best you can do with the original info?)
     b) SAE reconstructed embeddings (how much info survives the bottleneck?)
     c) SAE sparse activations (are the learned features individually useful?)
  4. Report accuracy / F1 / Spearman correlation for each

Key metrics:
  - "Reconstruction gap": probe(raw) - probe(reconstructed) — information lost
  - "Feature quality": probe(sparse activations) — how good are the features themselves
  - "Sparsity-quality tradeoff": plot feature quality vs L0 (number of active features)
```

#### Step 3: Implement as `eval_probes.py`

```python
# Pseudocode for the evaluation script
def evaluate_sae(sae, embedding_model, datasets, device):
    results = {}
    for name, dataset in datasets.items():
        # Get raw embeddings
        raw_embeds = embed(embedding_model, dataset.texts)

        # Get SAE outputs
        with torch.no_grad():
            encoder_out = sae.encode(raw_embeds)
            reconstructed = sae.decode(encoder_out.top_acts, encoder_out.top_indices)

            # Dense activations (for sparse feature probe)
            sparse_acts = torch.zeros(len(raw_embeds), sae.num_latents)
            sparse_acts.scatter_(1, encoder_out.top_indices, encoder_out.top_acts)

        # Train probes
        results[name] = {
            "raw_ceiling": train_probe(raw_embeds, dataset.labels),
            "reconstructed": train_probe(reconstructed, dataset.labels),
            "sparse_features": train_probe(sparse_acts, dataset.labels),
            "fvu": compute_fvu(raw_embeds, reconstructed),
            "l0": encoder_out.top_acts.shape[-1],  # effective sparsity
        }
    return results
```

#### Step 4: Compare across SAE variants

Run the evaluation for each configuration you want to compare:
- TopK vs Gated vs JumpReLU (same k, same expansion_factor)
- With/without k-annealing
- With/without tilted ERM
- Different k values (8, 32, 64, 128)
- Different expansion factors (8, 32, 64)

Plot: **probe accuracy vs FVU** — you want SAEs in the upper-left (low FVU, high
probe accuracy). An SAE that has slightly worse FVU but much better probe accuracy
has learned more useful features.

#### Step 5: Feature-level analysis

Beyond aggregate probe accuracy, look at individual features:
- **Feature activation frequency:** histogram of how often each feature fires.
  Healthy SAEs have a log-uniform-ish distribution; bad ones have many dead features
  and a few that fire on everything.
- **Feature specificity:** For each feature, what inputs activate it most? Cluster the
  top-activating inputs and see if they share a coherent theme.
- **Feature redundancy:** Cosine similarity between decoder weight vectors. High
  similarity between features = wasted capacity.

### Implementation priority

1. **First: FVU + dead feature % comparison** (already logged to wandb)
2. **Second: AG News + SST-2 probes** (fast, decisive, covers topic + sentiment)
3. **Third: STS-B correlation** (tests if continuous similarity structure is preserved)
4. **Fourth: Feature-level analysis** (only after you find a promising configuration)

### What "good enough" looks like

- Probe accuracy on SAE sparse features within **5%** of raw embedding probes
- Dead feature % under **10%** after full training
- Features show coherent activation patterns when inspected manually
- FVU < 0.05 (95%+ variance explained)

If k-annealing + tilted ERM get you there, LISTA may not be worth the complexity.
If there's still a significant gap, LISTA is the next thing to try.
