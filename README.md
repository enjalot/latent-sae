# latent-sae

This is essentially a fork of [EleutherAI/sae](https://github.com/EleutherAI/sae) focused on training Sparse Autodencoders on Sentence transformer embeddings. The main differences are:  
1) Focus on training only one model on one set of input
2) Load training data (embeddings) quickly from disk



## Inference

```python
# !pip install latentsae
from latentsae import Sae
sae_model = Sae.load_from_hub("enjalot/sae-nomic-text-v1.5-FineWeb-edu-100BT", "64_32")
# or from disk
sae_model = Sae.load_from_disk("models/sae_64_32.3mq7ckj7")

# Get some embeddings
texts = ["Hello world", "Will I ever halt?", "Goodbye world"]
from sentence_transformers import SentenceTransformer
emb_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
embeddings = emb_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

features = sae_model.encode(normalized_embeddings)
print(features.top_indices)
print(features.top_acts)
```

See [notebooks/eval.ipynb](notebooks/eval.ipynb) for an example of how to use the model for extracting features from an embedding dataset.

## Training

The main way to train (that I've gotten working) is using modal_labs infrastructure 
```bash
modal run train_modal.py --batch-size 512 --grad-acc-steps 4 --k 64 --expansion-factor 128
```

I do have some initial code for training locally

```bash
python train_local.py --batch-size 512 --grad-acc-steps 4 --k 64 --expansion-factor 128 
```

## Data Preparation
I wrote a detailed article on the methodology behind the data, training and analysis of the SAEs trained with this repo:
[Latent Taxonomy Methodology](https://enjalot.github.io/latent-taxonomy/articles/about)

I used [Modal Labs](https://modal.com) to rent VMs and GPUs for the data preprocessing and training. See [enjalot/fineweb-modal](https://github.com/enjalot/fineweb-modal) for the scripts used to preprocess the FineWeb-EDU 10BT and 100BT samples and embed them with [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5).

I first trained on the 10BT sample, chunked into 500 token chunks which is available [on HuggingFace](https://huggingface.co/datasets/enjalot/fineweb-edu-sample-10BT-chunked-500-nomic-text-v1.5). This gave 25 million embeddings to train on.
From the wandb charts it looked like the model could improve further with more data so I then prepared 10x the embeddings with the 100BT sample. I'm working on uploading that to HF still.

For locally testing the code I downloaded a single parquet file from the dataset.
For the full training run, I downloaded the whole dataset to disk in a modal volume, then processed it into sharded torch .pt files using this script: [torched.py](https://github.com/enjalot/fineweb-modal/blob/main/torched.py)

## Parameters
The main parameters I tried to change were:

- batch-size: how many embeddings in a batch (bigger is better?) settled on 512 for performance tradeoff
- grad-acc-steps: how many steps to skip updating gradient. simulates bigger batch size. not sure the penalty for making this really big. settled on 4 with batch size of 512
- k: sparsity; how many top features to consider. fewer is sparser and more interpretable, but worse error. tried 64 and 128 but unsure how to measure quality differences yet
- expansion factor: multiply times dimensions of input embedding (768 in case of nomic). chose 32 and 128 to give ~25k and ~100k features respectively.

### Open questions
Another thought I have that I might try is to process the data into even smaller chunks. At 500 tokens the samples are quite large and I believe we are essentially aggregating a lot of features across those tokens. 
If we chunked at something like 100 tokens each sample would be much more granular and we would also have 5x more training data.
Again, I'm not sure how I'd evaluate the quality tradeoff of this yet.

Part of the motivation with this repo and the [fineweb-modal](https://github.com/enjalot/fineweb-modal) repo is to make it easier
to train SAEs on other datasets. FineWeb-EDU has certain desirable properties for some down-stream tasks, but I can imagine training on a large dataset of code or a more general corpus like RedPajama v2.

