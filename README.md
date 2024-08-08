# latent-sae

__WARNING__: This repo is very experimental and being actively developed for experimentation. The API for the SAE models as well as the organization of the models will likely change. The models that have been trained will also get re-trained as I prepare more training data.

Most of the code for SAE comes from https://github.com/EleutherAI/sae

#TODO: fully document data usage
Currently training on https://huggingface.co/datasets/enjalot/fineweb-edu-sample-10BT-chunked-500-nomic-text-v1.5
For locally testing the code I downloaded a sample of the dataset.
For training, I downloaded the whole dataset to disk in a modal volume, then processed it into sharded torch .pt files using this script:
https://github.com/enjalot/fineweb-modal/blob/main/torched.py

## Inference

```python
model = Sae.load_from_hub("enjalot/sae-nomic-text-v1.5-FineWeb-edu-10BT", "64_32")
# or from disk
model = Sae.load_from_disk("models/sae_64_32.3mq7ckj7")
```

See [notebooks/eval.ipynb](notebooks/eval.ipynb) for an example of how to use the model for extracting features from an embedding dataset.

## Training

The main way to train (that I've gotten working) is using modal_labs infrastructure 
```bash
modal run src/run_modal.py --batch-size 512 --grad-acc-steps 4 --k 64 --expansion-factor 128
```

I do have some initial code for training locally

```bash
python latentsae/run.py --batch-size 512 --grad-acc-steps 4 --k 64 --expansion-factor 128 
```

