"""
Train an SAE on Modal Labs GPU infrastructure.

Examples:
  modal run train_modal.py --batch-size 512 --grad-acc-steps 4 --k 64 --expansion-factor 32
  modal run train_modal.py --batch-size 512 --sae-type gated --k 64
"""
from latentsae.trainer import SaeTrainer, TrainConfig
from latentsae.utils.data_loader import ShardedEmbeddingDataset

from simple_parsing import parse
from modal import App, Image, Secret, Volume, enter, gpu, method

# ── Configuration ──────────────────────────────────────────────
# Override these for your experiment, or pass via CLI args
DATASETS = [
    "/embeddings/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train",
    "/embeddings/RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2/train",
    "/embeddings/pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2/train",
]
D_IN = 384
WANDB_PROJECT = "sae-all-minilm-l6-v2"
GPU_CONFIG = gpu.A10G()

st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2",
        "numpy==1.26.3",
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
        "einops==0.7.0",
        "bitsandbytes",
        "safetensors",
        "dataclasses",
        "tqdm",
        "pyarrow",
        "datasets",
        "simple_parsing",
        "triton",
        "wandb",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

with st_image.imports():
    import torch

app = App("train-sae")


@app.cls(
    gpu=GPU_CONFIG,
    concurrency_limit=1,
    timeout=60 * 60 * 10,
    container_idle_timeout=1200,
    allow_concurrent_inputs=1,
    image=st_image,
    volumes={
        "/embeddings": Volume.from_name("embeddings", create_if_missing=True),
        "/checkpoints": Volume.from_name("checkpoints", create_if_missing=True),
    },
    secrets=[Secret.from_name("enjalot-wandb-secret")],
)
class RemoteTrainer:
    @enter()
    def start_engine(self):
        self.device = torch.device("cuda")

    @method()
    def train(self, args):
        dataset = ShardedEmbeddingDataset(
            args.dataset,
            cache_size=10,
            d_in=args.d_in,
            shuffle=True,
            warm_up_cache=False,
            file_type="npy",
        )
        trainer = SaeTrainer(args, dataset, self.device)
        trainer.fit()


@app.local_entrypoint()
def run(
    batch_size: int,
    expansion_factor: int,
    k: int,
    grad_acc_steps: int,
    sae_type: str = "topk",
):
    args_to_parse = [
        f"--batch_size={batch_size}",
        f"--expansion_factor={expansion_factor}",
        f"--k={k}",
        f"--grad_acc_steps={grad_acc_steps}",
        f"--sae_type={sae_type}",
    ]
    args = parse(TrainConfig, args=args_to_parse)
    args.dataset = DATASETS
    args.d_in = D_IN
    args.wandb_project = WANDB_PROJECT
    args.checkpoints_directory = "/checkpoints"

    job = RemoteTrainer()
    job.train.remote(args)
