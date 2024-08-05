import sys

from trainer import SaeTrainer
from utils.data_loader import StreamingEmbeddingDataset

from simple_parsing import parse
from modal import App, Image, Secret, Volume, build, enter, exit, gpu, method

DATASET = f"/embeddings/fineweb-edu-sample-10BT-chunked-500-HF4"
GPU_CONCURRENCY = 1
# GPU_CONFIG = gpu.A100(size="80GB")
# GPU_CONFIG = gpu.A100(size="40GB")
GPU_CONFIG = gpu.A10G()
# GPU_CONFIG = gpu.H100()

st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2",
        "numpy==1.26.3",
        "transformers==4.39.3",
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
        "einops==0.7.0",
        "bitsandbytes",
        "safetensors",
        "accelerate",
        "dataclasses",
        "tqdm",
        "pyarrow",
        "datasets",
        "simple_parsing",
        "triton",
        "wandb"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # .run_function(
    #     download_model_to_image,
    #     timeout=60 * 20,
    #     kwargs={
    #         "model_dir": MODEL_DIR,
    #         "model_name": MODEL_ID,
    #         "model_revision": MODEL_REVISION,
    #     },
    #     secrets=[Secret.from_name("huggingface-secret")],
    # )
)

with st_image.imports():
    import torch


app = App(
    "train-fineweb-sae"
)  

@app.cls(
    gpu=GPU_CONFIG,
    # cpu=16,
    concurrency_limit=GPU_CONCURRENCY,
    timeout=60 * 60 * 10, # 10 hours
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
        print("starting engine")

    @method()
    def train(self, args):

        # TRAIN
        print(f"Training on '{args.dataset}' (split '{args.split}')")
        # print(f"Storing model weights in {model.dtype}")
        dataset = StreamingEmbeddingDataset(args.dataset, args.data_type, args.embedding_column, split=args.split)
        trainer = SaeTrainer(args, dataset, self.device)
        trainer.fit()



@app.local_entrypoint()
def run(batch_size: int, expansion_factor: int, k: int, grad_acc_steps: int):
    from trainer import TrainConfig
    class RunConfig(TrainConfig):
        dataset: str = "./notebooks/data/test_train2"
        """Path to the dataset to use for training."""

        data_type: str = "huggingface"
        """Type of the dataset to use for training. 
        current options: huggingface or parquet"""

        split: str = "train"
        """Dataset split to use for training."""

        embedding_column: str = "embedding"
        """Column to use for embedding."""

        wandb_project: str = "sae-fw-modal-nomic-text-v1.5"
        """Wandb project name."""

     # Filter out the script name and 'run' command if present
    args_to_parse = [f"--batch_size={batch_size}", f"--expansion_factor={expansion_factor}", f"--k={k}", f"--grad_acc_steps={grad_acc_steps}"]
    
    args = parse(RunConfig, args=args_to_parse)
    args.dataset = DATASET
    # the trainer saves in the run_name directory
    args.checkpoints_directory = "/checkpoints"

    job = RemoteTrainer()
    job.train.remote(args)
