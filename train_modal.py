"""
Train an SAE on Modal Labs GPU infrastructure.

Examples:
  # Using YAML config (recommended):
  modal run train_modal.py --config experiments/configs/gpu_bench_base.yaml --gpu-type a10g

  # With overrides (comma-separated):
  modal run train_modal.py --config experiments/configs/gpu_bench_base.yaml \
    --gpu-type t4 --override "sae.k=128,train.batch_size=1024"

  # Sweep (runs all configs sequentially on Modal):
  modal run train_modal.py --config experiments/configs/arch_sweep_base.yaml \
    --sweep experiments/configs/arch_sweep_type.yaml --gpu-type a10g

  # Legacy CLI args (still supported for quick one-offs):
  modal run train_modal.py --batch-size 512 --grad-acc-steps 4 --k 64 --expansion-factor 32
"""
from latentsae.trainer import SaeTrainer, TrainConfig
from latentsae.utils.data_loader import ShardedEmbeddingDataset

from simple_parsing import parse
from modal import App, Image, Secret, Volume

# ── Default configuration (used when no YAML config provided) ──
DEFAULT_DATASETS = [
    "/embeddings/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train",
    "/embeddings/RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2/train",
    "/embeddings/pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2/train",
]
DEFAULT_D_IN = 384
DEFAULT_WANDB_PROJECT = "sae-all-minilm-l6-v2"

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
        "pyyaml",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("latentsae")
    .add_local_python_source("experiments")
)

with st_image.imports():
    import torch

app = App("train-sae")

VOLUMES = {
    "/embeddings": Volume.from_name("embeddings", create_if_missing=True),
    "/checkpoints": Volume.from_name("checkpoints", create_if_missing=True),
}
SECRETS = [Secret.from_name("enjalot-wandb-secret")]
COMMON_KWARGS = dict(
    max_containers=5,
    timeout=60 * 60 * 10,
    scaledown_window=1200,
    single_use_containers=True,
    image=st_image,
    volumes=VOLUMES,
    secrets=SECRETS,
)


def _do_train(args, datasets, d_in, file_type="npy", n_samples=None):
    """Shared training logic for all GPU types."""
    device = torch.device("cuda")

    dataset = ShardedEmbeddingDataset(
        datasets,
        cache_size=10,
        d_in=d_in,
        shuffle=True,
        warm_up_cache=False,
        file_type=file_type,
    )

    if n_samples and n_samples < len(dataset):
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(n_samples)))
        print(f"Subsampled to {n_samples:_} samples")

    trainer = SaeTrainer(args, dataset, device)
    trainer.fit()


# ── Per-GPU training functions ──
# Modal requires gpu= at decoration time, so we define one function per GPU type.
# In modal >=1.0 the gpu parameter is a string (e.g. "T4", "A10G", "A100-40GB").

@app.function(gpu="T4", **COMMON_KWARGS)
def train_t4(args, datasets, d_in, file_type="npy", n_samples=None):
    _do_train(args, datasets, d_in, file_type, n_samples)


@app.function(gpu="A10G", **COMMON_KWARGS)
def train_a10g(args, datasets, d_in, file_type="npy", n_samples=None):
    _do_train(args, datasets, d_in, file_type, n_samples)


@app.function(gpu="A100-40GB", **COMMON_KWARGS)
def train_a100(args, datasets, d_in, file_type="npy", n_samples=None):
    _do_train(args, datasets, d_in, file_type, n_samples)


GPU_DISPATCH = {
    "t4": train_t4,
    "a10g": train_a10g,
    "a100_40gb": train_a100,
}


@app.local_entrypoint()
def run(
    # YAML config mode (recommended)
    config: str = "",
    sweep: str = "",
    override: str = "",
    gpu_type: str = "a10g",
    # Legacy CLI args (used when config is not provided)
    batch_size: int = 512,
    expansion_factor: int = 32,
    k: int = 64,
    grad_acc_steps: int = 4,
    sae_type: str = "topk",
    k_anneal: bool = False,
    multi_topk: bool = False,
    auxk_alpha: float = 0.0,
    tilted_erm_tilt: float = 0.0,
    use_amp: bool = True,
    save_every: int = 1000,
    run_name: str = "",
    wandb_project: str = "",
):
    train_fn = GPU_DISPATCH.get(gpu_type)
    if not train_fn:
        raise ValueError(f"Unknown gpu_type: {gpu_type}. Choose from: {list(GPU_DISPATCH.keys())}")

    if config:
        import yaml as _yaml
        from experiments.experiment_config import ExperimentConfig, generate_sweep_configs

        exp_config = ExperimentConfig.from_yaml(config)

        # Apply overrides
        if override:
            overrides = {}
            for pair in override.split(","):
                pair = pair.strip()
                if not pair:
                    continue
                k_ov, v_ov = pair.split("=", 1)
                overrides[k_ov] = v_ov
            exp_config.apply_overrides(overrides)

        exp_config.infra.gpu_type = gpu_type

        if sweep:
            with open(sweep) as f:
                sweep_data = _yaml.safe_load(f)
            sweep_params = sweep_data.get("sweep", {})
            configs = generate_sweep_configs(exp_config, sweep_params)
            print(f"Running sweep: {len(configs)} configs on {gpu_type} (parallel)")

            # Dispatch all configs in parallel using .spawn()
            handles = []
            for i, cfg in enumerate(configs):
                cfg.infra.gpu_type = gpu_type
                print(f"  Spawning {i+1}/{len(configs)}: {cfg.name} [{cfg.config_hash()}]")
                train_config = cfg.to_train_config()
                h = train_fn.spawn(
                    train_config,
                    cfg.data.datasets,
                    cfg.data.d_in,
                    cfg.data.file_type,
                    cfg.data.n_samples,
                )
                handles.append((cfg.name, h))

            # Wait for all to complete
            print(f"\nAll {len(handles)} jobs spawned. Waiting for completion...")
            for name, h in handles:
                try:
                    h.get()
                    print(f"  Done: {name}")
                except Exception as e:
                    print(f"  FAILED: {name} — {e}")
        else:
            train_config = exp_config.to_train_config()
            print(f"Running: {exp_config.name} on {gpu_type}")
            train_fn.remote(
                train_config,
                exp_config.data.datasets,
                exp_config.data.d_in,
                exp_config.data.file_type,
                exp_config.data.n_samples,
            )
    else:
        # Legacy CLI mode
        args_to_parse = [
            f"--batch_size={batch_size}",
            f"--expansion_factor={expansion_factor}",
            f"--k={k}",
            f"--grad_acc_steps={grad_acc_steps}",
            f"--sae_type={sae_type}",
        ]
        if k_anneal:
            args_to_parse.append("--k_anneal=True")
        if multi_topk:
            args_to_parse.append("--multi_topk=True")
        if auxk_alpha > 0:
            args_to_parse.append(f"--auxk_alpha={auxk_alpha}")
        if tilted_erm_tilt > 0:
            args_to_parse.append(f"--tilted_erm_tilt={tilted_erm_tilt}")
        if not use_amp:
            args_to_parse.append("--use_amp=False")
        if save_every != 1000:
            args_to_parse.append(f"--save_every={save_every}")

        args = parse(TrainConfig, args=args_to_parse)
        args.dataset = DEFAULT_DATASETS
        args.d_in = DEFAULT_D_IN
        args.wandb_project = wandb_project or DEFAULT_WANDB_PROJECT
        args.run_name = run_name or None
        args.checkpoints_directory = "/checkpoints"
        args.gpu_type = gpu_type

        train_fn.remote(args, DEFAULT_DATASETS, DEFAULT_D_IN)
