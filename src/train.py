import torch
import wandb
import yaml
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sae import SparseAutoencoder
from utils.data_loader import get_streaming_dataloader, get_dummy_dataloader
from utils.training import train_epoch, validate, SAELoss


def train(config):
    # Initialize wandb
    run = wandb.init(project="sae-nomic-text-v1.5", job_type="training", config=config)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"Using config: {config}")
    
    # Load data based on config
    if config['data_type'] == 'dummy':
        train_loader = get_dummy_dataloader(config['num_train_samples'], 768, batch_size=config['batch_size'])
        val_loader = get_dummy_dataloader(config['num_val_samples'], 768, batch_size=config['batch_size'])
    else:
        train_loader = get_streaming_dataloader(config['train_data_path'], config['data_type'], config['embedding_column'], batch_size=config['batch_size'], split="train")
        val_loader = get_streaming_dataloader(config['val_data_path'], config['data_type'], config['embedding_column'], batch_size=config['batch_size'], split="test")
    
    
    model = SparseAutoencoder(
        input_dim=768, 
        hidden_dim=768*config['expansion_factor'],
        k=config['k'],
        auxk=config['auxk'],
        dead_steps_threshold=config['dead_steps_threshold']
    ).to(device)
    
    model.init_weights()
    
    def warmup_lambda(epoch):
        if epoch < 10:
            return (epoch + 1) / 10
        return 1

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    # scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    criterion = SAELoss(l1_lambda=config['l1_lambda'], auxk_lambda=config['auxk_lambda'])
    
    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, max_steps=None, accumulation_steps=config['accumulation_steps'])

        # For validation, we need to reset the iterator
        if config['data_type'] == 'dummy':
            val_loader = get_dummy_dataloader(config['num_val_samples'], 768, batch_size=config['batch_size'])
        else:
            val_loader = get_streaming_dataloader(config['val_data_path'], config['data_type'], config['embedding_column'], batch_size=config['batch_size'], split="test")

        val_loss = validate(model, val_loader, criterion, device, max_steps=None)
        
        scheduler.step(val_loss)
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "best_model.pth"))
        
        # Log feature density histogram
        # if epoch % 5 == 0:  # Log every 5 epochs to reduce overhead
        feature_density = (model.stats_last_nonzero == 0).float().mean().item()
        wandb.log({"feature_density": feature_density})
        print("FEATURE DENSITY", feature_density)
        print("stats_last_nonzero min:", model.stats_last_nonzero.min().item())
        print("stats_last_nonzero max:", model.stats_last_nonzero.max().item())
        print("stats_last_nonzero mean:", model.stats_last_nonzero.float().mean().item())

        unique_features = torch.zeros(model.hidden_dim, dtype=torch.bool)
        with torch.no_grad():
            sample_batch = next(iter(val_loader)).to(device)
            _, activated = model(sample_batch)
            unique_features |= (activated.sum(dim=0) > 0)
            print(f"Unique features used: {unique_features.sum().item()}")
            # wandb.log({"activation_histogram": wandb.Histogram(activations.cpu().numpy())})
    
    wandb.finish()
    return best_val_loss

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the model with the specified configuration file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    train(config)