import torch
# from tqdm import tqdm
from tqdm.auto import tqdm
import wandb


def log_gradients(model, step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_abs = param.grad.abs()
            grad_mean = grad_abs.mean().item()
            grad_median = grad_abs.median().item()
            grad_max = grad_abs.max().item()
            grad_min = grad_abs.min().item()
            grad_std = grad_abs.std().item()
            
            # Log to console
            print(f"Gradient for {name} at step {step}:")
            print(f"  Mean: {grad_mean:.6f}")
            print(f"  Median: {grad_median:.6f}")
            print(f"  Max: {grad_max:.6f}")
            print(f"  Min: {grad_min:.6f}")
            print(f"  Std: {grad_std:.6f}")
            
            # Log to wandb
            wandb.log({
                f"gradient/{name}/mean": grad_mean,
                f"gradient/{name}/median": grad_median,
                f"gradient/{name}/max": grad_max,
                f"gradient/{name}/min": grad_min,
                f"gradient/{name}/std": grad_std,
            }, step=step)
            
            # Log histogram to wandb
            wandb.log({f"gradient/{name}/histogram": wandb.Histogram(grad_abs.cpu().numpy())}, step=step)

def check_vanishing_gradients(model, threshold=1e-5):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.norm().item()
            
            if grad_norm < threshold:
                print(f"Potential vanishing gradient in {name}:")
                print(f"  Gradient norm: {grad_norm:.6f}")
                print(f"  Parameter norm: {param_norm:.6f}")
                print(f"  Ratio: {grad_norm/param_norm:.6f}")
            
            # Log the ratio to wandb
            wandb.log({f"grad_param_ratio/{name}": grad_norm/param_norm})

# Call this function in your training loop
check_vanishing_gradients(model)
def log_activations(model, batch, step):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        _, activated = model(batch)
    
    # Compute statistics
    non_zero = (activated != 0).float().mean().item()
    activation_mean = activated.mean().item()
    activation_median = activated.median().item()
    activation_max = activated.max().item()
    activation_min = activated.min().item()
    activation_std = activated.std().item()
    
    # Log to console
    print(f"Activations at step {step}:")
    print(f"  Non-zero fraction: {non_zero:.6f}")
    print(f"  Mean: {activation_mean:.6f}")
    print(f"  Median: {activation_median:.6f}")
    print(f"  Max: {activation_max:.6f}")
    print(f"  Min: {activation_min:.6f}")
    print(f"  Std: {activation_std:.6f}")
    
    # Log to wandb
    wandb.log({
        "activations/non_zero_fraction": non_zero,
        "activations/mean": activation_mean,
        "activations/median": activation_median,
        "activations/max": activation_max,
        "activations/min": activation_min,
        "activations/std": activation_std,
    }, step=step)
    
    # Log histogram to wandb
    # wandb.log({"activations/histogram": wandb.Histogram(activated.cpu().numpy())}, step=step)
    
    # Log sparsity pattern
    # sparsity_pattern = (activated != 0).float().mean(dim=0).cpu().numpy()
    # wandb.log({"activations/sparsity_pattern": wandb.Image(plt.imshow(sparsity_pattern.reshape(1, -1), aspect='auto', cmap='viridis'))}, step=step)
    
    model.train()  # Set model back to training mode

class SAELoss(torch.nn.Module):
    def __init__(self, l1_lambda, auxk_lambda):
        super(SAELoss, self).__init__()
        self.l1_lambda = l1_lambda
        self.auxk_lambda = auxk_lambda
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, decoded, target, activated, autoencoder):
        mse = self.mse_loss(decoded, target)
        l1 = self.l1_lambda * activated.abs().sum()
        auxk = self.auxk_lambda * autoencoder.auxk_loss(activated)
        return mse + l1 + auxk

def unit_norm_decoder_grad_adjustment_(autoencoder):
    decoder_weight_grad = autoencoder.decoder.weight.grad
    if decoder_weight_grad is not None:
        proj = torch.sum(autoencoder.decoder.weight * decoder_weight_grad, dim=0, keepdim=True)
        decoder_weight_grad.sub_(proj * autoencoder.decoder.weight)



def get_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def get_weight_update_ratio(model, optimizer):
    ratios = []
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                ratio = p.grad.norm() / (p.data.norm() + 1e-7)
                ratios.append(ratio.item())
    return sum(ratios) / len(ratios) if ratios else 0


def train_epoch(model, dataloader, optimizer, criterion, device, max_steps=None, accumulation_steps=1):
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Use tqdm without specifying total number of items
    pbar = tqdm(enumerate(dataloader), desc="Training Epoch")
    
    optimizer.zero_grad()  # Zero gradients at the beginning of the epoch
    
    for i, batch in pbar:
        if max_steps is not None and i >= max_steps:
            break
        
        batch = batch.to(device)
        decoded, activated = model(batch)
        loss = criterion(decoded, batch, activated, model)

        if i % 100 == 0:  # Check every 100 batches
            log_gradients(model, i)
            log_activations(model, batch, i)
            # print("Activation stats:")
            # print("  Non-zero activations:", (activated != 0).float().mean().item())
            # print("  Max activation:", activated.max().item())
            # print("  Mean activation:", activated.mean().item())
        
        # Normalize the loss to account for accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            # Compute gradient norm before clipping
            # grad_norm_before = get_gradient_norm(model)
            # # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            # # Compute gradient norm after clipping
            # grad_norm_after = get_gradient_norm(model)
            
            update_ratio = get_weight_update_ratio(model, optimizer)
            
            unit_norm_decoder_grad_adjustment_(model)
            optimizer.step()
            optimizer.zero_grad()
            model.unit_norm_decoder_()
            
            wandb.log({
                'weight_update_ratio': update_ratio,
                # 'grad_norm_before_clip': grad_norm_before,
                # 'grad_norm_after_clip': grad_norm_after,
            })
        
        total_loss += loss.item() * accumulation_steps  # Denormalize the loss for logging
        num_batches += 1
        
        # Update progress bar with current average loss
        pbar.set_postfix({
            'avg_loss': total_loss / num_batches,
            'batch_loss': loss.item() * accumulation_steps,
            # 'grad_norm': grad_norm_after
        })
        
        # Log to wandb
        wandb.log({
            'batch_loss': loss.item() * accumulation_steps,  # Denormalize the loss for logging
            'avg_loss': total_loss / num_batches,
            'batch': i
        })
        
    epoch_avg_loss = total_loss / num_batches if num_batches > 0 else 0
    wandb.log({'epoch_avg_loss': epoch_avg_loss})
    
    return epoch_avg_loss


def validate(model, dataloader, criterion, device, max_steps=None):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        # Use tqdm without specifying total number of items
        pbar = tqdm(enumerate(dataloader), desc="Validation")
        
        for i, batch in pbar:
            if max_steps is not None and i >= max_steps:
                break
            
            batch = batch.to(device)
            decoded, activated = model(batch)
            loss = criterion(decoded, batch, activated, model)
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar with current average loss
            pbar.set_postfix({'avg_loss': total_loss / num_batches})
    
    return total_loss / num_batches if num_batches > 0 else 0
