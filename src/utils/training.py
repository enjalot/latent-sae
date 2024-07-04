import torch
from tqdm import tqdm
# import wandb

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


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training Epoch"):
        batch = batch.to(device)
        optimizer.zero_grad()
        decoded, activated = model(batch)
        loss = criterion(decoded, batch, activated, model)
        loss.backward()
        unit_norm_decoder_grad_adjustment_(model)
        optimizer.step()
        model.unit_norm_decoder_()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch = batch.to(device)
            decoded, activated = model(batch)
            loss = criterion(decoded, batch, activated, model)
            total_loss += loss.item()
    return total_loss / len(dataloader)
