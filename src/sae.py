import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        values, indices = input.topk(k, dim=-1)
        ctx.save_for_backward(indices, input)
        return torch.where(input >= values.min(dim=-1, keepdim=True)[0], input, torch.zeros_like(input))

    @staticmethod
    def backward(ctx, grad_output):
        indices, input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.scatter_(1, indices, 0)
        return grad_input, None

class TopK(nn.Module):
    def __init__(self, k):
        super(TopK, self).__init__()
        self.k = k

    def forward(self, x):
        return TopKFunction.apply(x, self.k)


class SmoothTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        values, indices = input.topk(k, dim=-1)
        ctx.save_for_backward(input, values, indices)
        return torch.where(input >= values.min(dim=-1, keepdim=True)[0], input, torch.zeros_like(input))

    @staticmethod
    def backward(ctx, grad_output):
        input, values, indices = ctx.saved_tensors
        grad_input = grad_output.clone()
        threshold = values.min(dim=-1, keepdim=True)[0]
        grad_input *= (input >= threshold).float()
        return grad_input, None

class SmoothTopK(nn.Module):
    def __init__(self, k):
        super(SmoothTopK, self).__init__()
        self.k = k

    def forward(self, x):
        return SmoothTopKFunction.apply(x, self.k)


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, k, auxk=None, dead_steps_threshold=10000):
        super(SparseAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.auxk = auxk
        self.dead_steps_threshold = dead_steps_threshold

        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        # self.topk = TopK(k)
        self.topk = SmoothTopK(k)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        # self.ln1 = nn.LayerNorm(hidden_dim)
        # self.ln2 = nn.LayerNorm(input_dim)


        self.register_buffer("stats_last_nonzero", torch.zeros(hidden_dim, dtype=torch.long))

    def init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        self.decoder.weight.data = self.decoder.weight.data.T.contiguous().T
        self.unit_norm_decoder_()

    def unit_norm_decoder_(self):
        with torch.no_grad():
            self.decoder.weight.data /= self.decoder.weight.data.norm(dim=0, keepdim=True)

    def forward(self, x):
        x = x - self.pre_bias
        # encoded = self.encoder(x)
        encoded = self.bn1(self.encoder(x))
        # encoded = self.ln1(self.encoder(x))
        
        encoded = F.normalize(encoded, p=2, dim=-1)  # L2 normalization

        # TopK activation
        # print("Before TopK - non-zero:", (encoded != 0).float().mean().item())
        activated = self.topk(encoded)
        # print("After TopK - non-zero:", (activated != 0).float().mean().item())
        
        # Update stats for dead features
        self.update_stats(activated)
        
        # decoded = self.decoder(activated)
        decoded = self.bn2(self.decoder(activated))
        # decoded = self.ln2(self.decoder(activated))
        return decoded + self.pre_bias, activated

    def update_stats(self, activated):
        nonzero = (activated != 0).any(0)
        self.stats_last_nonzero *= 1 - nonzero.long()
        self.stats_last_nonzero += 1

    def encode(self, x):
        x = x - self.pre_bias
        return self.topk(self.encoder(x))

    def auxk_loss(self, activated):
        if self.auxk is None:
            return 0
        
        dead_mask = self.stats_last_nonzero > self.dead_steps_threshold
        auxk_activated = self.topk(activated * dead_mask.float())
        return auxk_activated.abs().sum()

def unit_norm_decoder_grad_adjustment_(autoencoder):
    decoder_weight_grad = autoencoder.decoder.weight.grad
    if decoder_weight_grad is not None:
        proj = torch.sum(autoencoder.decoder.weight * decoder_weight_grad, dim=0, keepdim=True)
        decoder_weight_grad.sub_(proj * autoencoder.decoder.weight)

class SAELoss(nn.Module):
    def __init__(self, l1_lambda, auxk_lambda):
        super(SAELoss, self).__init__()
        self.l1_lambda = l1_lambda
        self.auxk_lambda = auxk_lambda
        self.mse_loss = nn.MSELoss()

    def forward(self, decoded, target, activated, autoencoder):
        mse = self.mse_loss(decoded, target)
        l1 = self.l1_lambda * activated.abs().sum()
        auxk = self.auxk_lambda * autoencoder.auxk_loss(activated)
        return mse + l1 + auxk