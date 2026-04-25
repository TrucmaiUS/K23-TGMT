import torch
import torch.nn as nn


class PatchWeightRegularization(nn.Module):
    def __init__(self, target_mean=0.5, alpha_reg=1.0, beta_reg=1.0):
        super().__init__()
        self.target_mean = target_mean
        self.alpha_reg = alpha_reg
        self.beta_reg = beta_reg

    def forward(self, patch_weights):
        if patch_weights is None:
            return None

        if patch_weights.dim() == 3:
            patch_weights = patch_weights.squeeze(-1)

        mean_per_sample = patch_weights.mean(dim=1)
        var_per_sample = patch_weights.var(dim=1, unbiased=False)

        loss_mean = (mean_per_sample - self.target_mean).pow(2).mean()
        loss_var = -var_per_sample.mean()
        return self.alpha_reg * loss_mean + self.beta_reg * loss_var
