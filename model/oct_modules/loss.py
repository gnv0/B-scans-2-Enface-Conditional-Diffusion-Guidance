from torch import nn
import torch
from collections import defaultdict

class StyleTransferLoss(nn.Module):
    def __init__(self, alpha, beta, layer_weights=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.layer_weights = layer_weights or {}
        self.mse_loss = nn.MSELoss()

    def gram_matrices(self, style_representations):
        gram_matrices_representation = defaultdict()
        for layer_name, feature in style_representations.items():
            b, c, n = feature.shape
            feature_trans = torch.transpose(feature, 1, 2)
            gram = torch.bmm(feature, feature_trans)
            gram_matrices_representation[layer_name] = gram / float(c * n)
        
        return gram_matrices_representation

    def flatten_representation(self, representation):
        flattened_representation = defaultdict()
        for layer_name, feature in representation.items():
            flattened_representation[layer_name] = torch.flatten(feature, start_dim=2)
        return flattened_representation

    def forward(self, input_representation, style_representation):
        loss = 0
        input_grams = self.gram_matrices(self.flatten_representation(input_representation))
        style_grams = self.gram_matrices(self.flatten_representation(style_representation))

        common_layers = [layer for layer in input_grams.keys() if layer in style_grams]
        for layer_name in common_layers:
            omega_l = self.layer_weights.get(layer_name, 1.0)
            loss += omega_l * self.beta * self.mse_loss(input_grams[layer_name], style_grams[layer_name])

        return loss
