from torchvision.models import vgg19
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor
import torch

class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        # NOTE:
        # To follow the paper, replace this backbone with a VGG19
        # pretrained on B-scan classification.
        try:
            from torchvision.models import VGG19_Weights
            backbone = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        except Exception:
            backbone = vgg19(pretrained=True)
        self.vgg19 = backbone.features

        first_conv = self.vgg19[0]
        gray_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=(first_conv.bias is not None))
        with torch.no_grad():
            gray_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
            if first_conv.bias is not None:
                gray_conv.bias.copy_(first_conv.bias)
        self.vgg19[0] = gray_conv

        self.max_pool_layers = [4, 9, 18, 27, 36]

        for param in self.vgg19.parameters():
            param.requires_grad_(False)

        for idx, module in enumerate(self.vgg19):
            if hasattr(module, 'inplace'):
                self.vgg19[idx].inplace = False
            if idx in self.max_pool_layers:
                self.vgg19[idx] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.model = create_feature_extractor(self.vgg19, {
            '1': 'conv1_1',
            '3': 'conv1_2',
            '6': 'conv2_1',
            '8': 'conv2_2',
            '11': 'conv3_1',
            '13': 'conv3_2',
            '15': 'conv3_3',
            '17': 'conv3_4',
            '20': 'conv4_1',
            '22': 'conv4_2',
            '24': 'conv4_3',
            '26': 'conv4_4',
            '29': 'conv5_1',
            '31': 'conv5_2',
            '33': 'conv5_3',
            '35': 'conv5_4',
        })
        

    def forward(self, x):
        output = self.model(x)
        
        return output
