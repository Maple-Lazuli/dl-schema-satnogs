"""torchvision ResNet18"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class ResNet18(nn.Module):
    """Sample ResNet model."""

    def __init__(self, cfg=None):
        super().__init__()
        resnet18 = models.resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*nn.ModuleList(resnet18.children())[:-1])
        self.fc = nn.Linear(in_features=resnet18.fc.in_features, out_features=1)
        # may want to add more fully connected and dropouts
    def forward(self, x):
        x = self.feature_extractor(x)
        # Flatten extra dimensions after average pooling layer.
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    img = torch.rand(1, 3,1542, 623)

    res = ResNet18()
    print(res(img))



