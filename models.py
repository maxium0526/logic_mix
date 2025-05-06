import torch
from torchvision.models import resnet101
from torch import nn

class Model(torch.nn.Module):
    def __init__(self, num_categories):
        super().__init__()

        self.network = resnet101(weights='IMAGENET1K_V2')
        self.network.fc = nn.Linear(2048, num_categories)

    def forward(self, x):
        return self.network(x)
        