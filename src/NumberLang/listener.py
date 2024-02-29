import torch
from torch import nn


def gumbel_softmax(logits, tau=1, hard=False, dim=-1):
    gumbels = -torch.empty_like(logits).exponential_().log()  # Sample from Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # Add the logits
    y_soft = gumbels.softmax(dim)

    if hard:
        # Create a one-hot encoding from the soft distribution y_soft
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    return y_soft


class Listener(nn.Module):
    def __init__(self, bits):
        super("Listener", self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                1, bits * 2, kernel_size=3, stride=1, padding=1
            ),  # Increase channel depth
            nn.ReLU(),
            nn.Conv2d(
                bits * 2, bits * 4, kernel_size=3, stride=1, padding=1
            ),  # Additional conv layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(bits * 4),  # Add batch normalization
        )
        # Adjust for additional convolutional layers and increased channel depth
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))  # Adjust pooling size

        # Adjust the linear layer to match new pooling output size
        self.fc_layers = nn.Sequential(
            nn.Linear(bits * 4 * 5 * 5, 128),  # Adjust based on new output size
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout for regularization
            nn.Linear(128, bits),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        # Applying sigmoid to output probabilities for each bit being 1
        x = gumbel_softmax(x, tau=1, hard=False, dim=-1)
        return x
