from torch import nn
from device import device
from torch.autograd import Function


class ArgMaxSTE(Function):
    @staticmethod
    def forward(ctx, input):
        # Perform the argmax operation and adjust indices to the desired range
        indices = input.argmax(dim=-1) + 38
        return (
            indices.float()
        )  # Ensure the output is floating-point for gradient computation

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass the gradient unchanged
        return grad_output


class Scribe(nn.Module):
    """
    The Scribe network converts a binary number to a sequence where each element
    maps to an integer between 38 and 63, using a softmax output to represent probabilities
    across the 26 classes (integers 38 through 63).
    """

    def __init__(self, bits, output_length):
        super(Scribe, self).__init__()
        self.output_length = output_length
        # Number of classes is 27 (integers 38 through 63 inclusive)
        self.num_classes = 63 - 38 + 2  # Equals 27 (a-z plus space)
        self.fc_layers = nn.Sequential(
            nn.Linear(bits, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout for regularization
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),  # Add batch normalization
            nn.Linear(256, 512),  # Increase complexity
            nn.ReLU(),
            nn.Linear(512, self.output_length * self.num_classes),
        )

    def forward(self, x):
        x = self.fc_layers(x)
        # Ensure the reshaping reflects the intended [batch_size, output_length, num_classes]
        x = x.view(-1, self.output_length, self.num_classes)
        # x = gumbel_softmax(x)

        return x
