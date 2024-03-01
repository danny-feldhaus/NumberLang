import torch
from torch import nn
from device import device
from scribe import ArgMaxSTE


def initialize_speaker() -> nn.Module:
    tacotron2 = torch.hub.load(
        repo_or_dir="NVIDIA/DeepLearningExamples:torchhub",
        model="nvidia_tacotron2",
        source="github",
        model_math="fp16",
        force_reload=False,
        map_location=device,
    )
    tacotron2.eval()
    for param in tacotron2.parameters():
        param.requires_grad = False  # Freeze the model
    return tacotron2


def map_scribe_to_speaker(scribe_logits: torch.Tensor) -> torch.Tensor:
    """
    Converts logits from the Scribe model into integers in the range [38, 63] by first applying
    a softmax function to convert logits into probabilities, then using a custom Straight-Through
    Estimator for argmax to select the highest probability class, and finally adjusting the indices
    to the desired range.

    Args:
        logits (torch.Tensor): The raw output logits from the Scribe model, shaped
                               [batch_size, output_length, num_classes].

    Returns:
        torch.Tensor: A tensor of the same batch size and sequence length, but each element
                      is an integer in the range [38, 63].
    """
    # Replace the direct argmax operation with the STE version
    return ArgMaxSTE.apply(scribe_logits)
