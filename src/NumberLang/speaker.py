import torch
from torch import nn
import torch.nn.functional as F
from device import device
from listener import gumbel_softmax


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
    a softmax function to convert logits into probabilities, then using argmax to select the
    highest probability class, and finally adjusting the indices to the desired range.

    Args:
        logits (torch.Tensor): The raw output logits from the Scribe model, shaped
                               [batch_size, output_length, num_classes].

    Returns:
        torch.Tensor: A tensor of the same batch size and sequence length, but each element
                      is an integer in the range [38, 63].
    """
    # Apply Gumbel-Softmax to convert logits into a differentiable approximation of hard indices
    soft_indices = F.gumbel_softmax(scribe_logits, tau=1, hard=False, dim=-1)

    # Optionally, map the soft indices to the expected range [38, 63]
    # Note: This step may require adjustment based on how you intend to use the output with Tacotron2

    return soft_indices + 38
