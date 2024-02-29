from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from .scribe import Scribe
from .listener import Listener
from .speaker import map_scribe_to_speaker
from .device import device


@dataclass
class ResultCollection:
    """A collection of inputs and outputs from the three models, generated during training."""

    scribe_in: torch.Tensor
    scribe_out: torch.Tensor

    speaker_in: torch.Tensor
    speaker_out: torch.Tensor

    listener_out: torch.Tensor


def run_through_all_models(
    binary_input: torch.Tensor, scribe: Scribe, speaker: nn.Module, listener: Listener
) -> ResultCollection:
    binary_input = binary_input.to(device)

    char_logits = scribe(binary_input)

    mapped_char_logits = map_scribe_to_speaker(char_logits)
    char_logits_lengths = torch.full(
        (char_logits_lengths.size(0),), scribe.output_length, dtype=torch.long
    ).to(device)
    spectrogram, _, _ = speaker.infer(mapped_char_logits, char_logits_lengths)
    spectrogram = spectrogram.unsqueeze(1)
    binary_output = listener(spectrogram)
    return ResultCollection(
        scribe_in=binary_input,
        scribe_out=char_logits,
        speaker_in=mapped_char_logits,
        speaker_out=spectrogram,
        listener_out=binary_output,
    )


def train(
    scribe: Scribe,
    speaker: nn.Module,
    listener: Listener,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion,
    epochs: int,
):
    """
    Trains the Scribe and Listener networks using a given DataLoader, optimizer, and loss function.
    Tacotron2 is used in a frozen state to generate audio spectrograms from the mapped integers output
    by Scribe.
    """
    assert epochs > 0, "Must train for at least 1 epoch"

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )
    scribe = scribe.to(device)
    speaker = speaker.to(device)
    listener = listener.to(device)

    for epoch in range(epochs):
        scribe.train()
        listener.train()
        total_loss = 0
        # Wrap the dataloader with tqdm for a progress bar
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{epochs}", total=len(dataloader)
        )
        for decimal_input, binary_input in progress_bar:
            optimizer.zero_grad()
            results = run_through_all_models(binary_input, scribe, speaker, listener)
            # Calculate loss using the custom numeric difference criterion
            loss = criterion(results.speaker_in, results.listener_out)

            # Backpropagation
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update the progress bar
            progress_bar.set_postfix(loss=loss.item())
        average_loss = total_loss / len(dataloader)
        print(f"Completed Epoch {epoch+1}, Average Loss: {average_loss:.4f}")
        scheduler.step(average_loss)
