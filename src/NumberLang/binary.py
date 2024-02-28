from typing import Tuple
import unittest

import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor


def binary_to_decimal(binary_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a binary tensor to its decimal equivalent. This function supports both 1D and 2D tensors.

    Args:
        binary_tensor (torch.Tensor): A 1D or 2D tensor representing binary numbers.
                                       For a 1D tensor, the shape should be (num_bits,).
                                       For a 2D tensor, the shape should be (batch_size, num_bits).

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) or a scalar tensor if input is 1D,
                      representing the decimal equivalents.
    """
    # Ensure binary_tensor is 2D (batch_size, num_bits)
    if binary_tensor.dim() == 1:
        binary_tensor = binary_tensor.unsqueeze(0)

    powers = torch.arange(binary_tensor.size(-1), device=binary_tensor.device).flip(0)
    multiplier = torch.pow(2, powers).float()
    binary_tensor = binary_tensor.float()

    decimal = torch.matmul(binary_tensor, multiplier)

    # If the original input was 1D, return a scalar tensor
    if decimal.size(0) == 1:
        return decimal.squeeze()

    return decimal


class BinaryNumberDataset(Dataset):
    """
    A PyTorch Dataset that generates decimal/binary number pairs.
    """

    def __init__(self, bits):
        self.bits = bits

    def __len__(self):
        return 2**self.bits - 1

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        # Generate a random binary number
        binary_num = torch.randint(0, 2, (self.bits,)).float()
        # decimal_num = binary_to_decimal(binary_num)
        return (binary_num, binary_num)


def make_binary_dataloader(n_bits: int, batch_size: int) -> DataLoader:
    """
    Create a dataloader of binary numbers, for training.

    Args:
        n_bits (int): Number of bits in the binary numbers
        batch_size (int): The number of binary/decimal pairs to give in a batch.

    """
    dataset = BinaryNumberDataset(n_bits)
    return DataLoader(dataset, batch_size, shuffle=True)


class TestBinaryToDecimalConversion(unittest.TestCase):

    def test_binary_to_decimal_1D(self):
        """Test the binary_to_decimal function with a 1D tensor."""
        binary_tensor = torch.tensor([1, 0, 1, 1])  # Binary for 11
        expected_decimal = torch.tensor(11.0)
        decimal_tensor = binary_to_decimal(binary_tensor)
        self.assertTrue(torch.equal(decimal_tensor, expected_decimal))

    def test_binary_to_decimal_2D(self):
        """Test the binary_to_decimal function with a 2D tensor."""
        binary_tensor = torch.tensor(
            [[1, 0, 1, 1], [1, 1, 0, 1]]
        )  # Binaries for 11 and 13
        expected_decimal = torch.tensor([11.0, 13.0])
        decimal_tensor = binary_to_decimal(binary_tensor)
        self.assertTrue(torch.all(torch.eq(decimal_tensor, expected_decimal)))


class TestBinaryNumberDataset(unittest.TestCase):

    def test_getitem(self):
        """Test the __getitem__ method of BinaryNumberDataset."""
        dataset = BinaryNumberDataset(bits=4)
        for _ in range(10):  # Test with 10 random samples
            idx = torch.randint(0, 2**4 - 1, (1,)).item()
            decimal_num, binary_num = dataset[idx]
            # Convert binary_num back to decimal to test correctness
            converted_decimal = binary_to_decimal(binary_num.unsqueeze(0)).item()
            self.assertEqual(decimal_num.item(), converted_decimal)


if __name__ == "__main__":
    unittest.main()
