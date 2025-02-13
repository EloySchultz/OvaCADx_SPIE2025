from typing import Union

import torch
import numpy as np

from monai.transforms import Transform


class Einsum(Transform):
    def __init__(self, equation: str):

        self.equation = equation

    def __call__(self, input: Union[np.ndarray, torch.Tensor]):

        if isinstance(input, np.ndarray):
            return np.einsum(self.equation, input)
        elif isinstance(input, torch.Tensor):
            return torch.einsum(self.equation, input)
        else:
            raise ValueError("Unsupported input type. Only numpy.ndarray and torch.Tensor are supported.")
