"""Helper functions for unit ops."""

import importlib.resources
from types import ModuleType
from typing import Union
import numpy as np


def relu(array: np.ndarray, max_value: Union[float, int]) -> np.ndarray:
    """numpy version of tf.ReLU(max_value)

    Args:
        array (np.ndarray): array to apply relu to.
        max_value (Union[float, int]): max value of resulting array.

    Returns:
        np.ndarray: array with same dimentions at array but values are between (0, max_value).
    """
    array = np.maximum(array, np.full(shape=array.shape, fill_value=0))
    array = np.minimum(array, np.full(shape=array.shape, fill_value=max_value))
    return array


def get_model(module: ModuleType, pattern: str, ftype: str = "pkl") -> str:
    """Get ml model from a module.

    Args:
        module (ModuleType): module that contains ml models.
        pattern (str): pick a particular model.
        ftype (str, optional): suffix of model fname. Default is "pkl".

    Returns:
        str: fname of model.
    """

    model_fnames = [fname for fname in list(importlib.resources.contents(module)) if ftype in fname]
    model_fname = [fname for fname in model_fnames if pattern in fname][0]

    return model_fname
