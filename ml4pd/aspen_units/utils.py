"""Helper functions for unit ops."""

import importlib.resources
from types import ModuleType
from typing import Union
import numpy as np
import pickle
from datetime import datetime


def relu(array: np.ndarray, max_value: Union[float, int] = None) -> np.ndarray:
    """numpy version of tf.ReLU(max_value)

    Args:
        array (np.ndarray): array to apply relu to.
        max_value (Union[float, int]): max value of resulting array.

    Returns:
        np.ndarray: array with same dimentions at array but values are between (0, max_value).
    """
    array = np.maximum(array, np.full(shape=array.shape, fill_value=0))
    if max_value:
        array = np.minimum(array, np.full(shape=array.shape, fill_value=max_value))
    return array


def get_model_fname(module: ModuleType, pattern: str, ftype: str = "pkl", date: str = None) -> str:
    """Get ml model from a module.

    Args:
        module (ModuleType): module that contains ml models.
        pattern (str): pick a particular model.
        ftype (str, optional): suffix of model fname. Default is "pkl".
        date (str, optional): date of model. Default is None.

    Returns:
        str: fname of model.
    """

    model_fnames = [fname for fname in list(importlib.resources.contents(module)) if (ftype in fname) and (pattern in fname)]

    if date is None:
        latest_date = "000101"
        lastest_model_fname = None
        for fname in model_fnames:
            segments = fname.split("_")
            for segment in segments:
                try:
                    model_date = datetime.strptime(segment, "%y%m%d")
                    if model_date > datetime.strptime(latest_date, "%y%m%d"):
                        latest_date = segment
                        lastest_model_fname = fname
                        break
                except ValueError:
                    pass
    else:
        lastest_model_fname = [fname for fname in model_fnames if date in fname][0]

    return lastest_model_fname


def load_models(module_name: ModuleType, model_fname: str):

    models = []
    with importlib.resources.path(module_name, model_fname) as model_path:
        with open(model_path, "rb") as model_file:
            while True:
                try:
                    models.append(pickle.load(model_file))
                except EOFError:
                    break
    return models
