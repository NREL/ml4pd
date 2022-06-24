import itertools
import random
from abc import ABC, abstractmethod
from typing import List, Literal, Union

import numpy as np
import pandas as pd
from pydantic import validate_arguments
from pydantic.dataclasses import dataclass


@dataclass
class Dimension(ABC):
    """
    Base class for objects used to generate data samples.

    ## Parameters:
    - samples: The # of rows of generated list.
    - group: The # of columns of generated list.
    - sampling: sampling scheme. Depends on type of dimension.

    """

    samples: int = None
    sampling: str = None
    group: int = None

    @abstractmethod
    def generate(self):
        """Generate one/multiple values depending on samples."""
        pass


@dataclass
class Integer(Dimension):
    """Generates one / a list of integers based on specified sampling method."""

    min_value: int = None
    max_value: int = None
    samples: int = None
    sampling: Literal["random-uniform", "linear"] = "linear"

    def __post_init__(self):
        self.values: List[int] = None

    def generate(self):
        if self.sampling == "random-uniform":
            values = self.generate_random_uniform_values()
        elif self.sampling == "linear":
            values = self.generate_linear_values()

        if len(values) == 1:
            values = values.round(5)[0]
        else:
            values = values.round(5).tolist()
        return values

    def generate_random_uniform_values(self):
        return np.random.randint(low=self.min_value, high=self.max_value + 1, size=self.samples)

    def generate_linear_values(self):
        return np.linspace(start=self.min_value, stop=self.max_value, num=self.samples, dtype=int)


@dataclass
class Float(Dimension):

    min_value: Union[float, int] = None
    max_value: Union[float, int] = None
    samples: int = None
    sampling: Literal["random-uniform", "linear", "log"] = "linear"

    def generate(self):
        if self.sampling == "random-uniform":
            values = self.generate_random_uniform_values()
        elif self.sampling == "linear":
            values = self.generate_linear_values()
        elif self.sampling == "log":
            values = self.generate_log_values()

        if len(values) == 1:
            values = values.round(5)[0]
        else:
            values = values.round(5).tolist()
        return values

    def generate_random_uniform_values(self):
        return np.random.uniform(low=self.min_value, high=self.max_value, size=self.samples)

    def generate_linear_values(self):
        return np.linspace(start=self.min_value, stop=self.max_value, num=self.samples, dtype=float)

    def generate_log_values(self):
        return np.logspace(np.log10(self.min_value), np.log10(self.max_value), self.samples, dtype=float)


@dataclass
class Choice(Dimension):

    choices: list = None
    group: int = 1
    samples: int = None

    def generate(self):
        values = []
        for _ in range(self.samples):
            values.append(random.sample(self.choices, self.group))

        if self.group == 1:
            values = np.array(values).flatten().tolist()

        return values


@validate_arguments
def generate_grid(**kwargs: Union[Dimension, list, float, int, str]) -> pd.DataFrame:

    search_space = {}
    dtypes = []
    for key, range in kwargs.items():
        if isinstance(range, Dimension):
            values = range.generate()
            if not isinstance(values, list):
                search_space[key] = [values]
            else:
                search_space[key] = range.generate()
        elif isinstance(range, list):
            search_space[key] = range
        else:
            search_space[key] = [range]
        dtypes.append((key, object))

    # Turn into nested list
    search_space = list(search_space.values())

    # Created permutations
    search_space = np.array(list(itertools.product(*search_space)), dtype=dtypes)

    # Turn into df
    df = pd.DataFrame(search_space).infer_objects()

    return df
