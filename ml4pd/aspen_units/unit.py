"""Base class for unit ops."""

import warnings
from abc import ABC, abstractmethod
from dataclasses import field
from typing import Any, ClassVar, Dict, List

import networkx
import numpy as np
import pandas as pd
from pint import UnitRegistry
from pydantic.dataclasses import dataclass

from ml4pd import registry
from ml4pd.streams.stream import Stream

ureg = UnitRegistry()


@dataclass
class UnitOp(ABC):
    """Base class for aspen unit ops."""

    unit_no: ClassVar[int] = -1
    base_type: ClassVar[str] = "unit-op"
    object_id: str = None
    before: Any = field(default_factory=list)
    after: Any = field(default_factory=dict)
    check_data: bool = True
    verbose: bool = False
    fillna: bool = True
    na_value: float = 0

    def __post_init__(self):

        self.status: np.ndarray = None
        self.unit_data: pd.DataFrame = None
        self.data: pd.DataFrame = None
        self.networkx: networkx.DiGraph = None
        self._quant_with_units: Dict[str, Dict[str, List[str]]] = {}

        # Fill up quants with units:
        for key in self.__dataclass_fields__:
            if "_units" in key:
                self._quant_with_units[key] = {self.__dataclass_fields__[key].default: []}

        for key in self.__dataclass_fields__:
            if "grp" in self.__dataclass_fields__[key].metadata:
                for pattern in ["rate", "duty", "pres", "temp"]:
                    if pattern in key:
                        unit = list(self._quant_with_units[f"{pattern}_units"].keys())[0]
                        self._quant_with_units[f"{pattern}_units"][unit].append(key)

        registry.add_element(self)

    @abstractmethod
    def __call__(self, **kwds: Any) -> Any:
        """Combine streams & unit data, then call _predict."""

    @abstractmethod
    def _predict(self, **kwds: Any) -> Any:
        """Produce unit op results and create product streams."""

    def __eq__(self, other: object) -> bool:
        """Compare unit ops based on its ML feature."""
        return all((self.unit_data.to_numpy() == other.unit_data.to_numpy()).flatten())

    def _update_connections(self, input_stream: Stream, output_streams: Dict[str, Stream]):
        """Add unit's object_id to input & output streams."""

        if input_stream.object_id not in self.before:
            self.before.append(input_stream.object_id)
            input_stream.after = self.object_id

        for stream_type, stream in output_streams.items():
            self.after[stream_type] = stream.object_id

    def _add_to_graph(self, feed_stream: Stream, unit_type: str):
        """Update networkx with feed stream and unit."""

        if feed_stream.before is None and len(self.before) == 0:
            self.networkx = networkx.DiGraph()

        if feed_stream.before is None:
            self.networkx.add_node(f"{feed_stream.object_id}_input")
            self.networkx.add_edge(f"{feed_stream.object_id}_input", self.object_id, label=feed_stream.object_id)
        else:
            self.networkx = getattr(registry.get_element(feed_stream.before), "networkx")
            self.networkx.add_edge(feed_stream.before, self.object_id, label=feed_stream.object_id)

        self.networkx.add_node(self.object_id, type=unit_type)

    def _check_units(self):
        """
        Loop through _quant_with_units dict and convert user specified quantities
        to default units to conform to training data. Will throw an error for
        more 'exotic' units. In that case, user will have to manually convert units.
        """

        for unit_type, default_and_quants in self._quant_with_units.items():
            default_unit = list(default_and_quants.keys())[0]
            user_unit = getattr(self, unit_type)
            if user_unit != default_unit:
                quants = default_and_quants[default_unit]
                for quant in quants:
                    user_quant = getattr(self, quant)
                    if user_quant not in [None, np.nan]:
                        default_quant = ureg.Quantity(user_quant, ureg(user_unit)).to(default_unit).magnitude
                        if isinstance(default_quant, np.ndarray):
                            setattr(self, quant, default_quant.tolist())
                        else:
                            setattr(self, quant, default_quant)
                setattr(self, unit_type, default_unit)

    def _format_unit_data(self) -> pd.DataFrame:
        """Collect unit data into dataframe."""

        unit_data = {}
        for key in self.__dataclass_fields__:
            if "grp" in self.__dataclass_fields__[key].metadata:
                group = self.__dataclass_fields__[key].metadata["grp"]
                if group == "str":
                    unit_data[f"{self.object_id}_{key}"] = self.__annotations__[key].__args__.index(getattr(self, key))
                elif group == "num":
                    unit_data[f"{self.object_id}_{key}"] = getattr(self, key)

        try:
            unit_data = pd.DataFrame(unit_data).fillna(np.nan)
        except ValueError:
            unit_data = pd.DataFrame(unit_data, index=[0])

        return unit_data

    def _combine_unit_and_stream_data(self, feed) -> pd.DataFrame:
        """Concat feed & unit data. If they are not of compatible length, raise error."""

        if len(self.unit_data) == len(feed.data):
            data = pd.concat([self.unit_data, feed.data], axis=1)

        elif len(self.unit_data) == 1 and len(feed.data) != 1:
            warnings.warn("Stream and unit data shapes don't exactly match so broadcasting will be used.", stacklevel=2)
            data = pd.concat([self.unit_data.copy(deep=True)] * len(feed.data), ignore_index=True)
            data = pd.concat([data, feed.data], axis=1)

        elif len(feed.data) == 1 and len(self.unit_data) != 1:
            warnings.warn("Stream and unit data shapes don't exactly match so broadcasting will be used.", stacklevel=2)
            data = pd.concat([feed.data.copy(deep=True)] * len(self.unit_data), ignore_index=True)
            data = pd.concat([data, self.unit_data], axis=1)

        else:
            raise ValueError("Data shape doesn't match input stream.")

        return data

    def clear_data(self):
        """Clear user-defined data."""

        # Delete ml input data
        for key in self.__dataclass_fields__:
            if "grp" in self.__dataclass_fields__[key].metadata:
                delattr(self, key)

        # Delete other data
        delattr(self, "status")
        delattr(self, "unit_data")
        delattr(self, "data")

    def get_input_data(self) -> pd.DataFrame:
        """Get data used for ML.

        Returns:
            x: pd.DataFrame of input data.
        """

        if self.fillna:
            x = self.data.fillna(self.na_value).select_dtypes(exclude="object")
        else:
            x = self.data.select_dtypes(exclude="object")
        return x