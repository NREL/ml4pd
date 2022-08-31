"""Simulate aspen material streams."""

import warnings
from dataclasses import field
from functools import partial
from typing import Dict, List, Literal, Union

import numpy as np
import pandas as pd
from pint import UnitRegistry
from pydantic import validate_arguments
from pydantic.dataclasses import dataclass

from ml4pd import components
from ml4pd.streams.stream import Stream
from ml4pd.utils import timer

ureg = UnitRegistry()

__all__ = ["MaterialStream"]


@dataclass(eq=False)
class MaterialStream(Stream):
    """
    Emulates an Aspen material stream. Its primary purpose is to get **molecular**,
    **flowrate** and **state data** from user and perform data checks before passing this
    information along to other unit ops.

    ```python

    MaterialStream(
        temperature: Union[float, List[float]] = None
        vapor_fraction: Union[float, List[float]] = None
        pressure: Union[float, List[float]] = None
        stream_type: Literal["feed", "bott", "dist", "vapor", "liquid"] = "feed"
        pressure_units: str = "atm"
        flowrates_units: str = "kmol/hr"
        temperature_units: Literal["degC", "degF", "degK", "degR"] = "degC"
        check_data: bool = True
        verbose: bool = False
        object_id: str = None
        before: str = None
        after: str = None
    )(
        molecules: Dict[str, Union[str, List[str]]],
        flowrates: Dict[str, Union[float, List[float]]],
        **kwargs,
    )

    ```

    ## Data Checks & Manipulations

    - 2 out of `temperature`, `vapor_fraction`, and `temperature` must be specified.
    - User can input the units of the input data via `flowrate_units`, `temperature_units`,
    and `pressure_units`. `MaterialStream` will then convert the quantities to the
    default units so that they agree with training data.
    - For `flowrates`, errors will be raised if there are `NaN`, or negative values. Also,
    keys to this dictionary must be of `f"flowrate_{suffix}"` format.
    - For  `molecules,` errors will be raised if there are NaN or the specified
    molecules are not found in `components`. Also, keys to this dictionary must be of
    `f"name_{suffix}"` format.
    - For each row, sort `molecules` & `flowrates` so that the lightest molecules are to the left.
    - Input must obey type hints (which is enforced by Pydantic).

    ## Support for Batch Predictions
    `MaterialStream` currently has limited support for broadcasting when

    - there is 1 set of `molecules` and multiple sets of `flowrates`.
    - there are n sets of `molecules` and `flowrates` and 1 set of state variables.

    The best way to to take advantage of batch predictions is to prepare a dataframe of
    input parameters and input individual columns.

    ## Useful Attributes
    - `object_id`: unique id of object. Automatically generated but can be set during
    initialization, and will be checked against a global registry to prevent duplicates.
    - `before`: id of the unit that produced this stream.
    - `after`: id of the unit that this stream feeds into.
    - `data`: a pandas df containing all data used for ML.
    - `temperature`: predicted if instance is produced by a unit operation.
    - `flow`: predicted if instance is produced by a unit operation.

    ## Vapor Fraction/Temperature/Pressure
    - To reduce the number of ML models needed, `MaterialStream` created by ML will only predict
    temperature & vapor fraction since only 2 of 3 variables are needed to fully specify the stream.
    - User-created `MaterialStream` currently only suspports vapor fraction & temperature or
    vapor fraction & pressure. The range of these variables depend on the unit op the stream feeds to.

    """

    # ------ DO NOT CHANGE THE ORDER OF THESE VARIABLES. WILL MESS UP ML. ------ #
    temperature: Union[float, List[float]] = field(default=None, repr=False)
    vapor_fraction: Union[float, List[float]] = field(default=None, repr=False)
    pressure: Union[float, List[float]] = field(default=None, repr=False)
    # ------ DO NOT CHANGE THE ORDER OF THESE VARIABLES. WILL MESS UP ML. ------ #

    stream_type: Literal["feed", "bott", "dist", "liquid", "vapor"] = "feed"
    pressure_units: str = "atm"
    flowrates_units: str = "kmol/hr"
    temperature_units: Literal["degC", "degF", "degK", "degR"] = "degC"

    def __post_init__(self):
        """
        Ensure every newly instantiated stream has a unique identifier &
        add instance to a registry to later be used to graph modelling.
        """

        self.flow: pd.DataFrame = None
        self.flow_norm: pd.DataFrame = None
        self.flow_sum: pd.Series = None
        self._flow_columns: List[str] = []
        self._state_columns: List[str] = []
        self._suffixes: List[str] = None
        self._name_columns: List[str] = None
        self._mw_idx: np.ndarray = None

        MaterialStream.unit_no += 1
        if self.object_id is None:
            self.object_id = f"{self.stream_type}{MaterialStream.unit_no}"

        super().__post_init__()

    @validate_arguments
    def __call__(
        self,
        molecules: Dict[str, Union[str, List[str]]],
        flowrates: Dict[str, Union[float, List[float]]],
        **kwargs,
    ):
        """
        Get molecules & flowrates data then assemble df after multiple checks.

        Args:
            molecules (Dict[str, Union[str, List[str]]]): dictionary of
                molecules w/ 'name_{suffix}' format for columns.
            flowrates (Dict[str, Union[float, List[float]]]): dictionary of
                corresponding flowrates w/ 'flowrate_{suffix}' format for columns.

        Raises:
            AttributeError: if keys in kwargs aren't in __init__.

        Returns:
            MaterialStream: current instance to be fed into columns.
        """

        for key, value in kwargs.items():
            if key not in self.__dataclass_fields__:
                raise AttributeError(f"{key} not recognized.")
            setattr(self, key, value)

        try:
            self.data = pd.DataFrame(molecules)
        except ValueError:
            self.data = pd.DataFrame({key: [value] for key, value in molecules.items()})

        self._name_columns = self.data.columns.to_list()
        self._suffixes = [col.split("_")[-1] for col in self.data.columns]

        try:
            self.flow = pd.DataFrame(flowrates)
        except ValueError:
            self.flow = pd.DataFrame({key: [value] for key, value in flowrates.items()})

        if self.check_data:
            with timer(verbose=self.verbose, operation="data check", unit=self.object_id) as _:
                self._check_state_variables()
                self._check_prefix()
                self._check_suffix()
                self._check_flowrates()
                self._check_molecules()
                self._convert_units()

        with timer(verbose=self.verbose, operation="data prep", unit=self.object_id) as _:
            self.flow_sum = self.flow.sum(axis=1)
            self.flow_norm = self.flow.div(self.flow_sum, axis=0)
            self._broadcast_molecules_and_flowrates()
            self._sort_molecules_by_weight()
            self._add_mol_data()
            self._add_flowrates_and_state_variables()
            self._add_comp_no()

        return self

    def clear_data(self):
        """Clear user-defined data. Most useful for flowsheets."""

        # Delete state variables
        delattr(self, "temperature")
        delattr(self, "vapor_fraction")
        delattr(self, "pressure")

        # Delete data and flow
        delattr(self, "data")
        delattr(self, "flow")

        # Delete private variables
        delattr(self, "_suffixes")
        delattr(self, "_flow_columns")
        delattr(self, "_state_columns")
        delattr(self, "_mw_idx")

    def _sort_molecules_by_weight(self):
        """
        Sort molecular names & flowrates by molecular weights.
        It will also set the '_mw_idx' attribute to later be used on training
        data's output.
        """

        data_copy = self.data.copy(deep=True)

        # Get molecular weights to sort
        molecular_weight = components.data[["name", "MolWt"]]
        molecular_weight_columns = []
        name_columns = []
        for suffix in self._suffixes:
            molecular_weight_columns.append(f"MolWt_{suffix}")
            name_columns.append(f"name_{suffix}")
            mol_data = molecular_weight.rename(columns={"name": f"name_{suffix}", "MolWt": f"MolWt_{suffix}"})
            data_copy = pd.merge(data_copy, mol_data, how="left")

        # Sort molecules and flowrates by sorted indices of molecular weights
        molecular_weight_idx = data_copy[molecular_weight_columns].to_numpy().argsort()
        data_copy = np.take_along_axis(data_copy[name_columns].to_numpy(), molecular_weight_idx, axis=1)
        flowrates = np.take_along_axis(self.flow.to_numpy(), molecular_weight_idx, axis=1)
        flowrates_norm = np.take_along_axis(self.flow_norm.to_numpy(), molecular_weight_idx, axis=1)

        # Replace input data with sorted data
        self.data = pd.DataFrame(data_copy, columns=self.data.columns)
        self.flow = pd.DataFrame(flowrates, columns=self.flow.columns)
        self.flow_norm = pd.DataFrame(flowrates_norm, columns=self.flow_norm.columns)

        # Add idx to instance for other uses like sorting y-values during training
        self._mw_idx = molecular_weight_idx

    def _convert_units(self):
        """Convert units of pressure, temperature, and flowrates into default ones."""

        if self.pressure_units != "atm" and self.pressure is not None:
            self.pressure = (self.pressure * ureg(self.pressure_units)).to("atm").magnitude
            self.pressure_units = "atm"

        if self.temperature_units != "degC" and self.temperature is not None:
            original_quant = ureg.Quantity(self.temperature, self.temperature_units)
            self.temperature = original_quant.to("degC").magnitude
            self.temperature_units = "C"

        if self.flowrates_units != "kmol/hr" and self.flow is not None:
            original_quant = self.flow.to_numpy() * ureg(self.flowrates_units)
            default_quant = original_quant.to("kmol/hr").magnitude
            self.flow = pd.DataFrame(default_quant, columns=self.flow.columns)
            self.flowrates_units = "kmol/hr"

    def _broadcast_molecules_and_flowrates(self):
        """Attempt to match the shapes of molecules & flowrates.

        Raises:
            ValueError: if len(data) != len(flow) and neither is 1.
        """

        if len(self.data) == 1 and len(self.flow_norm) != 1:
            warnings.warn("will broadcast molecules to flowrates.", stacklevel=2)
            self.data = pd.concat([self.data] * len(self.flow_norm), ignore_index=True)
        elif len(self.data) != 1 and len(self.flow_norm) == 1:
            warnings.warn("will broadcast flowrates to molecules.", stacklevel=2)
            self.flow_norm = pd.concat([self.flow_norm] * len(self.data), ignore_index=True)
        elif len(self.flow_norm) != len(self.flow_norm):
            raise ValueError("Molecules & Flowrates shape don't match.")

    def _add_flowrates_and_state_variables(self):
        """Add flowrates & state variables to data, with some support for broadcasting.

        Raises:
            ValueError: if data & flowrates are not of the right shape.
        """

        for suffix in self._suffixes:
            self.data[f"{self.object_id}_flowrate_{suffix}"] = self.flow_norm[f"flowrate_{suffix}"]
            self._flow_columns.append(f"{self.object_id}_flowrate_{suffix}")

        for col in ["temperature", "vapor_fraction", "pressure"]:
            if getattr(self, col) is None:
                self.data[f"{self.object_id}_{col}"] = np.nan
            else:
                self.data[f"{self.object_id}_{col}"] = getattr(self, col)
            self._state_columns.append(f"{self.object_id}_{col}")

    def _add_comp_no(self):
        """Add number of molecular types in a stream."""

        self.data[f"{self.object_id}_comp_no"] = (self.data[self._flow_columns] != 0.0).sum(axis=1)

    def _add_mol_data(self):
        """Get molecular data from components object and add them self.data."""

        def add_suffix(col, suf):
            return col + suf

        for suffix in self._suffixes:
            mol_data = components.data.rename(mapper=partial(add_suffix, suf=f"_{suffix}"), axis="columns")
            self.data = pd.merge(self.data, mol_data, how="left")

    def _check_prefix(self):
        """Check prefixes of molecules & flowrates dictionaries.

        Raises:
            ValueError: if prefix of molecules isn't 'name.'
            ValueError: if prefix of flowrates isn't 'flowrate.'
        """

        molecules_prefix = [col.split("_")[0] for col in self.data.columns]
        flowrates_prefix = [col.split("_")[0] for col in self.flow.columns]

        if (len(set(molecules_prefix)) != 1) or ("name" not in set(molecules_prefix)):
            raise ValueError("Only 1 allowed prefix for molecules columns: name.")
        if (len(set(flowrates_prefix)) != 1) or ("flowrate" not in set(flowrates_prefix)):
            raise ValueError("Only 1 allowed prefix for flowrates columns: flowrate.")

    def _check_suffix(self):
        """Check suffixes of molecules  flowrates dictionaries.

        Raises:
            ValueError: if suffixes in molecules aren't unique.
            ValueError: if suffixes in flowrates aren't unique.
            ValueError: if suffixes in flowrates & molecules don't agree.
        """

        molecules_suffix = [col.split("_")[-1] for col in self.data.columns]
        flowrates_suffix = [col.split("_")[-1] for col in self.flow.columns]

        if len(set(molecules_suffix)) != len(molecules_suffix):
            raise ValueError("Suffixes in molecules must be unique.")
        if len(set(flowrates_suffix)) != len(flowrates_suffix):
            raise ValueError("Suffixes in flowrates must be unique.")
        if set(flowrates_suffix) != set(molecules_suffix):
            raise ValueError("Suffixes in molecules & flowrates must agree.")

    def _check_state_variables(self):
        """Check that state variables make sense.

        Raises:
            ValueError: If more or less than 2 of them are specified.
        """

        state_variables = [self.temperature, self.vapor_fraction, self.pressure]
        if sum([var is not None for var in state_variables]) != 2:
            raise ValueError("Expected exactly 2 out of state variables.")

    def _check_flowrates(self):
        """Check that flowrates make sense and normalize them if specified by user.

        Raises:
            ValueError: if there are NaN in flowrates.
            ValueError: if there are flowrates smaller than 0.
        """

        if self.flow.isna().sum().sum() != 0:
            raise ValueError("Found NaN in flowrates.")

        if (self.flow < 0).sum().sum() > 0:
            raise ValueError("flowrates cannot be smaller than 0.")

    def _check_molecules(self):
        """Check that molecules make sense.

        Raises:
            ValueError: if there are NaN in molecules.
            ValueError: if molecules aren't pre-specified in components.
        """

        if self.data.isna().sum().sum() != 0:
            raise ValueError("Found NaN in molecules.")

        unique_molecules = set(self.data.to_numpy().flatten())
        if len(unique_molecules - set(components.data["name"])) != 0:
            print(set(components.data["name"]))
            raise ValueError(f"{unique_molecules -  set(components.data['name'])} not in components.")
