"""Predict the aspen output of a distillation column."""

import warnings
from dataclasses import field
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
from ml4pd.aspen_units.unit import UnitOp
from ml4pd.streams import MaterialStream
from ml4pd.aspen_units.utils import get_model_fname, relu, load_models
from ml4pd.streams.utils import update_material_streams
from ml4pd.utils import timer

try:
    from ml4pd_models import distillation_models
except ModuleNotFoundError:
    print("Couldn't find ml4pd_models. Check https://github.com/NREL/ml4pd_models.")

from pydantic import validate_arguments
from pydantic.dataclasses import dataclass


@dataclass(eq=False)
class Distillation(UnitOp):
    """
    Emulates Aspen's RadFrac column. It gathers data from feed stream,
    and combine additional unit-specific input to make predictions on
    duties and bottom & distillate streams.

    ```python
    Distillation(
        no_stages: Union[int, List[int]] None
        pressure: Union[float, List[float]] = None
        dist_rate: Union[float, List[float]] = None
        reflux_ratio: Union[float, List[float]] = None
        bott_rate: Union[float, List[float]] = None
        reflux_rate: Union[float, List[float]] = None
        boilup_rate: Union[float, List[float]] = None
        boilup_ratio: Union[float, List[float]] = None
        dist_to_feed_ratio: Union[float, List[float]] = None
        bott_to_feed_ratio: Union[float, List[float]] = None
        condensor_duty: Union[float, List[float]] = None
        reboiler_duty: Union[float, List[float]] = None
        free_water_reflux_ratio: Union[float, List[float]] = None
        feed_stage: Union[int, List[int]] = None
        calc_type: Literal["equil", "rate-based"] = None
        condensor_type: Literal["total", "partial-v", "partial-vl", "none"] = None
        reboiler_type: Literal["kettle", "thermosiphon", "none"] = None
        valid_phases: Literal["vl", "vll", "vl-fwc", "vl-fwns", "vl-dwc", "vl-dwas"] = None
        convergence: Literal["std", "petro/wb", "snil", "azeo", "cryo", "custom"] = None
        feed_stage_convention: Literal["above", "on", "vapor", "liquid"] = None
        rate_units: str = "kmol/hr"
        duty_units: str = "kJ/hr"
        pres_units: str = "atm"
        property_method: Literal["nrtl"] = "nrtl"
        verbose: bool = False
        check_data: bool = True
        object_id: str = None
    )(feed_stream: MaterialStream, **kwargs)
    ```

    ## Data Checks & Manipulations

    - Will raise an error if `no_stages` and `pressure` aren't specified.
    - If `feed_stage` isn't specified, will default to `round(no_stages/2)`.
    Also, if feed_stage > no_stages + 1, it wll trigger an error.
    - User can input the units of the input data via `rate_units`, `duty_units` and `pres_units`.
    The corresponding quantities will be converted to default units.
    - 2 out of operating configurations (`dist_rate`, `reflux_rate`, etc.) must be specified.
    - Input must obey type hints (enforced by Pydantic).

    ## Support for Batch Predictions
    `Distillation` currently has limtied support for broadcasting when

    - The incoming feed stream has 1 row of data while current unit data has multiple.
    - The incoming feed stream has multiple rows of data while the current unit data has 1.

    The best way to to take advantage of batch predictions is to prepare a dataframe of
    input parameteras and input individual columns.

    ## Useful Attributes

    - `object_id`: unique id of object. Can be set during initialization, and will be checked
    against a global registry to prevent duplicates.
    - `before`: list of feed streams.
    - `after`: list of output streams.
    - `data`: a pandas df containing all data used for ML.
    - `condensor_duty`: predicted given input.
    - `reboiler_duty`: predicted given input.
    - `status`: numpy array specifying which row of ML df should be trusted. Similar to how
    Aspen indicates Errors.

    """

    # ------ DO NOT CHANGE THESE VARIABLES. WILL MESS UP ML. ------ #
    no_stages: Union[int, List[int]] = field(default=None, repr=False, metadata={"grp": "num"})
    pressure: Union[float, List[float]] = field(default=None, repr=False, metadata={"grp": "num"})
    dist_rate: Union[float, List[float]] = field(default=None, repr=False, metadata={"grp": "num"})
    reflux_ratio: Union[float, List[float]] = field(default=None, repr=False, metadata={"grp": "num"})
    bott_rate: Union[float, List[float]] = field(default=None, repr=False, metadata={"grp": "num"})
    reflux_rate: Union[float, List[float]] = field(default=None, repr=False, metadata={"grp": "num"})
    boilup_rate: Union[float, List[float]] = field(default=None, repr=False, metadata={"grp": "num"})
    boilup_ratio: Union[float, List[float]] = field(default=None, repr=False, metadata={"grp": "num"})
    dist_to_feed_ratio: Union[float, List[float]] = field(default=None, repr=False, metadata={"grp": "num"})
    bott_to_feed_ratio: Union[float, List[float]] = field(default=None, repr=False, metadata={"grp": "num"})
    condensor_duty: Union[float, List[float]] = field(default=None, repr=False, metadata={"grp": "num"})
    reboiler_duty: Union[float, List[float]] = field(default=None, repr=False, metadata={"grp": "num"})
    free_water_reflux_ratio: Union[float, List[float]] = field(default=None, repr=False, metadata={"grp": "num"})
    feed_stage: Union[int, List[int]] = field(default=None, repr=False, metadata={"grp": "num"})
    calc_type: Literal["equil", "rate-based"] = field(default="equil", metadata={"grp": "str"})
    condensor_type: Literal["total", "partial-v", "partial-vl", "none"] = field(default="total", metadata={"grp": "str"})
    reboiler_type: Literal["kettle", "thermosiphon", "none"] = field(default="kettle", metadata={"grp": "str"})
    valid_phases: Literal["vl", "vll", "vl-fwc", "vl-fwns", "vl-dwc", "vl-dwas"] = field(default="vl", metadata={"grp": "str"})
    convergence: Literal["std", "petro/wb", "snil", "azeo", "cryo", "custom"] = field(default="std", metadata={"grp": "str"})
    feed_stage_convention: Literal["above", "on", "vapor", "liquid"] = field(default="above", metadata={"grp": "str"})
    # ------ DO NOT CHANGE THESE VARIABLES. WILL MESS UP ML. ------ #

    rate_units: str = "kmol/hr"
    duty_units: str = "kJ/hr"
    pres_units: str = "atm"
    property_method: Literal["nrtl"] = "nrtl"

    def __post_init__(self):

        # Add unique id to column and log it in registry.
        Distillation.unit_no += 1
        if self.object_id is None:
            self.object_id = f"D{Distillation.unit_no}"

        self.after: Dict[str, str] = {"bott": None, "dist": None}
        self.model_fname: str = None
        self.fillna = False

        super().__post_init__()

    @validate_arguments
    def __call__(self, feed_stream: MaterialStream, **kwargs) -> Tuple[MaterialStream, MaterialStream]:

        for key, value in kwargs.items():
            if key not in self.__dataclass_fields__:
                raise AttributeError(f"{key} not recognized.")
            setattr(self, key, value)

        if feed_stream.data is not None:

            if self.check_data:
                with timer(verbose=self.verbose, operation="data check", unit=self.object_id):
                    self._check_feed_stage()
                    self._check_num_cols()
                    self._check_units()

            with timer(verbose=self.verbose, operation="data prep", unit=self.object_id):
                self.unit_data = self._format_unit_data()
                self.data = self._combine_unit_and_stream_data(feed_stream)

            with timer(verbose=self.verbose, operation="ML", unit=self.object_id):
                bott_stream, dist_stream = self._predict(feed_stream)
        else:
            bott_stream = MaterialStream(stream_type="bott", before=self.object_id)
            dist_stream = MaterialStream(stream_type="dist", before=self.object_id)

        self._add_to_graph(feed_stream, "distillation")
        self._update_connections(input_stream=feed_stream, output_streams={"bott": bott_stream, "dist": dist_stream})

        return bott_stream, dist_stream

    def _predict(self, feed_stream: MaterialStream) -> Tuple[MaterialStream, MaterialStream]:

        self.model_fname = get_model_fname(module=distillation_models, pattern=f"distillation_{len(feed_stream._suffixes)}_")
        stat_model, flow_model, duty_model, temp_model = load_models(distillation_models, self.model_fname)

        # Prepare data
        x = self.get_input_data()

        # Get predictions
        stat = stat_model.predict(x)
        flow = flow_model.predict(x)
        duty = duty_model.predict(x)
        temp = temp_model.predict(x)

        # Get flowrates
        bott_flow_perc = relu(flow, max_value=1.0)
        bott_flow = feed_stream.flow * bott_flow_perc
        dist_flow = feed_stream.flow - bott_flow

        # Turn flowrates into dictionaries with correct keys.
        col_dict = dict(
            zip(
                feed_stream._flow_columns,
                [f"flowrate_{suffix}" for suffix in feed_stream._suffixes],
            )
        )
        bott_flow = bott_flow.rename(columns=col_dict).to_dict("list")
        dist_flow = dist_flow.rename(columns=col_dict).to_dict("list")

        # Set heat & status for column.
        self.condensor_duty = (-relu(-duty[:, 0]) * feed_stream.flow_sum).tolist()
        self.reboiler_duty = (relu(duty[:, 1]) * feed_stream.flow_sum).tolist()
        if feed_stream.status is not None:
            self.status = stat * feed_stream.status
        else:
            self.status = stat

        # Prepare bottom & distillate stream data.
        bott_data = {
            "vapor_fraction": 0,
            "temperature": temp[:, 1].tolist(),
            "molecules": feed_stream.data[feed_stream._name_columns].to_dict("list"),
            "flowrates": bott_flow,
        }

        dist_data = {
            "vapor_fraction": 0,
            "temperature": temp[:, 0].tolist(),
            "molecules": feed_stream.data[feed_stream._name_columns].to_dict("list"),
            "flowrates": dist_flow,
        }

        if None not in self.after.values():
            bott_stream, dist_stream = update_material_streams(streams={self.after["bott"]: bott_data, self.after["dist"]: dist_data})

        else:
            bott_stream = MaterialStream(stream_type="bott")(**bott_data, before=self.object_id)
            dist_stream = MaterialStream(stream_type="dist")(**dist_data, before=self.object_id)

        bott_stream.status = self.status
        dist_stream.status = self.status

        return bott_stream, dist_stream

    @validate_arguments
    def get_loaded_model(self, model_type: Literal["stat", "flow", "duty", "temp"]):
        """Get underlying ML model for further analysis

        Args:
            model_type (str): can be 'stat', 'flow', 'duty', or 'temp'

        Returns:
            model: ML model for a specific variable.
        """
        stat_model, flow_model, duty_model, temp_model = load_models(distillation_models, self.model_fname)

        if model_type == "stat":
            return stat_model
        elif model_type == "flow":
            return flow_model
        elif model_type == "duty":
            return duty_model
        elif model_type == "temp":
            return temp_model

    def _check_feed_stage(self):
        """Make sure feed stage exists and makes sense.

        Raises:
            ValueError: if feed stage larger than # of stages.
        """

        if self.feed_stage is None:
            warnings.warn("feed_stage not specified, setting it equal to half no_stages.", stacklevel=2)
            if isinstance(self.no_stages, list):
                self.feed_stage = list((np.array(self.no_stages) / 2).round())
            elif isinstance(self.no_stages, int):
                self.feed_stage = round(self.no_stages / 2)

        if isinstance(self.feed_stage, int) and isinstance(self.no_stages, int):
            if not 1 < self.feed_stage < self.no_stages + 1:
                raise ValueError("Feed stage must be between 1 and no_stages + 1.")
        else:
            if not all(np.array(self.feed_stage) <= np.array(self.no_stages) + 1):
                raise ValueError("Feed stage can't be lager than number of stages.")
            if not all(1 <= np.array(self.feed_stage)):
                raise ValueError("Feed stage must be >= 1.")

    def _check_num_cols(self):
        """Operating options must not be redundantly/ambiguously specified."""

        unit_num_cols = []
        for key in self.__dataclass_fields__:
            if "grp" in self.__dataclass_fields__[key].metadata:
                group = self.__dataclass_fields__[key].metadata["grp"]
                if group == "num":
                    unit_num_cols.append(key)

        if None in [self.no_stages, self.pressure]:
            raise ValueError("Please specify # of stages & pressure.")

        operating_options = list({col: getattr(self, col) for col in unit_num_cols}.values())
        if sum(np.array(operating_options, dtype="object") != None) != 5:
            raise ValueError("Please specify exactly 2 out of operating options.")

        if self.reboiler_duty is not None or self.condensor_duty is not None:
            raise ValueError("Distillation doesn't support inputting reboiler/condensor duty currently.")
