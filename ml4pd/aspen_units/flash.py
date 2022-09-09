"""Predict the aspen output of a flash column."""

from dataclasses import field
from typing import ClassVar, Dict, List, Literal, Tuple, Union

import numpy as np
from ml4pd.aspen_units.unit import UnitOp
from ml4pd.streams import MaterialStream
from ml4pd.aspen_units.utils import get_model_fname, relu, load_models
from ml4pd.streams.utils import update_material_streams
from ml4pd.utils import timer

try:
    from ml4pd_models import flash_models
except ModuleNotFoundError:
    print("Couldn't find ml4pd_models. Check https://github.com/NREL/ml4pd_models.")
from pydantic import validate_arguments
from pydantic.dataclasses import dataclass


@dataclass(eq=False)
class Flash(UnitOp):
    """
    Emulates Aspen's flash column. It gathers data from feed stream,
    and combine additional unit-specific input to make predictions on
    duties and vapor & liquid streams.

    ```python
    Flash(
        pressure: Union[float, List[float]] = None
        temperature: Union[float, List[float]] = None
        duty: Union[float, List[float]] = None
        vapor_fraction: Union[float, List[float]] = None
        valid_phases: Literal["vl", "vll", "vl-fw", "vl-dw"] = None
        pres_units: str = "atm"
        temp_units: Literal["degC", "degF", "degK", "degR"] = "degC"
        duty_units: str = "kJ/hr"
        property_method: Literal["nrtl"] = "nrtl"
        verbose: bool = False
        check_data: bool = True
        object_id: str = None
    )(feed_stream: MaterialStream, **kwargs)
    ```

    ## Data Checks & Manipulations

    - 2 out of `pressure`, `temperature`, `duty`, `vapor_fraction` must be specified.
    - Will raise an error if pressure drop results in `pressure < 0`.
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
    - `status`: numpy array specifying which row of ML df should be trusted. Similar to how
    Aspen indicates Errors.

    """

    unit_no: ClassVar[int] = -1

    # ------ DO NOT CHANGE THESE VARIABLES. WILL MESS UP ML. ------ #
    pressure: Union[float, List[float]] = field(default=None, repr=False, metadata={"grp": "num"})
    temperature: Union[float, List[float]] = field(default=None, repr=False, metadata={"grp": "num"})
    duty: Union[float, List[float]] = field(default=None, repr=False, metadata={"grp": "num"})
    vapor_fraction: Union[float, List[float]] = field(default=None, repr=False, metadata={"grp": "num"})
    valid_phases: Literal["vl", "vll", "vl-fw", "vl-dw"] = field(default="vl", repr=False, metadata={"grp": "str"})
    # ------ DO NOT CHANGE THESE VARIABLES. WILL MESS UP ML. ------ #

    pres_units: str = "atm"
    temp_units: Literal["degC", "degF", "degK", "degR"] = "degC"
    duty_units: str = "kJ/hr"
    property_method: Literal["nrtl"] = "nrtl"

    def __post_init__(self):

        # Add unique id to column and log it in registry.
        Flash.unit_no += 1
        if self.object_id is None:
            self.object_id = f"F{Flash.unit_no}"

        self.after: Dict[str, str] = {"vapor": None, "liquid": None}
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
                    self._check_redundancy()
                    self._check_pressure(feed_stream)
                    self._check_units()
            with timer(verbose=self.verbose, operation="data prep", unit=self.object_id):
                self._adjust_pressure(feed_stream)
                self.unit_data = self._format_unit_data()
                self.data = self._combine_unit_and_stream_data(feed_stream)

            with timer(verbose=self.verbose, operation="ML", unit=self.object_id):
                vapor_stream, liquid_stream = self._predict(feed_stream)

        else:
            vapor_stream = MaterialStream(stream_type="vapor", before=self.object_id)
            liquid_stream = MaterialStream(stream_type="liquid", before=self.object_id)

        self._add_to_graph(feed_stream, "flash")
        self._update_connections(input_stream=feed_stream, output_streams={"vapor": vapor_stream, "liquid": liquid_stream})

        return vapor_stream, liquid_stream

    def _predict(self, feed_stream: MaterialStream) -> Tuple[MaterialStream, MaterialStream]:

        model_fname = get_model_fname(module=flash_models, pattern=f"flash_{len(feed_stream._suffixes)}_")
        stat_model, flow_model, temp_model = load_models(flash_models, model_fname)

        # Prepare data
        x = self.get_input_data()

        # Get predictions
        stat = stat_model.predict(x)
        flow = flow_model.predict(x)
        temp = temp_model.predict(x)

        # Get flowrates
        liquid_flow_perc = relu(flow, max_value=1.0)
        liquid_flow = feed_stream.flow * liquid_flow_perc
        vapor_flow = feed_stream.flow - liquid_flow

        # Turn flowrates into dictionaries with correct keys.
        col_dict = dict(zip(feed_stream._flow_columns, [f"flowrate_{suffix}" for suffix in feed_stream._suffixes]))
        liquid_flow = liquid_flow.rename(columns=col_dict).to_dict("list")
        vapor_flow = vapor_flow.rename(columns=col_dict).to_dict("list")

        # set status for column
        if feed_stream.status is not None:
            self.status = stat * feed_stream.status
        else:
            self.status = stat

        # Prepare streams
        vapor_data = {
            "vapor_fraction": 1,
            "temperature": temp.mean(axis=1).tolist(),
            "molecules": feed_stream.data[feed_stream._name_columns].to_dict("list"),
            "flowrates": vapor_flow,
        }

        liquid_data = {
            "vapor_fraction": 0,
            "temperature": temp.mean(axis=1).tolist(),
            "molecules": feed_stream.data[feed_stream._name_columns].to_dict("list"),
            "flowrates": liquid_flow,
        }

        if None not in self.after.values():
            vapor_stream, liquid_stream = update_material_streams(streams={self.after["vapor"]: vapor_data, self.after["liquid"]: liquid_data})

        else:
            vapor_stream = MaterialStream(stream_type="vapor")(**vapor_data, before=self.object_id)
            liquid_stream = MaterialStream(stream_type="liquid")(**liquid_data, before=self.object_id)

        vapor_stream.status = self.status
        liquid_stream.status = self.status

        return vapor_stream, liquid_stream

    def _adjust_pressure(self, feed_stream: MaterialStream):
        """If flash pressure is 0, set it equal to feed pressure."""

        if isinstance(self.pressure, list):
            selector = (np.array(self.pressure) == 0) * 1
            new_pressure = self.pressure + (np.array(feed_stream.pressure) * selector)
            self.pressure = new_pressure.tolist()
        elif self.pressure == 0:
            self.pressure = feed_stream.pressure

    def _check_redundancy(self):
        """Only 2 out of 4 numerical variables can be set."""

        input_sum = sum([element is not None for element in [self.pressure, self.temperature, self.duty, self.vapor_fraction]])
        if input_sum != 2:
            raise ValueError("Specify exactly 2 out of pressure, temperature, duty, and vapor fraction.")

    def _check_pressure(self, feed_stream: MaterialStream):
        """Check that pressure drops don't lead to negative pressures."""

        if self.pressure is not None and feed_stream.pressure is not None:
            selector = (np.array(self.pressure) < 0) * 1
            test_flash_pressure = selector * self.pressure
            test_feed_pressure = selector * feed_stream.pressure
            test_pressure = test_feed_pressure + test_flash_pressure
            if (test_pressure < 0).sum() != 0:
                raise ValueError("Specified pressure drop would result in a pressure less than 0.")
