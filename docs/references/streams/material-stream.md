# Material Stream

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

## Example
!!! note
    `feed(molecules=molecules, flowrates=flowrates, **kwargs)` must be used to create/update `data`.

```python
from ml4pd import components
from ml4pd.streams import MaterialStream
from ml4pd.aspen_unit_ops import Distillation

components.set_components(["water", "ethanol", "acetone"])
molecules = {"name_A": ["water", "acetone"], "name_B": ["ethanol", "water"]}
flowrates = {"flowrate_A": [0.5, 0.3], "flowrate_B": [0.7, 0.7]}

feed = MaterialStream(pressure=3, vapor_fraction=0)(molecules=molecules, flowrates=flowrates)
```


