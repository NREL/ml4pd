# RadFrac

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

## Example
```python
from ml4pd import components
from ml4pd.streams import MaterialStream
from ml4pd.aspen_unit_ops import Distillation

components.set_components(["water", "ethanol", "acetone"])
molecules = {"name_A": ["water", "acetone"], "name_B": ["ethanol", "water"]}
flowrates = {"flowrate_A": [0.5, 0.3], "flowrate_B": [0.7, 0.7]}

feed = MaterialStream(pressure=3, vapor_fraction=0)
dist_col = Distillation(no_stages=40, feed_stage=10, boilup_ratio=0.5, reflux_ratio=0.5)

feed = feed(molecules=molecules, flowrates=flowrates)
bott, dist = dist_col(feed, pressure=3)
```


