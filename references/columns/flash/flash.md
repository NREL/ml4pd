# Flash

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


