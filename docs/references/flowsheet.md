# Flowsheet
Conceptual representation of flowsheets in Aspen.

```python
Flowsheet(
    input_streams: List[Stream],
    output_streams: List[Stream],
    object_id: str = None
)
```

!!! note
    `Flowsheet` can only currently handle tree-like process
    models (one input stream with no loops).

## Data Checks
- If passed graph isn't fully connected, an error will be raised.

## Methods

- `Flowsheet.plot_model()`: returns a graphviz digraph object.
- `Flowsheet.run(inputs: Dict[str, dict])`: Perform simulation. See [Example](#example).
- `Flowsheet.clear_data()`: clear both input & output from streams & unit ops.
- `Flowsheet.get_element(object_id: str)`: returns stream or unit op given its object_id.
- `Flowsheet._get_networkx(): returns networkx object that can be manually manipulated for plotting/debugging.

## Example
```python
from ml4pd import components
from ml4pd.flowsheet import Flowsheet
from ml4pd.streams import MaterialStream
from ml4pd.aspen_unit_ops import Distillation

components.set_components(["water", "ethanol", "acetone"])
molecules = {"name_A": ["water", "acetone"], "name_B": ["ethanol", "water"]}
flowrates = {"flowrate_A": [0.5, 0.3], "flowrate_B": [0.7, 0.7]}

feed_stream = MaterialStream(object_id="feed")
dist_column = Distillation(object_id="dist")
bott_stream, dist_stream = dist_column(feed_stream)

inputs = {
    "feed": {
        "vapor_fraction": 0,
        "pressure": [3, 4],
        "molecules": molecules,
        "flowrates": flowrates
    },
    "dist": {
        "no_stages": [5, 6],
        "pressure": [3, 4],
        "reflux_ratio": [1, 2],
        "boilup_ratio": [1, 2],
        "feed_stage": [2, 2]
    }
}

flowsheet = Flowsheet(input_streams=[feed_stream], output_streams=[bott_stream, dist_stream])
flowsheet.run(inputs=inputs)

```

