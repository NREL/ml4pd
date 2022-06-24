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

