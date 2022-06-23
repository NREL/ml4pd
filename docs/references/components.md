# Components

Object to specify molecules in the system and pass that information along to streams and unit ops.

## Useful Attributes:

- `data`: a df where each row is a molecule and each column is a feature.
- `__repr__`: table where each row is a specified molecule, and each column is a type of identifier.
    Useful for checking that `components` got the right molecules.

## Example
```python
from ml4pd import components
components.set_components(["acetone", "water", "ethanol"])
```

