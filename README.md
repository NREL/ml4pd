# ml4pd

Process design with Machine Learning.

## How to Install for Users

```
pip install ml4pd git+https://github.com/NREL/ml4pd_models.git@v1
```

## How to Set Up for Development

```   
git clone https://github.com/NREL/ml4pd.git
git clone https://github.com/NREL/ml4pd_models.git
mamba env create -f ml4pd/environment.yml
conda develop ml4pd
conda develop ml4pd_models
```

### Additional GitHub repositories for docs, tests and training
- `ml4pd_utils`: code base for generating & preparing data for training.
- `autoaspen`: database for data obtained by aspen & python.

### To debug docs
- Use `mkdocs serve` within `ml4pd` directory, then go to `localhost:8000`.
- `generate_site.py` gets docstrings (written in makrdown) from classes and put them in the right directory.
- To make changes to notebooks, add `ml4pd` to path with `sys.path.append()` or `conda develop`.

### Relationship with ml4pd_models
To minimize manual work, ml4pd dependends on a specific ml4pd_models github branch. When changes are made to
either ml4pd or ml4pd_models that will break compatibility, create new branch in ml4pd_models, and apply change
to workflow .yml files, this README and index.md in docs.