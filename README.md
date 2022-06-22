# ml4pd

Process design with Machine Learning.

## How to Set Up for Development

### via Conda
1. Clone repository with `git clone`.
2. Create new env. & install dependencies: `mamba env create -f ml4pd/environment.yml -n [env-name]`
3. Add repo to path with `conda develop ml4pd`.
4. Optional: register conda environment with jupyter notebook `python -m ipykernel install --user --name=ml4pd`

### Additional GitHub repositories for docs, tests and training
- `ml4pd_utils`: code base for generating & preparing data for training.
- `ml4pd_models`: to store model files.
- `autoaspen`: database for data obtained by aspen & python.

### To debug docs
- Use `mkdocs serve` within `ml4pd` directory, then go to `localhost:8000`.
- `generate_site.py` gets docstrings (written in makrdown) from classes and put them in the right directory.
- To make changes to notebooks, add `ml4pd` to path with `sys.path.append()` or `conda develop`.

### Relationship with ml4pd_models
To minimize manual work, ml4pd dependends on a specific ml4pd_models github branch. When changes are made to
either ml4pd or ml4pd_models that will break compatibility, create new branch in ml4pd_models, and link 
requirements.txt with the new branch.