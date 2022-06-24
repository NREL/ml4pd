# ML4PD

ML4PD is an open-source machine learning Python library that aims to speed up process modeling in Aspen.

## Installation

=== "pip"

    ```bash
    pip install ml4pd git+https://github.com/NREL/ml4pd_models.git@v1
    ```

    then download [graphviz](https://graphviz.org/download/) for visualization.


=== "development"
    ```
    git clone https://github.com/NREL/ml4pd.git
    git clone https://github.com/NREL/ml4pd_models.git
    mamba env create -f ml4pd/environment.yml
    conda develop ml4pd
    conda develop ml4pd_models
    ```

