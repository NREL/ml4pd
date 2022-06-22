import pathlib

from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest
from ml4pd import components
from ml4pd.aspen_units import Flash
from ml4pd.streams import MaterialStream
from sklearn.metrics import mean_absolute_error

no_stages = [10, 20]
pressure = [3, 4]
reflux_ratio = [0.3, 0.4]
boilup_ratio = [0.5, 0.6]
feed_stage = [5, 10]

molecules = {"name_A": ["water", "acetone"], "name_B": ["methane", "ethane"]}
flowrates = {"flowrate_A": [0.1, 0.2], "flowrate_B": [0.3, 0.4]}
vapor_fraction = [0.1, 0.2]
feed_pressure = [1, 2]


class TestDataChecks:
    @pytest.mark.parametrize(
        "test_input, expected",
        [
            ({"pressure": [10, 20]}, pytest.raises(ValueError)),
            ({"pressure": [10, 20], "temperature": [30, 40]}, does_not_raise()),
            ({"pressure": [10, 20], "temperature": [30, 40], "duty": [50, 60]}, pytest.raises(ValueError)),
            ({"pressure": [10, 20], "temperature": [30, 40], "duty": [50, 60], "vapor_fraction": [0.1, 0.2]}, pytest.raises(ValueError)),
        ],
    )
    def test_redudancy(self, test_input, expected):

        components.set_components(["water", "acetone", "methane", "ethane"])

        feed_stream = MaterialStream(vapor_fraction=vapor_fraction, pressure=feed_pressure)(molecules=molecules, flowrates=flowrates)
        column = Flash(**test_input)

        with expected:
            _ = column(feed_stream)

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            ([1, 2], [1, 2]),
            ([0, 2], [1, 2]),
        ],
    )
    def test_pressure_adjustment(self, test_input, expected):

        components.set_components(["water", "acetone", "methane", "ethane"])

        feed_stream = MaterialStream(vapor_fraction=vapor_fraction, pressure=feed_pressure)(molecules=molecules, flowrates=flowrates)

        input_dict = {"vapor_fraction": [0.1, 0.2], "pressure": test_input}

        column = Flash(**input_dict)
        _ = column(feed_stream)

        assert column.pressure == expected


class TestML:
    def test_mae_3(self, add_name_columns, rename_flowrate_columns, sort_targets_by_weight):

        fname = "220606_ketone_flash_3_aspen.csv"
        model = "flash"

        here = pathlib.Path(__file__).parent.resolve()

        raw_data = pd.read_csv(here / f"data/{model}/{fname}")
        input_molecules = pd.read_csv(here / f"data/{model}/mol_labels.csv")

        components.set_components(input_molecules["name"].to_list())

        data = add_name_columns(raw_data, input_molecules[["name", "mol"]])
        data = rename_flowrate_columns(data)

        feed_stream = MaterialStream(stream_type="feed")(
            vapor_fraction=data["feed_vapor_fraction"].to_list(),
            pressure=data["feed_pressure"].to_list(),
            molecules=data[["name_A", "name_B", "name_C"]].to_dict("list"),
            flowrates=data[["flowrate_A", "flowrate_B", "flowrate_C"]].to_dict("list"),
        )

        flash = Flash(
            pressure=data["feed_pressure"].to_list(),
            duty=data["flash_duty"].to_list(),
        )

        _, liquid_stream = flash(feed_stream)

        ordered_data = sort_targets_by_weight(data, feed_stream._mw_idx)
        ok_idx = np.array(ordered_data[ordered_data["Status"] == "OK"].index)

        mae_flow = mean_absolute_error(
            ordered_data.loc[ok_idx, ["flowrate_liquid_A", "flowrate_liquid_B", "flowrate_liquid_C"]], liquid_stream.flow.loc[ok_idx]
        )
        mae_temp = mean_absolute_error(ordered_data.loc[ok_idx, "temp_feed"], np.array(liquid_stream.temperature)[ok_idx])

        assert mae_flow <= 0.009
        assert mae_temp <= 14
