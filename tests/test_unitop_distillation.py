import math
import pathlib
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest
from ml4pd import components
from ml4pd.aspen_units import Distillation
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
    def test_feed_stage_fill_in(self):

        components.set_components(["water", "acetone", "methane", "ethane"])

        feed_stream = MaterialStream(vapor_fraction=vapor_fraction, pressure=feed_pressure)(molecules=molecules, flowrates=flowrates)

        input_dict = {
            "no_stages": no_stages,
            "pressure": pressure,
            "reflux_ratio": reflux_ratio,
            "boilup_ratio": boilup_ratio,
        }

        column = Distillation(**input_dict)
        _ = column(feed_stream)
        assert all(column.feed_stage == (np.array(column.no_stages) / 2).round())

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            ([5, 5], does_not_raise()),
            ([5, 21], does_not_raise()),
            ([5, 1], does_not_raise()),
            ([5, 0], pytest.raises(ValueError)),
            ([5, 22], pytest.raises(ValueError)),
        ],
    )
    def test_feed_stage_values(sel, test_input, expected):

        components.set_components(["water", "acetone", "methane", "ethane"])

        feed_stream = MaterialStream(vapor_fraction=vapor_fraction, pressure=feed_pressure)(molecules=molecules, flowrates=flowrates)

        input_dict = {
            "no_stages": no_stages,
            "pressure": pressure,
            "reflux_ratio": reflux_ratio,
            "boilup_ratio": boilup_ratio,
            "feed_stage": test_input,
        }

        with expected:
            column = Distillation(**input_dict)
            _ = column(feed_stream)

    def test_check_num_cols_1(self):

        components.set_components(["water", "acetone", "methane", "ethane"])

        feed_stream = MaterialStream(vapor_fraction=vapor_fraction, pressure=feed_pressure)(molecules=molecules, flowrates=flowrates)

        input_dict = {
            "no_stages": no_stages,
            "reflux_ratio": reflux_ratio,
            "boilup_ratio": boilup_ratio,
        }

        with pytest.raises(ValueError):
            _ = Distillation(**input_dict)(feed_stream)

    def test_check_num_cols_2(self):

        components.set_components(["water", "acetone", "methane", "ethane"])

        feed_stream = MaterialStream(vapor_fraction=vapor_fraction, pressure=feed_pressure)(molecules=molecules, flowrates=flowrates)

        input_dict = {"no_stages": no_stages, "pressure": pressure, "reflux_ratio": reflux_ratio, "boilup_ratio": boilup_ratio, "bott_rate": [2, 3]}

        with pytest.raises(ValueError):
            _ = Distillation(**input_dict)(feed_stream)

    def test_pressure_conversion(self):
        components.set_components(["water", "acetone", "methane", "ethane"])

        feed_stream = MaterialStream(vapor_fraction=vapor_fraction, pressure=feed_pressure)(molecules=molecules, flowrates=flowrates)

        input_dict = {"no_stages": no_stages, "pressure": pressure, "reflux_ratio": reflux_ratio, "boilup_ratio": boilup_ratio, "pres_units": "bar"}

        column = Distillation(**input_dict)
        _ = column(feed_stream)

        assert column.pressure == [2.960769800148039, 3.9476930668640517]
        assert column.pres_units == "atm"

    @pytest.mark.parametrize(
        "quantity, attribute, expected",
        [
            ([5, 5], "dist_rate", [0.005, 0.005]),
            ([5, 5], "bott_rate", [0.005, 0.005]),
            ([5, 5], "reflux_rate", [0.005, 0.005]),
            ([5, 5], "boilup_rate", [0.005, 0.005]),
        ],
    )
    def test_rate_conversion(self, quantity, attribute, expected):
        components.set_components(["water", "acetone", "methane", "ethane"])

        feed_stream = MaterialStream(vapor_fraction=vapor_fraction, pressure=feed_pressure)(molecules=molecules, flowrates=flowrates)

        input_dict = {"no_stages": no_stages, "pressure": pressure, "reflux_ratio": reflux_ratio, attribute: quantity, "rate_units": "mol/hr"}

        column = Distillation(**input_dict)
        _ = column(feed_stream)

        assert getattr(column, attribute) == expected
        assert column.rate_units == "kmol/hr"


class TestML:
    def test_mae_2(self, add_name_columns, rename_flowrate_columns, sort_targets_by_weight):

        fname = "220302_ketone_distillation_2_aspen.csv"
        model = "distillation"

        here = pathlib.Path(__file__).parent.resolve()

        raw_data = pd.read_csv(here / f"data/{model}/{fname}")
        input_molecules = pd.read_csv(here / f"data/{model}/mol_labels.csv")

        components.set_components(input_molecules["name"].to_list())

        data = add_name_columns(raw_data, input_molecules[["name", "mol"]])
        data = rename_flowrate_columns(data)

        feed_stream = MaterialStream(stream_type="feed")(
            vapor_fraction=data["vapor_fraction"].to_list(),
            pressure=data["feed_pressure"].to_list(),
            molecules=data[["name_A", "name_B"]].to_dict("list"),
            flowrates=data[["flowrate_A", "flowrate_B"]].to_dict("list"),
        )

        dist_col = Distillation(
            no_stages=data["no_stages"].to_list(),
            feed_stage=data["feed_stage"].to_list(),
            pressure=data["pressure_atm"].to_list(),
            reflux_ratio=data["ratio_reflux"].to_list(),
            boilup_ratio=data["ratio_boilup"].to_list(),
            verbose=False,
        )

        bott_stream, _ = dist_col(feed_stream)

        ordered_data = sort_targets_by_weight(data, feed_stream._mw_idx)
        ok_idx = np.array(ordered_data[ordered_data["Status"] == "OK"].index)

        mae = mean_absolute_error(ordered_data.loc[ok_idx, ["flowrate_bott_A", "flowrate_bott_B"]], bott_stream.flow.loc[ok_idx])
        assert mae <= 0.026
