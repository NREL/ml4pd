from ml4pd import components
from ml4pd.streams import MaterialStream
from ml4pd.aspen_units import Distillation
from ml4pd.flowsheet import Flowsheet
from contextlib import nullcontext as does_not_raise

import pytest

feed_1 = MaterialStream(object_id="feed_1")
colu_1 = Distillation(object_id="colu_1")
bott_1, dist_1 = colu_1(feed_1)

feed_2 = MaterialStream(object_id="feed_2")
colu_2 = Distillation(object_id="colu_2")
bott_2, dist_2 = colu_2(feed_2)


class TestGraphStructure:
    def test_disconnected_graph_1(self):
        with pytest.raises(ValueError):
            _ = Flowsheet(input_streams=[feed_1, feed_2], output_streams=[bott_1, dist_1])

    def test_disconnected_graph_2(self):
        with pytest.raises(ValueError):
            _ = Flowsheet(input_streams=[feed_1, feed_2], output_streams=[bott_1, dist_1, bott_2, dist_2])

    def test_disconnected_graph_3(self):
        with does_not_raise():
            _ = Flowsheet(input_streams=[feed_1], output_streams=[bott_1, dist_1])
            _ = Flowsheet(input_streams=[feed_2], output_streams=[bott_2, dist_2])

    def test_plotting(self):
        flowsheet = Flowsheet(input_streams=[feed_1], output_streams=[bott_1, dist_1])
        with does_not_raise():
            flowsheet.plot_model()


molecules = {"name_A": ["water", "acetone"], "name_B": ["ethanol", "methanol"]}

flowrates = {"flowrate_A": [0.1, 0.2], "flowrate_B": [0.3, 0.4]}

feed_pressure = [3, 4]
feed_vapor_fraction = [0.1, 0.2]

column_pressure = feed_pressure
column_no_stages = [10, 20]
column_feed_stage = [5, 10]
column_reflux_ratio = [1, 2]
column_boilup_ratio = [3, 4]

input_dict = {
    "feed_3": {"vapor_fraction": feed_vapor_fraction, "pressure": feed_pressure, "molecules": molecules, "flowrates": flowrates},
    "colu_3": {
        "no_stages": column_no_stages,
        "pressure": column_pressure,
        "reflux_ratio": column_reflux_ratio,
        "boilup_ratio": column_boilup_ratio,
        "feed_stage": column_feed_stage,
    },
    "colu_4": {
        "no_stages": column_no_stages,
        "pressure": column_pressure,
        "reflux_ratio": column_reflux_ratio,
        "boilup_ratio": column_boilup_ratio,
        "feed_stage": column_feed_stage,
    },
}

components.set_components(["water", "acetone", "ethanol", "methanol"])

# The Graph data way
feed_3 = MaterialStream(object_id="feed_3")
colu_3 = Distillation(object_id="colu_3")
bott_3, dist_3 = colu_3(feed_3)

colu_4 = Distillation(object_id="colu_4")
bott_4, dist_4 = colu_4(bott_3)
flowsheet = Flowsheet(input_streams=[feed_3], output_streams=[dist_3, bott_4, dist_4])
flowsheet.run(input_dict)

# The non-graph data way
feed_5 = MaterialStream(vapor_fraction=feed_vapor_fraction, pressure=feed_pressure)(molecules=molecules, flowrates=flowrates)
colu_5 = Distillation(**input_dict["colu_3"])
bott_5, dist_5 = colu_5(feed_5)

colu_6 = Distillation(**input_dict["colu_4"])
bott_6, dist_6 = colu_6(bott_5)


def test_feed_stream():
    assert (feed_3.data.to_numpy() != feed_5.data.to_numpy()).sum() == 0


def test_first_column():
    assert (colu_3.data.to_numpy() != colu_5.data.to_numpy()).sum() == 0
    assert colu_3.condensor_duty == colu_5.condensor_duty
    assert colu_3.reboiler_duty == colu_5.reboiler_duty
    assert all(colu_3.status == colu_5.status)


def test_first_output_streams():
    assert (bott_3.data.to_numpy() != bott_5.data.to_numpy()).sum() == 0
    assert (dist_3.data.to_numpy() != dist_5.data.to_numpy()).sum() == 0


def test_second_column():
    assert (colu_4.data.to_numpy() != colu_6.data.to_numpy()).sum() == 0
    assert colu_4.condensor_duty == colu_6.condensor_duty
    assert colu_4.reboiler_duty == colu_6.reboiler_duty
    assert all(colu_4.status == colu_6.status)


def test_second_output_streams():
    assert (bott_4.data.to_numpy() != bott_6.data.to_numpy()).sum() == 0
    assert (dist_4.data.to_numpy() != dist_6.data.to_numpy()).sum() == 0
