from ml4pd import components
from ml4pd.streams import MaterialStream
import pytest
import numpy as np
from contextlib import nullcontext as does_not_raise

one_instance_mole = {"name_A": "acetone", "name_B": "water"}
one_instance_flow = {"flowrate_A": 0.3, "flowrate_B": 0.5}
two_instance_mole = {"name_A": ["water", "acetone"], "name_B": ["ethanol", "ethanol"]}
two_instance_flow = {"flowrate_A": [0.3, 0.4], "flowrate_B": [0.5, 0.6]}
right_no_of_stage_variables = {"vapor_fraction": 1, "temperature": 3}


class TestDataChecks:

    one_instance = {"molecules": one_instance_mole, "flowrates": one_instance_flow, **right_no_of_stage_variables}
    two_instance = {"molecules": two_instance_mole, "flowrates": two_instance_flow, **right_no_of_stage_variables}

    wrongs_states = {"molecules": two_instance_mole, "flowrates": two_instance_flow, "vapor_fraction": 0}

    molecules_with_nan = {"name_A": ["water", None], "name_B": ["ethanol", "acetone"]}
    nan_mole = {"molecules": molecules_with_nan, "flowrates": two_instance_flow, **right_no_of_stage_variables}

    flowrates_with_nan = {"flowrate_A": [None, 0.4], "flowrate_B": [0.5, 0.6]}
    nan_flow = {"molecules": two_instance_mole, "flowrates": flowrates_with_nan, **right_no_of_stage_variables}

    negative_flowrates = {"flowrate_A": [-1, 0.4], "flowrate_B": [0.5, 0.6]}
    neg_flow = {"molecules": two_instance_mole, "flowrates": negative_flowrates, **right_no_of_stage_variables}

    molecules_not_specified = {"name_A": ["water", "methanol"], "name_B": ["ethanol", "acetone"]}
    wrong_molecules = {"molecules": molecules_not_specified, "flowrates": two_instance_flow, **right_no_of_stage_variables}

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            (one_instance, does_not_raise()),
            (two_instance, does_not_raise()),
            (wrongs_states, pytest.raises(ValueError)),
            (nan_mole, pytest.raises(ValueError)),
            (nan_flow, pytest.raises(ValueError)),
            (neg_flow, pytest.raises(ValueError)),
            (wrong_molecules, pytest.raises(ValueError)),
        ],
    )
    def test_data_conversion_and_checks(self, test_input, expected):
        components.set_components(["water", "acetone", "ethanol"])
        with expected:
            _ = MaterialStream()(**test_input)


class TestOutput:

    one_instance_pressure = {
        "molecules": one_instance_mole,
        "flowrates": one_instance_flow,
        "pressure": 30,
        "temperature": 30,
        "pressure_units": "bar",
    }

    one_instance_temperaturee = {
        "molecules": one_instance_mole,
        "flowrates": one_instance_flow,
        "pressure": 30,
        "temperature": 30,
        "temperature_units": "degK",
    }

    one_instance_flowrate = {
        "molecules": one_instance_mole,
        "flowrates": one_instance_flow,
        "pressure": 30,
        "temperature": 30,
        "flowrates_units": "mol/hr",
    }

    two_instance_pressure = {
        "molecules": two_instance_mole,
        "flowrates": two_instance_flow,
        "pressure": [30, 30],
        "temperature": [30, 30],
        "pressure_units": "bar",
    }

    two_instance_temperature = {
        "molecules": two_instance_mole,
        "flowrates": two_instance_flow,
        "pressure": [30, 30],
        "temperature": [30, 30],
        "temperature_units": "degK",
    }

    two_instance_flowrate = {
        "molecules": two_instance_mole,
        "flowrates": two_instance_flow,
        "pressure": [30, 30],
        "temperature": [30, 30],
        "flowrates_units": "mol/hr",
    }

    @pytest.mark.parametrize(
        "test_input, variable, expected",
        [
            (one_instance_pressure, "pressure", 29.607698001480387),
            (one_instance_temperaturee, "temperature", -243.150),
            (one_instance_flowrate, "flow", [[0.0005, 0.0003]]),
            (two_instance_pressure, "pressure", [29.607698001480387, 29.607698001480387]),
            (two_instance_temperature, "temperature", [-243.150, -243.150]),
            (two_instance_flowrate, "flow", [[0.0003, 0.0005], [0.0006, 0.0004]]),
        ],
    )
    def test_unit_conversion_and_sorting(self, test_input, variable, expected):
        stream = MaterialStream()(**test_input)

        if variable == "flow":
            answer = getattr(stream, variable).to_numpy()
        else:
            answer = getattr(stream, variable)

        assert all(np.isclose(answer, expected).flatten())
