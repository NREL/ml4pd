import importlib.resources
import json

from ml4pd import data
from ml4pd import components
from contextlib import nullcontext as does_not_raise
from pydantic import ValidationError
import pytest

with importlib.resources.path(data, "rdkit_features.txt") as rdkit_features_path:
    with open(rdkit_features_path, "r") as rdkit_features_file:
        rdkit_features = [line.strip() for line in rdkit_features_file.readlines()]

with importlib.resources.path(data, "thermo_dict.json") as thermo_dict_path:
    with open(thermo_dict_path, "r") as thermo_dict_file:
        thermo_dict = json.load(thermo_dict_file)


class TestDataChecks:
    acceptable_list_input_1 = ["water", "acetone"]
    acceptable_list_input_2 = ["CC(=O)C", "O"]
    acceptable_list_input_3 = ["67-64-1", "7732-18-5"]

    wrong_list_input = [18, 58]
    wrong_input_type_1 = 18
    wrong_input_type_2 = "water"

    list_input_with_duplicates = ["water", "water", "acetone"]
    list_input_with_1_molecule = ["water"]

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            (acceptable_list_input_1, does_not_raise()),
            (acceptable_list_input_2, does_not_raise()),
            (acceptable_list_input_3, does_not_raise()),
            (wrong_list_input, pytest.raises(ValidationError)),
            (wrong_input_type_1, pytest.raises(ValidationError)),
            (wrong_input_type_2, pytest.raises(ValidationError)),
            (list_input_with_duplicates, pytest.warns(UserWarning)),
            (list_input_with_1_molecule, pytest.warns(UserWarning)),
        ],
    )
    def test_data_conversion_and_checks(self, test_input, expected):
        with expected:
            components.set_components(test_input)


class TestOutput:

    identifiers = ["smiles", "iupac", "cas"]
    list_input = ["water", "acetone"]

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            (list_input, rdkit_features + list(thermo_dict.values()) + ["name"] + identifiers),
        ],
    )
    def test_output_columns(self, test_input, expected):
        components.set_components(test_input)
        assert set(components.data.columns.to_list()) == set(expected)

    list_input_with_duplicates = ["water", "water", "acetone"]

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            (list_input_with_duplicates, list_input_with_duplicates),
        ],
    )
    def test_output_rows(self, test_input, expected):
        components.set_components(test_input)
        print(len(components.data))
        assert len(components.data) == len(set(expected))
