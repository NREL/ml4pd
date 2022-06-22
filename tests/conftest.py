import pandas as pd
from functools import partial
import numpy as np
import re
import pytest


def get_suffixes(df: pd.DataFrame) -> list:

    identifier_columns = [col for col in df.columns if "mol_" in col]
    if len(identifier_columns) == 0:
        identifier_columns = [col for col in df.columns if "name_" in col]
        if len(identifier_columns) == 0:
            raise ValueError("Can't find identifier columns.")

    suffixes = [col.split("_")[-1] for col in identifier_columns]

    return suffixes


@pytest.fixture
def add_name_columns():
    def _add_name_columns(dataframe: pd.DataFrame, input_molecules: pd.DataFrame) -> pd.DataFrame:
        df = dataframe.copy(deep=True)
        suffixes = get_suffixes(df)

        def add_suffix(col, suf):
            return col + suf

        for suffix in suffixes:
            mol_data = input_molecules.rename(mapper=partial(add_suffix, suf=f"_{suffix}"), axis="columns")
            df = pd.merge(df, mol_data, how="left")

        return df

    return _add_name_columns


@pytest.fixture
def sort_targets_by_weight():
    def _sort_targets_by_weight(dataframe: pd.DataFrame, mw_idx: np.array):

        df = dataframe.copy(deep=True)

        # Get target component columns
        mol_columns = [col for col in df.columns if "mol_" in col]
        name_columns = [col for col in df.columns if "name_" in col]
        flow_liquid_columns = [col for col in df.columns if re.search("$flowrate_liq", col) != None]
        flow_vapor_columns = [col for col in df.columns if re.search("$flowrate_vap", col) != None]
        flow_dist_columns = [col for col in df.columns if "flowrate_dist" in col]
        flow_bott_columns = [col for col in df.columns if re.search("^flowrate_bott", col) != None]

        # Align flowrates and corresponding molecules
        for col_group in [mol_columns, name_columns, flow_liquid_columns, flow_vapor_columns, flow_dist_columns, flow_bott_columns]:

            try:
                df[col_group] = np.take_along_axis(df[col_group].to_numpy(), mw_idx, axis=1)
            except:
                pass

        return df

    return _sort_targets_by_weight


@pytest.fixture
def rename_flowrate_columns():
    def _rename_flowrate_columns(dataframe: pd.DataFrame) -> pd.DataFrame:

        df = dataframe.copy(deep=True)

        flowrate_columns = [col for col in df.columns if "flowrate_feed" in col]
        flowrate_dict = {col: f"flowrate_{col.split('_')[-1]}" for col in flowrate_columns}
        df = df.rename(columns=flowrate_dict)

        return df

    return _rename_flowrate_columns
