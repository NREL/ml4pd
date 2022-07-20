"""
Module gives a components object that a user can use to specify
the types of molecules to consider in the system.
"""

import importlib.resources
import json
import warnings
from typing import List

import pandas as pd
import thermo
from pydantic import BaseModel, StrictStr, validate_arguments
from rdkit.Chem import Descriptors, MolFromSmiles

from ml4pd import data

with importlib.resources.path(data, "rdkit_features.txt") as rdkit_features_path:
    with open(rdkit_features_path, "r") as rdkit_features_file:
        rdkit_features = [line.strip() for line in rdkit_features_file.readlines()]

with importlib.resources.path(data, "thermo_dict.json") as thermo_dict_path:
    with open(thermo_dict_path, "r") as thermo_dict_file:
        thermo_dict = json.load(thermo_dict_file)

__all__ = ["components"]


class Components(BaseModel):
    """
    Object to specify molecules in the system and pass that information along to streams and unit ops.

    ## Useful Attributes:

    - `data`: a df where each row is a molecule and each column is a feature of that molecule.
    - `__repr__`: table where each row is a specified molecule, and each column is a type of identifier.
        Useful for checking that `components` got the right molecules.
    ```
    """

    class Config:
        arbitrary_types_allowed = True

    data: pd.DataFrame = None
    fill_na: bool = True

    @validate_arguments
    def set_components(self, molecules: List[StrictStr]):
        """
        Specify molecules to use and get data for. Streams & unit ops will get
        data from here & check their input data against the specified molecules.

        Args:
            molecules (pd.DataFrame, dict, list): if pd.DataFrame/dict, the
            key/column must be 'name.' The molecules must have 'iupac-like'
            names. Don't input formula/cas/smiles or other representations.
            For example, to specify 2-butanone, butan-2-one also works.
        """
        data = self.get_components(molecules)
        if self.fill_na:
            nan_cols = data.columns[data.isna().any()].to_list()
            for col in nan_cols:
                data[col] = data[col].fillna(data[col].median())
        self.data = data

    @validate_arguments
    def add_components(self, new_molecules: List[StrictStr]):
        """
        Add more molecules to the existing list of molecules. Use ony after set_components().
        May be ueful when process produces molecules that are not accounted for initially.
        """
        duplicates = set(new_molecules) & set(self.data["name"].to_list())
        new_molecules = list(set(new_molecules))
        for molecule in duplicates:
            new_molecules.remove(molecule)
        new_data = self.get_components(new_molecules)
        data = pd.concat([self.data, new_data], axis=0, ignore_index=True)
        if self.fill_na:
            nan_cols = data.columns[data.isna().any()].to_list()
            for col in nan_cols:
                data[col] = data[col].fillna(data[col].median())
        self.data = data

    @staticmethod
    def _get_identifiers(data) -> pd.DataFrame:
        """
        Looks up different identifiers for the input molecules.

        Returns:
            identifiers (pd.DataFrame): with columns 'name', 'cas', 'smiles', 'iupac.'
        """
        unknown = []
        unknown_ind = []
        try:
            identifiers = thermo.datasheet.tabulate_constants(data["name"], full=True)[["smiles", "IUPAC name", "CAS"]].reset_index(drop=True)
            identifiers["name"] = data["name"]
            identifiers = identifiers.rename(columns={"IUPAC name": "iupac", "CAS": "cas"})
        except (ValueError, KeyError):
            identifiers = pd.DataFrame(columns=["smiles", "IUPAC name", "CAS"])
            for ind, row in data.iterrows():
                try:
                    new_molecule = thermo.datasheet.tabulate_constants(row["name"], full=True)[["smiles", "IUPAC name", "CAS"]]
                    identifiers = pd.concat([identifiers, new_molecule]).reset_index(drop=True)
                except:
                    unknown.append(row["name"])
                    unknown_ind.append(ind)
            warnings.warn(f"{unknown} aren't recognized.", stacklevel=2)

        if unknown:
            identifiers["name"] = data.copy(deep=True).drop(index=unknown_ind).reset_index(drop=True)["name"]
        else:
            identifiers["name"] = data["name"]
        identifiers = identifiers.rename(columns={"IUPAC name": "iupac", "CAS": "cas"})

        return identifiers

    @staticmethod
    def _get_rdkit_data(row: pd.Series) -> pd.Series:
        """
        For each row, look up rdkit data specified by the list rdkit_features.

        Args:
            row (pd.Series): must contain a 'smiles' column.

        Returns:
            (pd.Series): molecule + rdkit features.
        """
        smiles = row["smiles"]
        molecule = MolFromSmiles(smiles)

        if molecule:
            fns = [(x, y) for x, y in Descriptors.descList if x in rdkit_features]
            res = {}
            for feature, function in fns:
                try:
                    res[feature] = function(molecule)
                except ValueError:
                    res[feature] = None
        else:
            res = dict(zip(rdkit_features, [None] * (len(rdkit_features))))
            res["smiles"] = smiles

        return pd.Series(dict(**row, **res))

    @staticmethod
    def _get_thermo_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Get thermodynamic data from the thermo package for each molecule in df.

        Args:
            df (pd.DataFrame): must contain a "iupac" column.

        Returns:
            df (pd.DataFrame): original df with added thermo columns.
        """
        df_thermo = (
            thermo.datasheet.tabulate_constants(df["name"], full=True)[list(thermo_dict.keys()) + ["IUPAC name"]]
            .reset_index(drop=True)
            .rename(columns={"IUPAC name": "iupac"})
        )

        df = df.merge(df_thermo, left_on="iupac", right_on="iupac").copy(deep=True)

        return df

    @validate_arguments
    def get_components(self, molecules: List[StrictStr]):
        """
        Specify molecules to use and get data for. Streams & unit ops will get
        data from here & check their input data against the specified molecules.

        Args:
            molecules (pd.DataFrame, dict, list): if pd.DataFrame/dict, the
            key/column must be 'name.' The molecules must have 'iupac-like'
            names. Don't input formula/cas/smiles or other representations.
            For example, to specify 2-butanone, butan-2-one also works.
        """

        if len(molecules) != len(set(molecules)):
            warnings.warn("Dropping duplicates found in molecules.", stacklevel=2)
            molecules = list(set(molecules))

        if len(molecules) < 2:
            warnings.warn("There are less than two molecules in components.", stacklevel=2)

        data = pd.DataFrame({"name": molecules})
        identifiers = self._get_identifiers(data)
        data = data.merge(identifiers, left_on="name", right_on="name").dropna(how="all")
        data = data.apply(self._get_rdkit_data, axis=1)
        data = self._get_thermo_data(data).rename(columns=thermo_dict)

        return data

    def __repr__(self):
        """Returns a prettified string of the identifiers dataframe."""

        return self.data[["name", "iupac", "smiles", "cas"]].to_string()


components = Components()
