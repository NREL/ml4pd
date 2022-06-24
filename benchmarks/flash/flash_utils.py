import pandas as pd
from pathlib import Path
import re
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import matplotlib as mpl
from seaborn import heatmap
from math import ceil
from cycler import cycler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import List, Literal

mpl.style.use(
    {
        "font.size": 20,
        "axes.titlesize": 30,
        "axes.titlepad": 20,
        "axes.labelsize": 20,
        "axes.labelpad": 10,
        "ytick.labelsize": 20,
        "xtick.labelsize": 20,
        "lines.markersize": 1,
        "axes.formatter.use_mathtext": True,
        "axes.formatter.limits": [-1, 1],
        "axes.edgecolor": "white",
        "axes.facecolor": "black",
        "axes.labelcolor": "white",
        "axes.prop_cycle": cycler("color", ["#0B5E90", "#0079C2", "#00A4E4"]),
        "boxplot.boxprops.color": "white",
        "boxplot.capprops.color": "white",
        "boxplot.flierprops.color": "white",
        "boxplot.flierprops.markeredgecolor": "white",
        "boxplot.whiskerprops.color": "white",
        "figure.edgecolor": "black",
        "figure.facecolor": "black",
        "grid.color": "white",
        "lines.color": "white",
        "patch.edgecolor": "white",
        "savefig.edgecolor": "black",
        "savefig.facecolor": "black",
        "text.color": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "figure.subplot.hspace": 0.25,
    }
)

current_dir = Path(__file__).absolute().parent.resolve()


def get_mol_labels():

    mol_labels = pd.read_csv(current_dir / "data/mol_labels.csv")

    return mol_labels


def get_benchmark_data(comp: int, date: str, chemistry: str = "ketone"):

    data = pd.read_csv(current_dir / "data" / f"{date}_{chemistry}_flash_{comp}_aspen.csv")

    return data


def get_suffixes(df: pd.DataFrame) -> list:

    identifier_columns = [col for col in df.columns if "mol_" in col]
    if len(identifier_columns) == 0:
        identifier_columns = [col for col in df.columns if "name_" in col]
        if len(identifier_columns) == 0:
            raise ValueError("Can't find identifier columns.")

    suffixes = [col.split("_")[-1] for col in identifier_columns]

    return suffixes


def add_name_columns(dataframe: pd.DataFrame, input_molecules: pd.DataFrame) -> pd.DataFrame:

    df = dataframe.copy(deep=True)
    suffixes = get_suffixes(df)

    def add_suffix(col, suf):
        return col + suf

    for suffix in suffixes:
        mol_data = input_molecules.rename(mapper=partial(add_suffix, suf=f"_{suffix}"), axis="columns")
        df = pd.merge(df, mol_data, how="left")

    return df


def rename_flowrate_columns(dataframe: pd.DataFrame) -> pd.DataFrame:

    df = dataframe.copy(deep=True)

    flowrate_columns = [col for col in df.columns if "flowrate_feed" in col]
    flowrate_dict = {col: f"flowrate_{col.split('_')[-1]}" for col in flowrate_columns}
    df = df.rename(columns=flowrate_dict)

    return df


def get_flowrate_columns(df: pd.DataFrame) -> pd.DataFrame:
    suffixes = get_suffixes(df)
    flowrate_columns = [f"flowrate_{suffix}" for suffix in suffixes]
    return df[flowrate_columns]


def get_name_columns(df: pd.DataFrame) -> pd.DataFrame:
    suffixes = get_suffixes(df)
    name_columns = [f"name_{suffix}" for suffix in suffixes]
    return df[name_columns]


def sort_targets_by_weight(dataframe: pd.DataFrame, mw_idx: np.array):

    df = dataframe.copy(deep=True)

    # Get target component columns
    mol_columns = [col for col in df.columns if "mol_" in col]
    name_columns = [col for col in df.columns if "name_" in col]
    flow_liquid_columns = [col for col in df.columns if re.search("$flowrate_liq", col) != None]
    flow_vapor_columns = [col for col in df.columns if re.search("$flowrate_vap", col) != None]
    flow_perc_columns = [col for col in df.columns if "%_" in col]

    # Align flowrates and corresponding molecules
    for col_group in [mol_columns, name_columns, flow_liquid_columns, flow_vapor_columns, flow_perc_columns]:
        try:
            df[col_group] = np.take_along_axis(df[col_group].to_numpy(), mw_idx, axis=1)
        except:
            pass

    return df


def plot_confusion_matrix(y_true, y_pred):

    ok_ok = 100 * ((y_true["Status"] == "OK").to_numpy() * np.array(y_pred)).sum() / len(y_pred)
    nok_nok = 100 * ((y_true["Status"] != "OK").to_numpy() * (1 - np.array(y_pred))).sum() / len(y_pred)
    ok_nok = 100 * ((y_true["Status"] == "OK").to_numpy() * (1 - np.array(y_pred))).sum() / len(y_pred)
    nok_ok = 100 * ((y_true["Status"] != "OK").to_numpy() * np.array(y_pred)).sum() / len(y_pred)

    cm = np.array([[nok_nok, nok_ok], [ok_nok, ok_ok]])
    plt.figure(figsize=(15, 15))
    heatmap(pd.DataFrame(cm, columns=["Not OK", "OK"], index=["Not OK", "OK"]), annot=True, annot_kws={"fontsize": 30})
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def plot_flow(y_true, y_pred):
    fig, axs = plt.subplots(ceil(y_pred.shape[1] / 2), 2, figsize=(20, ceil(y_pred.shape[1] / 2) * 10))

    suffixes = [col.split("_")[-1] for col in y_pred.columns]

    for ax, suffix in zip(axs.ravel(), suffixes):

        true = y_true[f"flowrate_liquid_{suffix}"]
        pred = y_pred[f"flowrate_{suffix}"]

        ax.plot(true, true, color="white", linestyle="--", dashes=(5, 15))
        ax.scatter(true, pred)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Liquid Flowrate {suffix}")

    if len(suffixes) % 2 == 1:
        axs.ravel()[-1].axis("off")


def plot_temp(y_true, y_pred, ok_idx):

    true = y_true.loc[ok_idx, "temp_feed"].to_list()
    pred = np.array(y_pred)[ok_idx]

    plt.figure(figsize=(15, 15))
    plt.scatter(true, pred)
    plt.plot(true, true, color="white", linestyle="--", dashes=(5, 15))
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Output Temperature")


def evaluate_flow(y_true: pd.DataFrame, y_pred: pd.DataFrame, metric: str = "mae"):
    suffixes = [col.split("_")[-1] for col in y_pred.columns]
    flow_cols = [f"flowrate_liquid_{suffix}" for suffix in suffixes]

    if metric == "mae":
        scores = mean_absolute_error(y_true=y_true[flow_cols], y_pred=y_pred, multioutput="raw_values")
        mean = mean_absolute_error(y_true=y_true[flow_cols], y_pred=y_pred)
    elif metric == "rmse":
        scores = mean_squared_error(y_true=y_true[flow_cols], y_pred=y_pred, multioutput="raw_values", squared=False)
        mean = mean_squared_error(y_true=y_true[flow_cols], y_pred=y_pred, squared=False)

    score_dict = dict(zip(flow_cols + ["mean"], scores.tolist() + [mean]))
    df = pd.DataFrame(score_dict, index=[0])

    return df


def evaluate_temp(y_true: List[float], y_pred: List[float], ok_idx: List[int], metric: Literal["mae", "rmse"] = "mae"):

    true = y_true.loc[ok_idx, "temp_feed"].to_list()
    pred = np.array(y_pred)[ok_idx]

    if metric == "mae":
        score = mean_absolute_error(y_true=true, y_pred=pred)
    elif metric == "rmse":
        score = mean_squared_error(y_true=true, y_pred=pred, squared=False)

    return score
