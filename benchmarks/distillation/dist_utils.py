from functools import partial
import pandas as pd
import numpy as np
from seaborn import heatmap
from pathlib import Path
from cycler import cycler
from typing import List
import re
from ml4pd.aspen_units import Distillation
from ml4pd.streams import MaterialStream
from math import ceil
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib as mpl

# Plotting style
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


def add_flow_perc(dataframe: pd.DataFrame, impute: bool = True) -> pd.DataFrame:

    df = dataframe.copy(deep=True)
    suffixes = get_suffixes(df)
    for suffix in suffixes:
        df[f"%_flowrate_bott_{suffix}"] = df[f"flowrate_bott_{suffix}"] / df[f"flowrate_feed_{suffix}"]

    if impute:
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)

    return df


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


def get_suffixes(df: pd.DataFrame) -> list:

    identifier_columns = [col for col in df.columns if "mol_" in col]
    if len(identifier_columns) == 0:
        identifier_columns = [col for col in df.columns if "name_" in col]
        if len(identifier_columns) == 0:
            raise ValueError("Can't find identifier columns.")

    suffixes = [col.split("_")[-1] for col in identifier_columns]

    return suffixes


def get_flowrate_columns(df: pd.DataFrame) -> pd.DataFrame:
    suffixes = get_suffixes(df)
    flowrate_columns = [f"flowrate_{suffix}" for suffix in suffixes]
    return df[flowrate_columns]


def get_name_columns(df: pd.DataFrame) -> pd.DataFrame:
    suffixes = get_suffixes(df)
    name_columns = [f"name_{suffix}" for suffix in suffixes]
    return df[name_columns]


def sort_targets_by_weight(dataframe: pd.DataFrame, mw_idx: np.ndarray):
    df = dataframe.copy(deep=True)

    # Get target component columns
    mol_columns = [col for col in df.columns if "mol_" in col]
    name_columns = [col for col in df.columns if "name_" in col]
    flow_dist_columns = [col for col in df.columns if "flowrate_dist" in col]
    flow_bott_columns = [col for col in df.columns if re.search("^flowrate_bott", col) != None]
    flow_perc_columns = [col for col in df.columns if "%_" in col]

    # Align flowrates and corresponding molecules
    df[mol_columns] = np.take_along_axis(df[mol_columns].to_numpy(), mw_idx, axis=1)
    df[name_columns] = np.take_along_axis(df[name_columns].to_numpy(), mw_idx, axis=1)
    df[flow_bott_columns] = np.take_along_axis(df[flow_bott_columns].to_numpy(), mw_idx, axis=1)
    df[flow_dist_columns] = np.take_along_axis(df[flow_dist_columns].to_numpy(), mw_idx, axis=1)
    df[flow_perc_columns] = np.take_along_axis(df[flow_perc_columns].to_numpy(), mw_idx, axis=1)

    return df


#####################################################################################################################

# FOR BENCHMARKS & PLOTTING #

#####################################################################################################################


def get_benchmark_data(comp: int, date: str, chemistry: str = "ketone"):

    data = pd.read_csv(current_dir / "data" / f"{date}_{chemistry}_distillation_{comp}_aspen.csv")

    return data


def get_mol_labels(fname: str = None):
    if fname is None:
        data = pd.read_csv("https://github.nrel.gov/raw/Machine-Learning-for-Process-Design/autoaspen/master/distillation/mol_labels.csv")
    else:
        data = pd.read_csv(fname)

    return data


def evaluate_flow(y_true: pd.DataFrame, y_pred: pd.DataFrame, metric: str = "mae"):
    suffixes = [col.split("_")[-1] for col in y_pred.columns]
    flow_cols = [f"flowrate_bott_{suffix}" for suffix in suffixes]

    if metric == "mae":
        scores = mean_absolute_error(y_true=y_true[flow_cols], y_pred=y_pred, multioutput="raw_values")
        mean = mean_absolute_error(y_true=y_true[flow_cols], y_pred=y_pred)
    elif metric == "rmse":
        scores = mean_squared_error(y_true=y_true[flow_cols], y_pred=y_pred, multioutput="raw_values", squared=False)
        mean = mean_squared_error(y_true=y_true[flow_cols], y_pred=y_pred, squared=False)

    score_dict = dict(zip(flow_cols + ["mean"], scores.tolist() + [mean]))
    df = pd.DataFrame(score_dict, index=[0])

    return df


def evaluate_duty(y_true: pd.DataFrame, dist_col: Distillation, ok_idx: np.ndarray, metric: str = "mae"):
    duty_scores = []

    if metric == "mae":
        duty_scores.append(mean_absolute_error(y_true.loc[ok_idx, "duty_condensor"], np.array(dist_col.condensor_duty)[ok_idx]))
        duty_scores.append(mean_absolute_error(y_true.loc[ok_idx, "duty_reboiler"], np.array(dist_col.reboiler_duty)[ok_idx]))
    elif metric == "rmse":
        duty_scores.append(mean_squared_error(y_true.loc[ok_idx, "duty_condensor"], np.array(dist_col.condensor_duty)[ok_idx]), squared=False)
        duty_scores.append(mean_squared_error(y_true.loc[ok_idx, "duty_reboiler"], np.array(dist_col.reboiler_duty)[ok_idx]), squared=False)

    duty_scores.append(np.array(duty_scores).mean())

    score_dict = dict(zip(["condensor_duty", "reboiler_duty", "mean"], duty_scores))
    df = pd.DataFrame(score_dict, index=[0])

    return df


def evaluate_temp(y_true: pd.DataFrame, bott_stream: MaterialStream, dist_stream: MaterialStream, ok_idx: np.ndarray, metric: str = "mae"):
    temp_scores = []

    if metric == "mae":
        temp_scores.append(mean_absolute_error(y_true.loc[ok_idx, "temp_bott"], np.array(bott_stream.temperature)[ok_idx]))
        temp_scores.append(mean_absolute_error(y_true.loc[ok_idx, "temp_dist"], np.array(dist_stream.temperature)[ok_idx]))
    elif metric == "rmse":
        temp_scores.append(mean_squared_error(y_true.loc[ok_idx, "temp_bott"], np.array(bott_stream.temperature)[ok_idx]), squared=False)
        temp_scores.append(mean_squared_error(y_true.loc[ok_idx, "temp_dist"], np.array(dist_stream.temperature)[ok_idx]), squared=False)

    temp_scores.append(np.array(temp_scores).mean())

    score_dict = dict(zip(["bott_temp", "dist_temp", "mean"], temp_scores))
    df = pd.DataFrame(score_dict, index=[0])

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

        true = y_true[f"flowrate_bott_{suffix}"]
        pred = y_pred[f"flowrate_{suffix}"]

        ax.plot(true, true, color="white", linestyle="--", dashes=(5, 15))
        ax.scatter(true, pred)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Bottom Flowrate {suffix}")

    if len(suffixes) % 2 == 1:
        axs.ravel()[-1].axis("off")


def plot_duty(y_true, dist_col, ok_idx):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    axs[0].plot(y_true.loc[ok_idx, "duty_condensor"], y_true.loc[ok_idx, "duty_condensor"], color="white", linestyle="--", dashes=(5, 15))
    axs[0].scatter(y_true.loc[ok_idx, "duty_condensor"], pd.DataFrame(dist_col.condensor_duty).iloc[ok_idx])
    axs[0].set_xlabel("True")
    axs[0].set_ylabel("Predicted")
    axs[0].set_title("Condensor Duty (kJ/hr)")

    axs[1].plot(y_true.loc[ok_idx, "duty_reboiler"], y_true.loc[ok_idx, "duty_reboiler"], color="white", linestyle="--", dashes=(5, 15))
    axs[1].scatter(y_true.loc[ok_idx, "duty_reboiler"], pd.DataFrame(dist_col.reboiler_duty).iloc[ok_idx])
    axs[1].set_xlabel("True")
    axs[1].set_ylabel("Predicted")
    axs[1].set_title("Reboiler Duty (kJ/hr)")


def plot_temp(y_true, bott_stream, dist_stream, ok_idx):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    axs[0].plot(y_true.loc[ok_idx, "temp_bott"], y_true.loc[ok_idx, "temp_bott"], color="white", linestyle="--", dashes=(5, 15))
    axs[0].scatter(y_true.loc[ok_idx, "temp_bott"], np.array(bott_stream.temperature)[ok_idx])
    axs[0].set_xlabel("True")
    axs[0].set_ylabel("Predicted")
    axs[0].set_title("Bottom Temperature (C)")

    axs[1].plot(y_true.loc[ok_idx, "temp_dist"], y_true.loc[ok_idx, "temp_dist"], color="white", linestyle="--", dashes=(5, 15))
    axs[1].scatter(y_true.loc[ok_idx, "temp_dist"], np.array(dist_stream.temperature)[ok_idx])
    axs[1].set_xlabel("True")
    axs[1].set_ylabel("Predicted")
    axs[1].set_title("Distillate Temperature (C)")
