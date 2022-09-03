import nbformat as nbf
from pathlib import Path
from nbconvert.preprocessors import ExecutePreprocessor


current_dir = Path(__file__).absolute().parent.resolve()


def create_notebook(composition, dates, chemistries, ftype, unit):

    # Ingest & manipulate data
    nb = nbf.v4.new_notebook()

    data_ingestion = [
        nbf.v4.new_markdown_cell(f"# {composition} Components Data"),
        nbf.v4.new_code_cell(
            f"""\
import numpy as np
import time

from ml4pd import components
from ml4pd_utils.utils import prep_dist, evaluators, io, prep_df, plot_utils
from ml4pd.aspen_units import Distillation
from ml4pd.streams import MaterialStream

input_molecules = prep_dist.get_mol_labels()
components.set_components(input_molecules['name'].to_list())

raw_data = io.get_test_val_data(compositions=[{composition}], dates={dates}, unit="{unit}", chemistries={chemistries}, ftype="{ftype}")
data = prep_df.add_name_columns(raw_data, input_molecules[['name', 'mol']])
data = prep_dist.add_flow_perc(data)
data = prep_df.rename_flowrate_columns(data)
"""
        ),
    ]

    benchmark_info = [
        nbf.v4.new_markdown_cell("## Benchmark Info"),
        nbf.v4.new_code_cell("input_molecules.T"),
        nbf.v4.new_code_cell("raw_data.describe().T"),
    ]

    time_benchmark = [
        nbf.v4.new_markdown_cell("## Time Benchmark"),
        nbf.v4.new_code_cell(
            """
times = []
for i in range(0, 10):
    start_time = time.time()
    feed_stream = MaterialStream(stream_type="feed")(
        vapor_fraction=data['vapor_fraction'].to_list(),
        pressure=data['feed_pressure'].to_list(),
        molecules=prep_df.get_name_columns(data).to_dict('list'),
        flowrates=prep_df.get_flowrate_columns(data).to_dict('list'),
    )
    
    dist_col = Distillation(
        no_stages = data['no_stages'].to_list(),
        feed_stage =  data['feed_stage'].to_list(),
        pressure = data['pressure_atm'].to_list(),
        reflux_ratio = data['ratio_reflux'].to_list(),
        boilup_ratio = data['ratio_boilup'].to_list(),
        verbose=False,
        fillna=False
    )

    bott_stream, dist_stream = dist_col(feed_stream)
    
    times.append(time.time() - start_time)

ordered_data = prep_dist.sort_targets_by_weight(data, feed_stream._mw_idx)

average = np.mean(times).round(2)
std = np.std(times).round(2)

print(f"{len(data)} data pts take {average} +/- {std} seconds to predict.")
        """
        ),
    ]

    classifier_benchmark = [
        nbf.v4.new_markdown_cell("## Classifier Benchmark"),
        nbf.v4.new_code_cell(
            """\
ok_idx = np.array(ordered_data[ordered_data['Status'] == 'OK'].index)
plot_utils.plot_confusion_matrix(ordered_data, dist_col.status)
            """
        ),
    ]

    flowrates_benchmark = [
        nbf.v4.new_markdown_cell("## Flowrates Benchmark"),
        nbf.v4.new_code_cell(
            """\
prep_dist.plot_flow(
    all_true=ordered_data,
    y_pred=bott_stream.flow,
    data_slice={
        'Status': 'OK',
    }
)"""
        ),
    ]

    duty_benchmark = [
        nbf.v4.new_markdown_cell("## Duty Benchmark"),
        nbf.v4.new_code_cell(
            """\
plot_utils.plot(
    all_true=ordered_data,
    duty_condensor=dist_col.condensor_duty, 
    duty_reboiler=dist_col.reboiler_duty, 
    data_slice={"Status": "OK"}
)"""
        ),
    ]

    temperature_benchmark = [
        nbf.v4.new_markdown_cell("## Temperature Benchmark"),
        nbf.v4.new_code_cell(
            """\
plot_utils.plot(
    all_true=ordered_data, 
    temp_bott=bott_stream.temperature, 
    temp_dist=dist_stream.temperature, 
    data_slice={'Status': 'OK'},
)
            """
        ),
    ]

    mean_absolute_error = [
        nbf.v4.new_markdown_cell("## Mean Absolute Error"),
        nbf.v4.new_code_cell(
            """\
prep_dist.evaluate_flow(
    all_true=ordered_data, 
    y_pred=bott_stream.flow,
    metric='mae',
    data_slice={"Status": "OK"}
)"""
        ),
        nbf.v4.new_code_cell(
            """\
evaluators.evaluate(
    all_true=ordered_data, 
    duty_condensor=dist_col.condensor_duty, 
    duty_reboiler=dist_col.reboiler_duty, 
    data_slice={'Status': 'OK'},
    metric="mae"
)"""
        ),
        nbf.v4.new_code_cell(
            """\
evaluators.evaluate(
    all_true=ordered_data, 
    temp_bott=bott_stream.temperature, 
    temp_dist=dist_stream.temperature, 
    data_slice={'Status': 'OK'},
    metric="mae"
)"""
        ),
    ]

    nb_dir = current_dir / "benchmarks/distillation"
    fname = f"{nb_dir}/distillation_{composition}.ipynb"
    nb["cells"] = (
        data_ingestion
        + benchmark_info
        + time_benchmark
        + classifier_benchmark
        + flowrates_benchmark
        + duty_benchmark
        + temperature_benchmark
        + mean_absolute_error
    )

    # ep = ExecutePreprocessor(timeout=600, kernel_name="ml4pd")
    # ep.preprocess(nb, {"metadata": {"path": f"{nb_dir}/"}})

    with open(fname, "w", encoding="utf-8") as f:
        nbf.write(nb, f)


if __name__ == "__main__":
    current_dir = Path(__file__).absolute().parent.resolve()
    for comp in [2, 3, 4, 5]:
        create_notebook(comp, ["220803", "220821"], ["ketone", "vfa"], "aspen", "distillation")
