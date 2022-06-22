"""Helper functions for streams."""

from typing import Literal, Dict

import pandas as pd
from ml4pd import registry
from ml4pd.streams import MaterialStream
from pydantic import confloat, validate_arguments


@validate_arguments
def filter_by_mol_perc(
    output_stream: MaterialStream,
    mol_perc: confloat(ge=0, le=1),
    column_name: str = None,
    input_stream: MaterialStream = None,
    rel_perc: confloat(ge=0, le=1) = 0,
    mol_perc_direction: Literal["ge", "le"] = "ge",
    rel_perc_direction: Literal["ge", "le"] = "ge",
) -> pd.Index:
    """A function to pick instances of output stream that pass a specified threshold.

    Args:
        output_stream (MaterialStream): stream to filter.
        mol_perc (float): Only pick instances with output/sum(output) >= mol_perc for any
            column unless column_name is specified.
        column_name (str, optional): Column name to focus on filtering. Defaults to None.
        input_stream (MaterialStream, optional): To further filter results. Defaults to None.
        rel_perc (float, optional): Only instances if output/input >= rel_perc. Defaults to 0.

    Returns:
        pd.Index: Indices of instances that pass filter(s).
    """

    out_flow = output_stream.flow
    flow_perc = out_flow.div(out_flow.sum(axis=1), axis=0)

    if input_stream is not None:
        inp_flow = input_stream.flow
        flow_perc_rel_to_inp = out_flow.div(inp_flow)

    if column_name is not None:

        if mol_perc_direction == "ge":
            index_1 = set(out_flow[flow_perc[column_name] >= mol_perc].index)
        else:
            index_1 = set(out_flow[flow_perc[column_name] <= mol_perc].index)

        if input_stream is not None:

            if rel_perc_direction == "ge":
                index_2 = set(out_flow[flow_perc_rel_to_inp[column_name] >= rel_perc].index)
            else:
                index_2 = set(out_flow[flow_perc_rel_to_inp[column_name] <= rel_perc].index)

            index = pd.Index(index_1 & index_2)
        else:
            index = pd.Index(index_1)

    else:

        flow_max = flow_perc.max(axis=1)

        if mol_perc_direction == "ge":
            index_1 = set(out_flow[flow_max >= mol_perc].index)
        else:
            index_1 = set(out_flow[flow_max <= mol_perc].index)

        if input_stream is not None:
            flow_max_rel_to_inp = flow_perc_rel_to_inp.max(axis=1)

            if rel_perc_direction == "ge":
                index_2 = set(out_flow[flow_max_rel_to_inp >= rel_perc].index)
            else:
                index_2 = set(out_flow[flow_max_rel_to_inp <= rel_perc].index)

            index_3 = set(out_flow[flow_perc.idxmax(axis=1) == flow_perc_rel_to_inp.idxmax(axis=1)].index)
            index = pd.Index(index_1 & index_2 & index_3)
        else:
            index = pd.Index(index_1)

    return index


def update_material_streams(streams: Dict[str, dict]):
    """Add dict data to stream using object_id."""
    outputs = []

    for stream_id, data in streams.items():
        outputs.append(registry.get_element(stream_id)(**data))

    return tuple(outputs)
