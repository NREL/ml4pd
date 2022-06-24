"""Contains flowsheet object for connecting unit ops."""

import warnings
from typing import Dict, List, Union

import networkx as nx
from pydantic import validate_arguments

try:
    import graphviz as gv
    from graphviz import Source
except ImportError:
    warnings.warn("Couldn't import graphviz. plot_model() not usable.", stacklevel=2)

from ml4pd import registry
from ml4pd.aspen_units.unit import UnitOp
from ml4pd.streams.stream import Stream


class Flowsheet:
    """Conceptual representation of flowsheets in Aspen.

    ```python
    Flowsheet(
        input_streams: List[Stream],
        output_streams: List[Stream],
        object_id: str = None
    )
    ```

    !!! note
        `Flowsheet` can only currently handle tree-like process
        models (one input stream with no loops).

    ## Data Checks
    - If passed graph isn't fully connected, an error will be raised.

    ## Methods

    - `Flowsheet.plot_model()`: returns a graphviz digraph object.
    - `Flowsheet.run(inputs: Dict[str, dict])`: Perform simulation. See [Example](#example).
    - `Flowsheet.clear_data()`: clear both input & output from streams & unit ops.
    - `Flowsheet.get_element(object_id: str)`: returns stream or unit op given its object_id.
    - `Flowsheet._get_networkx(): returns networkx object that can be manually manipulated for plotting/debugging.
    """

    unit_no = -1

    @validate_arguments
    def __init__(
        self,
        input_streams: List[Stream],
        output_streams: List[Stream],
        object_id: str = None,
    ):
        self.input_streams = input_streams
        self.output_streams = output_streams
        self.passed = False

        Flowsheet.unit_no += 1
        if object_id is None:
            self.object_id = f"flowsheet_{Flowsheet.unit_no}"
        else:
            self.object_id = id

        self.node_attributes = {
            "distillation": {"shape": "rectangle", "fillcolor": "#ADD8E6", "style": "filled"},
            "flash": {"shape": "triangle", "fillcolor": "#ADD8E6", "style": "filled"},
        }

        # Get networkx graph and add output nodes.
        graphs = [registry.get_element(stream.before).networkx for stream in self.output_streams]
        graph = self._check_connections(graphs)
        self.networkx = self._add_output_nodes(graph)

    def plot_model(self) -> gv.Digraph:
        """Return graphviz object. If in .ipynb, the object is automatically plotted."""

        pydot_graph = nx.nx_pydot.to_pydot(self.networkx.copy())
        pydot_graph.set_rankdir("LR")

        for data_pair in list(self.networkx.nodes(data=True)):
            node = data_pair[0]
            data = data_pair[1]

            if "type" in data:
                node_type = data["type"]
                attributes = self.node_attributes[node_type]
                pydot_graph.get_node(node)[0].set_shape(attributes["shape"])
                pydot_graph.get_node(node)[0].set_fillcolor(attributes["fillcolor"])
                pydot_graph.get_node(node)[0].set_style(attributes["style"])

        return Source(pydot_graph.to_string())

    def _check_connections(self, graphs: List[nx.DiGraph]) -> nx.DiGraph:
        """Check if the graph is fully connected.

        Returns:
            nx.DiGraph: common graph if all checks pass.
        """

        if not all(graph == graphs[0] for graph in graphs):
            raise ValueError("Disconnected graph.")

        graph = graphs[0].copy()

        input_ids = [stream.object_id for stream in self.input_streams]
        edge_labels = [edge[2]["label"] for edge in graph.edges(data=True)]
        if not all(object_id in edge_labels for object_id in input_ids):
            raise ValueError("Disconnected graph.")

        return graph

    def _add_output_nodes(self, graph: Union[nx.DiGraph, gv.Digraph]):
        """_summary_

        Args:
            graph (Union[nx.DiGraph, gv.Digraph]): _description_

        Returns:
            _type_: _description_
        """

        new_graph = graph.copy()

        if isinstance(new_graph, gv.Digraph):
            for stream in self.output_streams:
                new_graph.edge(stream.before, f"{stream.object_id}_output", label=stream.object_id)
        elif isinstance(new_graph, nx.DiGraph):
            for stream in self.output_streams:
                new_graph.add_edge(stream.before, f"{stream.object_id}_output", label=stream.object_id)

        return new_graph

    @validate_arguments
    def run(self, inputs: Dict[str, dict]):
        """Pass data through the process model and perform predictions.

        Args:
            inputs (Dict[str, dict]): a nested dictionary. primary key is the
                object_id of stream/unit op that will receive the inner dict.

        Raises:
            RuntimeError: if flowsheet has been run and the data needs to be cleared.
            ValueError: if therea re multiple input streams.
        """

        if self.passed:
            raise RuntimeError("Please call .clear_data() before attempting a rerun.")
        if len(self.input_streams) != 1:
            raise ValueError("Can only handle 1 input stream at the moment.")

        stream_id = self.input_streams[0].object_id
        _ = self.input_streams[0](**inputs[stream_id])
        first_node = self._get_next_nodes(f"{stream_id}_input")[0]

        self._travel_graph(inputs, first_node)

    def _get_next_nodes(self, node_id: str) -> List[str]:
        """Find successors of node_id that aren't output nodes.

        Args:
            node_id (str): object_id of node to find successors for.

        Returns:
            List[str]: list of object_id of successors of node_id.
        """
        next_nodes = list(self.networkx.successors(node_id))
        next_nodes = [node for node in next_nodes if "output" not in node]
        return next_nodes

    def _travel_graph(self, inputs: Dict[str, dict], node_id: str):
        """Iterative Depth-First Search of a directed graph.

        Args:
            inputs (Dict[str, dict]): a nested dictionary. primary key is the
                object_id of stream/unit op that will receive the inner dict.
            node_id (str): object_id of current node.
        """
        current_node = registry.get_element(node_id)
        if current_node.data is None:
            stream = registry.get_element(current_node.before[0])
            _ = current_node(stream, **inputs[current_node.object_id])

            next_nodes = self._get_next_nodes(current_node.object_id)
            for node in next_nodes:
                self._travel_graph(inputs, node)

    def _get_initial_streams(self, input_streams: List[Stream]) -> List[str]:
        """Get streams that have to be fed first, as opposed to those that come in mid-prrocess."""

        adj = self.networkx.adj
        initial_streams = []
        for stream in input_streams:
            stream_id = stream.object_id
            aspen_unit = list(dict(adj[stream_id]).keys())[0]
            if len(list(self.networkx.predecessors[aspen_unit])) == 1:
                initial_streams.append(stream_id)

        if len(initial_streams) == 0:
            raise ValueError("No initial streams found.")

        return initial_streams

    def _get_networkx(self) -> nx.DiGraph:
        """Return networkx object primarily for debugging."""
        return self.networkx

    def clear_data(self):
        """Loop through the graph to clear streams & unit ops data."""
        graph = self.networkx.copy()
        edges = list(graph.edges(data=True))
        nodes = dict(graph.nodes(data=True))

        for edge in edges:
            stream = registry.get_element(edge[2]["label"])
            stream.clear_data()
        for node in nodes:
            try:
                column = registry.get_element(node)
                column.clear_data()
            except KeyError:
                pass

    @validate_arguments
    def get_element(self, object_id: str) -> Union[Stream, UnitOp]:
        """Return object of object_id, if it's in the graph.

        Args:
            object_id (str): object_id of object to search for.

        Returns:
            Union[Stream, UnitOp]: stream of unit op with object_id.
        """

        edge_labels = [edge[2]["label"] for edge in self.networkx.edges(data=True)]

        if self.networkx.has_node(object_id):
            element = registry.get_element(object_id)
        elif object_id in edge_labels:
            element = registry.get_element(object_id)
        else:
            raise ValueError("object_id not found in graph.")

        return element
