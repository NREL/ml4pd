from ml4pd import registry
from ml4pd.streams import MaterialStream
from ml4pd.aspen_units import Distillation
import pytest


def test_add_element():
    registry.clear_data()
    dist_column = Distillation(object_id="dist_1")
    feed_stream = MaterialStream(object_id="stream_1")

    assert registry.get_all_columns() == {"dist_1": dist_column}
    assert registry.get_all_streams() == {"stream_1": feed_stream}


def test_add_redundant_element():
    with pytest.raises(KeyError):
        MaterialStream(object_id="stream_1")


def test_remove_element():
    _ = MaterialStream(object_id="stream_2")
    registry.remove_element("stream_2")
    with pytest.raises(KeyError):
        registry.remove_element("stream_2")
