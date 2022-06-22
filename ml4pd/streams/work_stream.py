from ml4pd.streams.stream import Stream
from pydantic.dataclasses import dataclass

@dataclass(eq=False)
class WorkStream(Stream):
    pass