from pydantic.dataclasses import dataclass
from ml4pd.streams.stream import Stream

@dataclass(eq=False)
class HeatStream(Stream):
    pass
