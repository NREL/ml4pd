"""Base class for streams"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd
from pydantic.dataclasses import dataclass

from ml4pd import registry


@dataclass
class Stream(ABC):
    """Base class for streams."""

    unit_no: ClassVar[int] = -1
    base_type: ClassVar[str] = "stream"
    object_id: str = None
    before: str = None
    after: str = None
    check_data: bool = True
    verbose: bool = False

    def __post_init__(self):

        self.data: pd.DataFrame = None
        self.status: pd.DataFrame = None

        registry.add_element(self)

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Gather input into df for ML."""

    def __eq__(self, other: object) -> bool:
        return all((self.data.to_numpy() == other.data.to_numpy()).flatten())
