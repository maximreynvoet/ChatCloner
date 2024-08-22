
from torch import Tensor

from datatypes.datapoint import DataPoint


class MLInterface:
    @staticmethod
    def datapoint_to_bow_input(dp: DataPoint) -> Tensor:
        ...
