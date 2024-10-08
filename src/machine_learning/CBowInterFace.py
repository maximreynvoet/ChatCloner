from datatypes.MessageFragment import MessageFragment
from datatypes.datapoint import DataPoint
from datatypes.tensors.ml_tensors import CBOWInputTensor, MLInputTensor
from machine_learning.CBowModel import CBowModel
from machine_learning.MLInterface import MLInterface
from utils.IntRange import IntRange


class CBowInterFace(MLInterface):
    def __init__(self, model: CBowModel, allowed_token_range: IntRange) -> None:
        super().__init__(model)
        self._allowed_token_range = allowed_token_range

    def _generate_next_input(self, prev_input: CBOWInputTensor, out_fragment: MessageFragment) -> MLInputTensor:
        return prev_input.plus_previous_output(out_fragment)

    def _dp_to_model_in(self, dp: DataPoint) -> CBOWInputTensor:
        return CBOWInputTensor(dp.prev_tokens)

