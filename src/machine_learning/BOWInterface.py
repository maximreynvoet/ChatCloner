from datatypes.MessageFragment import MessageFragment
from datatypes.datapoint import DataPoint
from datatypes.tensors.ml_tensors import BOWInputTensor, MLInputTensor, MLOutputTensor
from machine_learning.BoWModel import BoWModel
from machine_learning.MLInterface import MLInterface


class BOWInterface(MLInterface):
    def _generate_next_input(self, prev_input: MLInputTensor, out_fragment: MessageFragment) -> MLInputTensor:
        return BOWInputTensor.from_previous_output(prev_input, out_fragment)

    def _dp_to_model_in(self, dp: DataPoint) -> MLInputTensor:
        return BOWInputTensor.from_datapoint(dp)


