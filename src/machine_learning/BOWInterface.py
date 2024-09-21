from tqdm import tqdm
from datasource.datapoint_provider import DatapointProvider
from datatypes.MessageFragment import MessageFragment
from datatypes.Person import Person, PersonManager
from datatypes.datapoint import DataPoint
from datatypes.tensors.pure_tensors import OneHotTensor
from datatypes.tensors.use_case_tensors import TalkerTensor, TokenCountTensor
from datatypes.tensors.ml_tensors import BOWInputTensor, BOWOutputTensor, MLInputTensor, MLOutputTensor
from machine_learning.BoWModel import BoWModel
from machine_learning.MLInterface import MLInterface

import torch
import torch.optim as optim

from utils.utils import Utils


class BOWInterface(MLInterface):
    """TODO Het is niet omdat nu alle methodes weg zijn dat deze classe voor niets dient. Gebruik het om UI te doen, prompt en van text in -> text uit
    """

    def __init__(self, model: BoWModel) -> None:
        super().__init__()
        self._model = model

    def _generate_next_input(self, prev_input: MLInputTensor, out_fragment: MessageFragment) -> MLInputTensor:
        return BOWInputTensor.from_previous_output(prev_input, out_fragment)
    
    def _dp_to_model_in(self, dp: DataPoint) -> MLInputTensor:
        return BOWInputTensor.from_datapoint(dp)

    def predict_output(self, input: MLInputTensor) -> MLOutputTensor:
        self._model.eval()
        return self._model(input)