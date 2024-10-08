from typing import List
from tqdm import tqdm
from datasource.datapoint_provider import DatapointProvider
from datatypes.tensors.ml_tensors import BOWInputTensor, BOWOutputTensor, MLOutputTensor

import torch.optim as optim

import torch.nn as nn

from datasource.datapoint_provider import DatapointProvider
from datatypes.tensors.ml_tensors import MLInputTensor, MLOutputTensor

"""
TODO ideas voor future models
- Victor 2024-09-08 08:45

Bag of words model (simpelst)

transformer (overfitting / niet genoeg data)

LSTM
RNN
"""


"TODO: Als finetuning gebruiken: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0"

class TextPredictor():
    """TODO type hinting met T Typevar van input, output
    De predictor is eig een model dat kan modelleren 
    WIP -V 2024-09-19
    """

    def predict(self, input: MLInputTensor) -> MLOutputTensor:
        ...

class PytorchTextPredictor(TextPredictor, nn.Module):    
    def estimate_loss(self, test_set: DatapointProvider) -> float:
        ...
