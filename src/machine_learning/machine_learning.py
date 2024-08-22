from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from machine_learning.MLInterface import MLInputTensor, MLOutputTensor

"""
Bag of words model (simpelst)

transformer (overfitting / niet genoeg data)

LSTM
RNN
"""


"TODO: Als finetuning gebruiken: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0"

class TextPredictor:
    "Superclass that offers methods to predict token, be able to train etc..."
    
    def __init__(self, nb_features_in: int, nb_features_out: int) -> None:
        self._nb_features_in  = nb_features_in
        self._nb_features_out = nb_features_out

    def train_model(self) -> None:
        raise NotImplementedError()
    
    # TODO hier dinges
    def predict(self, input_tensor: MLInputTensor) -> MLOutputTensor:
        raise NotImplementedError()

class BoWModel(TextPredictor, nn.Module):
    "Simple bag of words model"

    def __init__(self, hidden_layer_sizes: List[int], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layers = [self._nb_features_in] + hidden_layer_sizes
        self._hidden_fc_layers = [nn.Linear(in_features= x, out_features=y) for x, y in zip(layers[:-1], layers[1:])]
        self._out_layer = nn.Linear(hidden_layer_sizes[-1], self._nb_features_out)

    def forward(self, x):
        for layer in self._hidden_fc_layers:
            x = layer(x)
        x = self._out_layer(x)
        return F.softmax(x)
    
    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.forward(input_tensor)
    
    def train_model(self) -> None:
        ...
    
