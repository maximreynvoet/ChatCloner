from machine_learning.machine_learning import TextPredictor


import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import List


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