from typing import List
import torch
import torch.nn as nn

class FullyConnectedModule(nn.Module):
    "Mimics SKLearns definition of neural networks where you can simply define the number of layers"

    def __init__(self, layer_sizes: List[int], slope: float = 0.01) -> None:
        super(FullyConnectedModule, self).__init__()
        self._slope = slope

        self._hidden_fc_layers = nn.ModuleList(
            [nn.Linear(in_features= x, out_features=y) for x, y in zip(layer_sizes[:-2], layer_sizes[1:])]
            )
        
        self._out_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])

    def forward(self, x):
        x = x.float()
        # Forward through the hidden layers with activation
        for layer in self._hidden_fc_layers:
            x = nn.functional.leaky_relu(layer(x), self._slope)
        
        x = self._out_layer(x) # Apparently do not do leaky relu in the end (better for classification, regression etc)
        return x
