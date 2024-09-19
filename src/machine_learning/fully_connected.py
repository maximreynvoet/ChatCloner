from typing import List
import torch.nn as nn

class FullyConnectedModule(nn.Module):
    "Mimics SKLearns definition of neural networks where you can simply define the number of layers"

    def __init__(self, nb_in: int, hidden_layer_sizes: List[int], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layers = [nb_in] + hidden_layer_sizes
        self._hidden_fc_layers = [nn.Linear(in_features= x, out_features=y) for x, y in zip(layers[:-1], layers[1:])]
        self._out_layer = nn.Linear(hidden_layer_sizes[-1], self._nb_features_out)