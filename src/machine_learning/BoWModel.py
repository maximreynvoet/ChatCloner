from machine_learning.MLFeatures import BOWInputTensor, BOWOutputTensor
from machine_learning.machine_learning import TextPredictor


import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import List


class BoWModel(TextPredictor, nn.Module):
    "Simple bag of words model"

    """TODO niet zeker over function signatures hierbij, code werd geschreven vooralleer de sigs gemaakt werden"""

    def __init__(self, hidden_layer_sizes: List[int], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layers = [self._nb_features_in] + hidden_layer_sizes
        self._hidden_fc_layers = [nn.Linear(in_features= x, out_features=y) for x, y in zip(layers[:-1], layers[1:])]
        self._out_layer = nn.Linear(hidden_layer_sizes[-1], self._nb_features_out)

    def forward(self, x: BOWInputTensor) -> BOWOutputTensor:
        """TODO dit
        Denken aan mss een joint network 
            (eerst de token probs reduceren van dim 1024 (?) -> 64 ofzo 
                (mss eerst 1024->512->256->128->64), 
            en vanuit de reduced tokens en de talker input)
            
        Noot: 1024 is maar illustratief van wat de mogelijke aantal tokens is, is mss minder (hangt af v tokenizer)
            """
        
        reduced_tokens = x.token_counts # TODO layers om van size n-> ... -> 64 gaan
        # Een aantal inferenties / lagen met reduced tokens en talker
        # Vanuit de reduced token / talker embedding tezamen -> predict next token (uit de 1024), en de volgende talker
        
        for layer in self._hidden_fc_layers:
            x = layer(x)
        x = self._out_layer(x)

        return F.softmax(x)

    def train_model(self) -> None:
        ...