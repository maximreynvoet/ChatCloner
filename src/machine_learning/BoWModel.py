from sympy import reduce_abs_inequalities
from datatypes.tensors.ml_tensors import BOWInputTensor, BOWOutputTensor
from machine_learning.TextPredictor import PytorchTextPredictor, TextPredictor


import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import List

from machine_learning.fully_connected import FullyConnectedModule
from utils.utils import Utils


class BoWModel(PytorchTextPredictor):
    """Simple bag of words model
    We model this by a joint network (or siamese network, whatever term you like most)"""

    def __init__(self,
                 nb_tokens: int,
                 nb_people: int,
                 token_hidden: FullyConnectedModule,
                 people_hidden: FullyConnectedModule,
                 siamese_hidden: FullyConnectedModule,
                 token_out: FullyConnectedModule,
                 people_out: FullyConnectedModule) -> None:
        "TODO de type hinting kan beter zijn (mss FullyConnected -> nn.Module (als dat goed genoeg is))"
        self._nb_tokens = nb_tokens
        self._nb_people = nb_people
        self._token_hidden = token_hidden
        self._people_hidden = people_hidden
        self._siamese_hidden = siamese_hidden
        self._token_out = token_out
        self._people_out = people_out

    def forward(self, x: BOWInputTensor) -> BOWOutputTensor:
        """TODO kan ik gwn fullyConnected callen ?"""
        siamese_tokens = self._token_hidden(x.token_counts)
        siamese_people = self._people_hidden(x.talker_tensor)
        siamese_in = torch.cat(siamese_tokens, siamese_people)
        siamese_out = self._siamese_hidden(siamese_in)
        token_out = self._token_out(siamese_out)
        people_out = self._people_out(siamese_out)
        return BOWOutputTensor(token_out, people_out)

    
    @staticmethod
    def loss(pred_out: BOWOutputTensor, true_out: BOWOutputTensor):
        """TODO mss niet in deze class ?
        TODO ook mss andere weights voor token / person error?
            En ook andere soort loss voor token of person (cross entropy, en een distance)
        """
        ...

    def train_model(self) -> None:
        ...


