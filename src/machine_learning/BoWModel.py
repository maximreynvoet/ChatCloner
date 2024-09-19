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
                 people_out: FullyConnectedModule,
                 *args, **kwargs) -> None:
        "TODO de type hinting kan beter zijn (mss FullyConnected -> nn.Module (als dat goed genoeg is))"
        self._nb_tokens = nb_tokens
        self._nb_people = nb_people
        self._token_hidden = token_hidden
        self._people_hidden = people_hidden
        self._siamese_hidden = siamese_hidden
        self._token_out = token_out
        self._people_out = people_out

    @staticmethod
    def get_default_epmty_instance(nb_tokens: int, nb_people: int) -> "BoWModel":
        """Returns an instance of a BoWModel
        
        TODO -> factory method
        In de factory: gemakkelijk hyperparameters toevoegen (om hyperparameter optimization te doen)
            Deze params hier: mogelijks te veel, mogelijks te weinig, geen idee /shrug
            -V 2024-09-19

        TODO mss ook betere API om fullyconnected te maken 
            - FullyConnected.FromSequence(start, nb_repeats)
            - FullyConnected.FromPowerLaw(stars, end, power, (max_len))

        nb_tokens: ongeveer 256 ish
        nb_people: 4-8
        """

        token_siamese_in = 64
        people_siamese_in = nb_people
        people_repeat = 4
        siamese_repeat = 4
        siamese_in = token_siamese_in + people_siamese_in
        siamese_out = siamese_in
        
        return BoWModel(nb_tokens=nb_tokens,
                        nb_people=nb_people,
                        token_hidden=FullyConnectedModule(nb_tokens, Utils.reduce_sequence_power(nb_tokens, token_siamese_in, 0.75)),
                        people_hidden=FullyConnectedModule(nb_people, [nb_people] * people_repeat),
                        siamese_hidden=FullyConnectedModule(siamese_in, [siamese_in] * siamese_repeat),
                        people_out=FullyConnectedModule(siamese_out, Utils.reduce_sequence_power(siamese_out, nb_people, 0.5)),
                        token_out=FullyConnectedModule(siamese_out, Utils.reduce_sequence_power(siamese_out, nb_tokens, 0.5))
                        )

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
