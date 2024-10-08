from dataclasses import dataclass
from typing import List

from attr import dataclass
import torch
from datatypes.Message import Message
from datatypes.MessageFragment import MessageFragment
from datatypes.Person import PersonManager
from datatypes.Token import Token
from datatypes.datapoint import DataPoint
from datatypes.tensors.use_case_tensors import TalkerProbabilityTensor, TokenCountTensor, TokenProbabilityTensor
from datatypes.tensors.use_case_tensors import TalkerTensor
from other.tokenizer import Tokenizer
from utils.utils import Utils

"""
File that groups all the tensor types that are relevant for machine learning

Je zou kunnen zeggen dat dit beetje overbodig kan zijn zoveel types maken, maar ik deel deze mening niet. Deze types zijn nuttig om primitive obsession tegen te gaan en is duidelijker
"""


@dataclass
class MLInputTensor:
    """Base class that represents the input of a ML model
    Every ML model is able to chose what this entails"""

@dataclass
class MLOutputTensor:
    """Base class that represents what a ML model outputs
    Every ML model must follow this convention"""
    token_prob: TokenProbabilityTensor
    talker_prob: TalkerProbabilityTensor

    @staticmethod
    def from_datapoint(dp: DataPoint) -> 'MLOutputTensor':
        "TODO maak de types kloppen"
        token_tensor = Utils.get_one_hot_tensor(Tokenizer.NUMBER_TOKENS, dp.current_token)
        talker_tensor= Utils.get_one_hot_tensor(PersonManager.get_nb_persons(), dp.current_talker.to_int())
        return MLOutputTensor(token_tensor, talker_tensor)
    
    def as_message_fragment(self, temperature: float) -> MessageFragment:
        token_idx = Utils.sample_logit(self.token_prob, temperature)
        talker_idx = Utils.sample_logit(self.talker_prob, temperature)

        return MessageFragment(token_idx, talker_idx)

@dataclass
class BOWOutputTensor(MLOutputTensor):
    ...
@dataclass
class BOWInputTensor(MLInputTensor):
    "TODO oei, neen niet goed, geen time_talked gebruikt!"
    token_counts: TokenCountTensor
    talker_tensor: TalkerTensor

    @staticmethod
    def from_datapoint(dp: DataPoint) -> 'BOWInputTensor':
        token_indices = torch.tensor(dp.prev_tokens)
        if token_indices.numel() == 0: token_counts = torch.zeros(Tokenizer.NUMBER_TOKENS).as_subclass(TokenCountTensor)
        else: token_counts =  torch.bincount(token_indices, minlength= Tokenizer.NUMBER_TOKENS).as_subclass(TokenCountTensor)
        
        talker_idx = dp.current_talker.to_int()
        talker_tensor = Utils.get_one_hot_tensor(PersonManager.get_nb_persons(), talker_idx).as_subclass(TalkerTensor)
        
        return BOWInputTensor(token_counts, talker_tensor)
    
    @staticmethod
    def from_previous_output(previous_input: "BOWInputTensor", output_fragment: MessageFragment) -> "BOWInputTensor":

        new_tokens_tensor = previous_input.token_counts.add_one(output_fragment.token_id).as_subclass(TokenCountTensor)
        new_talker_tensor = TalkerTensor.from_idx(output_fragment.talker_id, PersonManager.get_nb_persons()).as_subclass(TalkerTensor)

        return BOWInputTensor(new_tokens_tensor, new_talker_tensor)

@dataclass
class CBOWInputTensor(MLInputTensor):
    tokens: List[Token]

    def plus_previous_output(self, output: MessageFragment) -> 'CBOWInputTensor':
        token = Token(output.token_id)
        return CBOWInputTensor(self.tokens + [token])

@dataclass
class CBOWOutputTensor(TokenProbabilityTensor):
    ...

