from dataclasses import dataclass

import torch
from datatypes.Person import PersonManager
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

@dataclass
class BOWOutputTensor(MLOutputTensor):
    ...
@dataclass
class BOWInputTensor(MLInputTensor):
    token_counts: TokenCountTensor
    talker_tensor: TalkerTensor

    @staticmethod
    def from_datapoint(dp: DataPoint) -> 'BOWInputTensor':
        token_indices = torch.tensor(dp.prev_tokens)
        token_counts =  torch.bincount(token_indices, minlength= Tokenizer.NUMBER_TOKENS).as_subclass(TokenCountTensor)
        
        talker_idx = dp.current_talker.to_int()
        talker_tensor = Utils.get_one_hot_tensor(PersonManager.get_nb_persons(), talker_idx).as_subclass(TalkerTensor)
        
        return BOWInputTensor(token_counts, talker_tensor)
    
    @staticmethod
    def from_previous_output(previous_input: "BOWInputTensor", previous_output: BOWOutputTensor, temperature: float) -> "BOWInputTensor":
        new_token_idx = Utils.sample_logit(previous_output.token_prob, temperature)
        new_tokens_tensor = previous_input.token_counts.add_one(new_token_idx).as_subclass(TokenCountTensor)

        new_talker = Utils.sample_logit(previous_output.talker_prob, temperature)
        new_talker_tensor = TalkerTensor.from_idx(new_talker, PersonManager.get_nb_persons()).as_subclass(TalkerTensor)

        return BOWInputTensor(new_tokens_tensor, new_talker_tensor)

