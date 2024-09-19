from dataclasses import dataclass
from datatypes.tensors.use_case_tensors import TalkerProbabilityTensor, TokenCountTensor, TokenProbabilityTensor
from datatypes.tensors.use_case_tensors import TalkerTensor

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

@dataclass
class BOWInputTensor(MLInputTensor):
    token_counts: TokenCountTensor
    talker_tensor: TalkerTensor


@dataclass
class BOWOutputTensor(MLOutputTensor):
    ...
