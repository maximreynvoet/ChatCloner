"""File that groups all the tensors that have something to do with the use case (modelling of tokens)"""

from torch import Tensor

from datatypes.tensors.pure_tensors import CountTensor, OneHotTensor, ProbabilityTensor


class TokenCountTensor(CountTensor):
    """Represents a tensor of counts of tokens
    This has fixed size of n (the number of tokens)
    Each element is a positive (or zero) int
    """


class TokenProbabilityTensor(ProbabilityTensor):
    """Represents a tensor of probability of tokens
    This has fixed size of n (the number of tokens)
    Each element is a positive (or zero) float, and sums up to one
    """


class TalkerTensor(OneHotTensor):
    "Represents a one-hot tensor of who is talking"


class TalkerProbabilityTensor(ProbabilityTensor):
    "Represents a probability distribution of who is talking"
