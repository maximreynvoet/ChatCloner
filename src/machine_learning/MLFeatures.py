from torch import Tensor


from dataclasses import dataclass

from utils.utils import Utils

"""
Je zou kunnen zeggen dat dit beetje overbodig kan zijn zoveel types maken, maar ik deel deze mening niet. Deze types zijn nuttig om primitive obsession tegen te gaan en is duidelijker
"""


class ProbabilityTensor(Tensor):
    "Tensor that represents a probability distribution"

class OneHotTensor(Tensor):

    @staticmethod
    def from_idx(idx: int, size: int) -> "OneHotTensor":
        t = Utils.get_one_hot_tensor(size, idx)
        return t.as_subclass(OneHotTensor)        

@dataclass
class MLOutputTensor:
    "Base class that represents what a ML model outputs"


@dataclass
class MLInputTensor:
    "Base class that represents the input of a ML model"


class TokenCountTensor(Tensor):
    """Represents a tensor of counts of tokens
    This has fixed size of n (the number of tokens)
    Each element is a positive (or zero) int
    """

    def add_one(self, idx: int) -> "TokenCountTensor":
        self[idx] += 1
        return self

    def add_to(self, other: "TokenCountTensor") -> "TokenCountTensor":
        return (self + other).as_subclass(TokenCountTensor)


class TokenProbabilityTensor(ProbabilityTensor):
    """Represents a tensor of probability of tokens
    This has fixed size of n (the number of tokens)
    Each element is a positive (or zero) float, and sums up to one
    """


class TalkerTensor(OneHotTensor):
    "Represents a one-hot tensor of who is talking"


class TalkerProbabilityTensor(ProbabilityTensor):
    "Represents a probability distribution of who is talking"


"TODO denken aan hoe de user switch zal gemodelleerd worden"


@dataclass
class BOWInputTensor(MLInputTensor):
    "TODO een tensor voor token counts, en een tensor voor andere data"
    token_counts: TokenCountTensor
    talker_tensor: TalkerTensor


class BOWOutputTensor(MLOutputTensor):
    token_prob: TokenProbabilityTensor
    talker_prob: TalkerProbabilityTensor
