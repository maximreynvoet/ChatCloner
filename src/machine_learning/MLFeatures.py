from torch import Tensor


from dataclasses import dataclass

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

class TokenProbabilityTensor(Tensor):
    """Represents a tensor of probability of tokens
    This has fixed size of n (the number of tokens)
    Each element is a positive (or zero) float, and sums up to one
    """

class TalkerTensor(Tensor):
    "Represents a one-hot tensor of who is talking"

class TalkerProbabilityTensor(Tensor):
    "Represents a probability distribution of who is talking"


"TODO denken aan hoe de user switch zal gemodelleerd worden"
@dataclass
class BOWInputTensor(MLInputTensor):
    "TODO een tensor voor token counts, en een tensor voor andere data"
    token_counts: TokenCountTensor
    talker_tensor: TalkerTensor


class BOWOutputTensor(MLOutputTensor):
    ...

    def get_most_likely_tokens(self): ...

    def sample_token(self): ...

    "TODO in welke classe"
