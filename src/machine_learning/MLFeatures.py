from torch import Tensor


from dataclasses import dataclass


@dataclass
class MLOutputTensor:
    next_token_logits: Tensor
    next_person_logits: Tensor


@dataclass
class MLInputTensor:
    tensor: Tensor