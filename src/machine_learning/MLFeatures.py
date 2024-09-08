from torch import Tensor


from dataclasses import dataclass

@dataclass
class MLOutputTensor:
    next_token_logits: Tensor
    next_person_logits: Tensor

@dataclass
class MLInputTensor:
    tensor: Tensor


class BOWInputTensor(MLInputTensor):
    "TODO een tensor voor token counts, en een tensor voor andere data"
    ...

class BOWOutputTensor(MLOutputTensor):
    ...