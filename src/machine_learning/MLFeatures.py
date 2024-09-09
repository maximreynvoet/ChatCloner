from torch import Tensor


from dataclasses import dataclass

@dataclass
class MLOutputTensor:
    ...

@dataclass
class MLInputTensor:
    ...


class BOWInputTensor(MLInputTensor):
    "TODO een tensor voor token counts, en een tensor voor andere data"
    
    ...

class BOWOutputTensor(MLOutputTensor):
    ...