from torch import Tensor
import torch


class Utils:
    ...

    @staticmethod
    def get_one_hot_tensor(tensor_len: int, idx: int) -> Tensor:
        t = torch.zeros(tensor_len)
        t[idx] = 1
        return t