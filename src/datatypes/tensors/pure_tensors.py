"""File that groups all the "pure" types of tensors together (probability distribution tensors, count tensors, ...)"""


from torch import Tensor

from utils.utils import Utils


class ProbabilityTensor(Tensor):
    "Tensor that represents a probability distribution"


class CountTensor(Tensor):
    "Tensor that represents a count (ie all elements are ints)"

    def add_one(self, idx: int) -> "CountTensor":
        self[idx] += 1
        return self

    def add_to(self, other: "CountTensor") -> "CountTensor":
        return (self + other).as_subclass(CountTensor)
    

class OneHotTensor(Tensor):

    @staticmethod
    def from_idx(idx: int, size: int) -> "OneHotTensor":
        t = Utils.get_one_hot_tensor(size, idx)
        return t.as_subclass(OneHotTensor)
