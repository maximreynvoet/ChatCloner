from typing import List
from torch import Tensor
import torch

from datatypes.tensors.pure_tensors import ProbabilityTensor
from datatypes.tensors.pure_tensors import OneHotTensor
from datatypes.tensors.use_case_tensors import TokenProbabilityTensor


class Utils:
    ...

    @staticmethod
    def get_one_hot_tensor(tensor_len: int, idx: int) -> OneHotTensor:
        t = torch.zeros(tensor_len)
        t[idx] = 1
        return t.as_subclass(OneHotTensor)
    
    @staticmethod
    def adjust_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Adjusts the temperature of a logit distribution by scaling the logits.

        Args:
            logits (torch.Tensor): The input logits (unnormalized log probabilities).
            temperature (float): The temperature value to adjust the distribution.
                                - temperature < 1 makes the distribution more deterministic.
                                - temperature > 1 makes the distribution more random.

        Returns:
            torch.Tensor: The probabilities after adjusting the temperature.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0.")

        # Step 1: Scale the logits by the temperature
        scaled_logits = logits / temperature

        # Step 2: Convert the scaled logits into probabilities using softmax
        probabilities = torch.softmax(scaled_logits, dim=0)

        return probabilities
    
    @staticmethod
    def sample_logit(distribution: ProbabilityTensor, temperature: float) -> int:
        """Samples a token from the distribution with an adjusted temperature and returns the sampled index
        
        Args: 
            temperature: Zit in ]0, inf]. 
                t = 1 -> gwn sampling
                t < 1 -> meer deterministic
                t > 1 -> meer random (extremer / gwn noise als te veel)
        """

        scaled = Utils.adjust_temperature(distribution, temperature)
        idx = torch.multinomial(scaled, 1)
        return int(idx.item())
    
    @staticmethod
    def reduce_sequence_power(start: int, end: int, power: float) -> List[int]:
        "TODO ook een max_length van de sequence als argument (zo verbied je te lang en de NN teveel params)"
        assert 0 < power < 1, f"The power for decreasing the sequence via a power must be between 0,1; not {power}"
        assert start > end > 0
        res = [start]
        while res[-1] > end:
            res.append(int(res[-1] ** power))
        return res
        
def test_reduce_power():
    "TODO waar moeten de testen ?"
    s = Utils.reduce_sequence_power(64, 1, 0.5)
    assert s == [64,32,16,8,4,2,1]
    