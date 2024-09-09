from torch import Tensor
import torch

from machine_learning.MLFeatures import TokenProbabilityTensor


class Utils:
    ...

    @staticmethod
    def get_one_hot_tensor(tensor_len: int, idx: int) -> Tensor:
        t = torch.zeros(tensor_len)
        t[idx] = 1
        return t
    
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
    def sample_token(token_distribution: TokenProbabilityTensor, temperature: float) -> int:
        """Samples a token from the distribution with an adjusted temperature and returns the sampled index
        
        Args: 
            temperature: Zit in ]0, inf]. 
                t = 1 -> gwn sampling
                t < 1 -> meer deterministic
                t > 1 -> meer random (extremer / gwn noise als te veel)
        """

        scaled = Utils.adjust_temperature(token_distribution, temperature)
        idx = torch.multinomial(scaled, 1)
        return int(idx.item())
