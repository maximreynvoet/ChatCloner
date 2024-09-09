from datatypes.Person import Person, PersonManager
from datatypes.datapoint import DataPoint
from machine_learning.MLFeatures import BOWInputTensor, BOWOutputTensor, OneHotTensor, TalkerTensor, TokenCountTensor
from machine_learning.MLInterface import MLInterface


import torch

from utils.utils import Utils


class BOWInterface(MLInterface):
    "TODO geen idee als andere interface subclassen een goed idee is (-V 2024-09-08 08:51)"
    ""

    def datapoint_to_bow_input(self, dp: DataPoint) -> BOWInputTensor:
        "TODO of is mss constructor in tensor beter? -V 2024-09-08"
        token_indices = torch.tensor(dp.prev_tokens)
        token_counts = torch.bincount(token_indices, minlength=self._tokenizer.get_nb_tokens()).as_subclass(TokenCountTensor)
        
        talker_idx = dp.current_talker.to_int()
        talker_tensor = Utils.get_one_hot_tensor(PersonManager.get_nb_persons(), talker_idx).as_subclass(TalkerTensor)
        
        return BOWInputTensor(token_counts, talker_tensor)
        
    
    def get_new_input(self, previous_input: BOWInputTensor, previous_output: BOWOutputTensor, temperature: float) -> BOWInputTensor:
        """Converts the previous output into a new input        """
        new_token_idx = Utils.sample_logit(previous_output.token_prob, temperature)
        new_talker = Utils.sample_logit(previous_output.talker_prob, temperature)

        new_tokens_tensor = previous_input.token_counts.add_one(new_token_idx)
        new_talker_tensor = TalkerTensor.from_idx(new_talker, PersonManager.get_nb_persons()).as_subclass(TalkerTensor)

        return BOWInputTensor(new_tokens_tensor, new_talker_tensor)