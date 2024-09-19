from datatypes.Person import Person, PersonManager
from datatypes.datapoint import DataPoint
from datatypes.tensors.pure_tensors import OneHotTensor
from datatypes.tensors.use_case_tensors import TalkerTensor, TokenCountTensor
from datatypes.tensors.ml_tensors import BOWInputTensor, BOWOutputTensor
from machine_learning.MLInterface import MLInterface


import torch

from utils.utils import Utils


class BOWInterface(MLInterface):

    def datapoint_to_bow_input(self, dp: DataPoint) -> BOWInputTensor:
        token_indices = torch.tensor(dp.prev_tokens)
        token_counts = torch.bincount(token_indices, minlength=self._tokenizer.get_nb_tokens()).as_subclass(TokenCountTensor)
        
        talker_idx = dp.current_talker.to_int()
        talker_tensor = Utils.get_one_hot_tensor(PersonManager.get_nb_persons(), talker_idx).as_subclass(TalkerTensor)
        
        return BOWInputTensor(token_counts, talker_tensor)
        
    def get_next_input(self, previous_input: BOWInputTensor, previous_output: BOWOutputTensor, temperature: float) -> BOWInputTensor:
        """Converts the previous output into a new input
        TODO betere naam"""
        new_token_idx = Utils.sample_logit(previous_output.token_prob, temperature)
        new_tokens_tensor = previous_input.token_counts.add_one(new_token_idx).as_subclass(TokenCountTensor)

        new_talker = Utils.sample_logit(previous_output.talker_prob, temperature)
        new_talker_tensor = TalkerTensor.from_idx(new_talker, PersonManager.get_nb_persons()).as_subclass(TalkerTensor)

        return BOWInputTensor(new_tokens_tensor, new_talker_tensor)
