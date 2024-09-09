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
        token_tensors = [self._token_to_tensor(x) for x in dp.prev_tokens]
        # TODO this is untested lol
        token_sum = torch.sum(torch.stack(token_tensors), dim=0)
        meta_tensor = self._datapoint_to_meta_feature(dp)
        return torch.concat([token_sum, meta_tensor])
    
    def get_new_input(self, previous_input: BOWInputTensor, previous_output: BOWOutputTensor, temperature: float) -> BOWInputTensor:
        """Converts the previous output into a new input        """
        new_token_idx = Utils.sample_logit(previous_output.token_prob, temperature)
        new_talker = Utils.sample_logit(previous_output.talker_prob, temperature)

        new_tokens_tensor = previous_input.token_counts.add_one(new_token_idx)
        new_talker_tensor = TalkerTensor.from_idx(new_talker, PersonManager.get_nb_persons()).as_subclass(TalkerTensor)

        return BOWInputTensor(new_tokens_tensor, new_talker_tensor)