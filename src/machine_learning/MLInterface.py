
from torch import Tensor
import torch
import torch.nn.functional as F


from datatypes.Token import Token
from datatypes.datapoint import DataPoint
from other.TokenizerDB import TokenizerDB
from other.tokenizer import Tokenizer


class ModelFeature:
    "Is oftewel een token oftewel een nieuwe persoon dat praat?"


class MLInterface:

    def __init__(self) -> None:
        self._tokenizer = Tokenizer.get_instance()
        self._nb_tokens = self._tokenizer.get_nb_tokens()
        self._nb_meta_features = ... # Nb people talking + 1 (for how long)
    
    def datapoint_to_bow_input(self, dp: DataPoint) -> Tensor:
        token_tensors = [self._token_to_tensor(x) for x in dp.prev_tokens]
        # TODO this is untested lol
        token_sum = torch.sum(torch.stack(token_tensors), dim=0) 
        meta_tensor = self._datapoint_to_meta_feature(dp)
        return torch.concat([token_sum, meta_tensor])

    def datapoint_to_output(self, dp: DataPoint) -> Tensor:
        if dp.is_new_person_talking(): 
            return torch.concat([self._get_empty_token_tensor(), self._datapoint_to_meta_feature(dp)])
        else:
            return torch.concat([self._token_to_tensor(dp.current_token), self._get_empty_meta_tensor()])
    
    def _token_to_tensor(self, token: Token) -> Tensor:
        "One hot encoding van tensor op de plaats van de token"
        t = torch.zeros(self._nb_tokens)
        t[token] = 1
        return t
    
    def _get_empty_token_tensor(self) -> Tensor:
        return torch.zeros(self._nb_tokens)
    
    def _get_empty_meta_tensor(self) -> Tensor:
        return torch.zeros(self._nb_meta_features)

    def _datapoint_to_meta_feature(self, dp: DataPoint) -> Tensor:
        """Returns a tensor that contains all features but the previous tokens.
        This includes the time talked, and who the current talker is"""
        talker_int = ... # TODO swarchen
        nb_people = ... # TODO swarchen
        
        one_hot = F.one_hot(torch.tensor([talker_int]), num_classes=nb_people).squeeze(0)
        integer_tensor = torch.tensor([dp.time_talked], dtype=one_hot.dtype)
        return torch.concat((integer_tensor, one_hot))
        

