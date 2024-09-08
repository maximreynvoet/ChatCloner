
from torch import Tensor
import torch
import torch.nn.functional as F


from datatypes.Person import Person
from datatypes.Token import Token
from datatypes.datapoint import DataPoint
from other.TokenizerDB import TokenizerDB
from other.tokenizer import Tokenizer
from utils.utils import Utils


class MLInterface:
    """
    TODO ja ik weet dat dit niet echt in orde is, slechte software-ontwerp en al.
    Ik ben van plan dit op te delen in een iets voor gwn bow models / gewoon subclassen

    -Victor (2024-09-08 08:40)
    
    """

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
    
    def out_tensor_to_str(self, tensor: Tensor) -> str:
        max_idx = int(torch.argmax(tensor).item())
        if max_idx <= self._nb_tokens: return self._tokenizer.token_to_str(max_idx)
        else: ... # TODO enkel de inpus of output idk moet weg byeee
    
    def _token_to_tensor(self, token: Token) -> Tensor:
        "One hot encoding van tensor op de plaats van de token"
        return Utils.get_one_hot_tensor(self._nb_tokens, token)
    
    def _get_empty_token_tensor(self) -> Tensor:
        return torch.zeros(self._nb_tokens)
    
    def _get_empty_meta_tensor(self) -> Tensor:
        return torch.zeros(self._nb_meta_features)

    def _datapoint_to_meta_feature_out(self, dp: DataPoint) -> Tensor:
        # Fack, aja ik kan gewoon ook double prediction doen
        # Model geeft zowel token als next talker
        ...


    def _datapoint_to_meta_feature_in(self, dp: DataPoint) -> Tensor:
        # TODO split met meta in en meta uit (in heeft time talked, out niet)
        """Returns a tensor that contains all features but the previous tokens.
        This includes the time talked, and who the current talker is"""
        return self._person_to_tensor(dp.current_talker) + self._time_talked_tensor(dp.time_talked)
    
    def _time_talked_tensor(self, time_talked) -> Tensor:
        return torch.tensor([time_talked])
        
    def _person_to_tensor(self, person: Person) -> Tensor:
        talker_int = ... # TODO swarchen
        nb_people = ... # TODO swarchen
        
        one_hot = F.one_hot(torch.tensor([talker_int]), num_classes=nb_people).squeeze(0)
        ...
