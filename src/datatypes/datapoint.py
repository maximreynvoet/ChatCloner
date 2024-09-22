from dataclasses import dataclass
from typing import List, Optional

from datatypes.Person import Person
from datatypes.Token import Token


@dataclass
class DataPoint:
    prev_tokens: List[Token] 
    "TODO more explanation if first elem of this list is the oldest or newest token ?"
    current_token: Token
    current_talker: Person
    previous_talker: Person
    time_talked: int

    def is_new_person_talking(self) -> bool:
        return self.time_talked == 0  # TODO klopt dit? Hoe wordt dit gemodelleerd? Maak test
    
    def minus_oldest(self) -> 'DataPoint':
        """Returns a copy of this datapoint, where the oldest token is deleted
        Useful for data augmentation
        """
        tokens = self.prev_tokens[1:] # TODO depends on what prev_tokens mean (when is first?)
        time_talked = self.time_talked
        if time_talked > len(tokens): time_talked = len(tokens)
        return DataPoint(tokens, self.current_token, self.current_talker, self.previous_talker, time_talked)

    def get_all_split(self, min_tokens=1) -> List["DataPoint"]:
        """Returns all the datapoints you can get out of this one by data augmentation (removing all previous tokens)
        TODO dit maar niet recursief want python kills speed met recursion
        """
        result, dp = [], self
        while len(dp.prev_tokens) >= min_tokens:
            result.append(dp)
            dp = dp.minus_oldest()
        return result
        
        

