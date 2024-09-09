from dataclasses import dataclass
from typing import List, Optional

from datatypes.Person import Person
from datatypes.Token import Token


@dataclass
class DataPoint:
    prev_tokens: List[Token]
    current_token: Token
    current_talker: Person
    previous_talker: Person
    time_talked: int

    def is_new_person_talking(self) -> bool:
        return self.time_talked == 0  # TODO klopt dit? Hoe wordt dit gemodelleerd? Maak test


