from dataclasses import dataclass
from typing import List, Optional

from datatypes.Person import Person
from datatypes.Token import Token


@dataclass
class DataPoint:
    prev_tokens: List[Token]
    # TODO remember dat current token een user is als er juist een switch gebeurd is (of andere manier om user change te modelleren)
    current_token: Token
    # TODO change Optional[Person] -> Person (maak gwn Unknown person die de None verandert); is veel beter voor ML / typing (-V)
    current_talker: Optional[Person]
    # TODO change Optional[Person] -> Person (maak gwn Unknown person die de None verandert); is veel beter voor ML / typing (-V)
    previous_talker: Optional[Person]
    time_talked: int

    def is_new_person_talking(self) -> bool:
        return self.time_talked == 0  # TODO klopt dit? Maak test


