from dataclasses import dataclass
from typing import List

from datatypes.Token import Token

@dataclass
class Person:
    ...

@dataclass
class DataPoint:
    prev_tokens: List[Token]
    current_token: Token # TODO remember dat current token een user is als er juist een switch gebeurd is (of andere manier om user change te modelleren)
    current_talker: Person
    previous_talker: Person
    time_talked: int

