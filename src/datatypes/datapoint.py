from dataclasses import dataclass
from typing import List

@dataclass
class Token:
    ...

@dataclass
class Person:
    ...

@dataclass
class DataPoint:
    prev_tokens: List[Token]
    current_token: Token
    current_talker: Person
    previous_talker: Person
    time_talked: int

