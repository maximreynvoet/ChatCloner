from typing import List
from attr import dataclass

from utils.sequences import Sequences



@dataclass
class BoWModelInitParam:
    "Parameters to initialize the BoWModel"

    nb_tokens: int
    nb_people: int

    token_hidden_seq: List[int]
    people_hidden_seq: List[int]
    siamese_hidden_seq: List[int]
    people_out_seq: List[int]
    token_out_seq: List[int]

    leaky_relu_slope: float

    def describe(self) -> str:
        return f"""Estimated nb params: {self.approx_size()}
tokens: {self.nb_tokens}
    encoding: {self.token_hidden_seq}
people: {self.nb_people}
    encoding: {self.people_hidden_seq}
siamese: 
    {self.siamese_hidden_seq}
token out: 
    {self.token_out_seq}
people out:
    {self.people_out_seq}
"""
    def get_all_sequences(self) -> List[List[int]]:
        return [
            self.token_hidden_seq,
            self.people_hidden_seq,
            self.siamese_hidden_seq,
            self.people_out_seq,
            self.token_out_seq,
        ]

    def approx_size(self) -> int:
        return sum(Sequences.sequence_product(x) for x in self.get_all_sequences())