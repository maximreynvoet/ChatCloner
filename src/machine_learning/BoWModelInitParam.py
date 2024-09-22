from typing import List
from attr import dataclass

from datatypes.Person import PersonManager
from other.tokenizer import Tokenizer
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

    @staticmethod
    def get_default_param() -> "BoWModelInitParam":
        # TODO ezo nie e

        nb_people = PersonManager.get_nb_persons()
        nb_tokens = Tokenizer.NUMBER_TOKENS
        return BoWModelInitParam.get_params_from_power(nb_people, nb_tokens, 0.5, 0.5)
        
    
    @staticmethod
    def get_params_from_power(nb_people: int, nb_tokens: int, token_in_power: float, siam_out_power: float) -> 'BoWModelInitParam':
        token_siamese_in = 64
        people_siamese_in = nb_people
        people_repeat = 4
        siamese_repeat = 4
        siamese_in = token_siamese_in + people_siamese_in
        siamese_out = siamese_in

        # token_in_power = 0.9
        # siam_out_power = 0.8

        return BoWModelInitParam(
                        nb_tokens=nb_tokens,
                        nb_people=nb_people,

                        token_hidden_seq=   Sequences.power_sequence(nb_tokens, token_siamese_in, token_in_power),
                        people_hidden_seq=  Sequences.repeat_sequence(nb_people, people_repeat),
                        siamese_hidden_seq= Sequences.repeat_sequence(siamese_in, siamese_repeat),
                        people_out_seq=     Sequences.power_sequence(siamese_out, nb_people, siam_out_power),
                        token_out_seq=      Sequences.power_sequence(siamese_out, nb_tokens, siam_out_power),
                        
                        leaky_relu_slope= 0.01
                        )