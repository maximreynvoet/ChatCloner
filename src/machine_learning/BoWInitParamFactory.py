from typing import List
from datatypes.Person import PersonManager
from machine_learning.BoWModelInitParam import BoWModelInitParam
from other.tokenizer import Tokenizer
from utils.sequences import Sequences

nb_tokens = Tokenizer.NUMBER_TOKENS
nb_people = PersonManager.get_nb_persons()

class BoWInitParamFactory:


    @staticmethod
    def get_default_params() -> "BoWModelInitParam":
        return BoWInitParamFactory.get_params_from_power(0.5, 0.5)
    
    
    @staticmethod
    def get_params_from_lengths(nb_tokens: int, nb_people: int,
        token_in_length: int, token_out_length: int,
        people_in_length: int, people_out_length: int,
        siamese_input: int, siamese_length: int, siamese_output: int
        ):
        
        token_in_seq = Sequences.linear_sequence_size(nb_tokens, siamese_input, token_in_length)
        people_in_seq = Sequences.linear_sequence_size(nb_people, siamese_input, people_in_length)
        siamese_seq = Sequences.linear_sequence_size(siamese_input, siamese_output, siamese_length)
        token_out_seq = Sequences.linear_sequence_size(siamese_output, nb_tokens, token_out_length)
        people_out_seq = Sequences.linear_sequence_size(siamese_output, nb_people, people_out_length)
        
        return BoWModelInitParam(nb_tokens, nb_people, token_in_seq, people_in_seq, siamese_seq, people_out_seq, token_out_seq, 0.01)
    
    @staticmethod
    def get_hyperparam_options(nb_tokens: int, nb_people: int) -> List[BoWModelInitParam]:
        sizes = [1,2,4,8]
        dims = [4,8,16,32,64,128,256]
        params = [BoWInitParamFactory.get_params_from_lengths(nb_tokens, nb_people, s, s, s, s, d, s, d) for s in sizes for d in dims]
        return sorted(params, key= lambda x: x.approx_size())   

    @staticmethod
    def get_params_from_power(token_in_power: float, siam_out_power: float) -> 'BoWModelInitParam':
        # TODO ezo nie e
        nb_people = PersonManager.get_nb_persons()
        nb_tokens = Tokenizer.NUMBER_TOKENS

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