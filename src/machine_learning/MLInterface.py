

from typing import List
from datasource.conversation_parser import ConversationParser
from datatypes.Conversation import Conversation
from datatypes.Message import Message
from datatypes.datapoint import DataPoint
from datatypes.tensors.ml_tensors import MLInputTensor, MLOutputTensor
from other.tokenizer import Tokenizer


class MLInterface:
    """
    TODO ja ik weet dat dit niet echt in orde is, slechte software-ontwerp en al.
    Ik ben van plan dit op te delen in een iets voor gwn bow models / gewoon subclassen

    -Victor (2024-09-08 08:40)

    TODO ook moeten er hier dingen geimplementeerd worden
    """

    def __init__(self) -> None:
        self._tokenizer = Tokenizer.get_instance()
        self._nb_tokens = self._tokenizer.get_nb_tokens()

    "TODO hoe modellen van X praat (kan niet in een str zijn)"

    def get_next_str_from_str_sequence(self, string: str) -> str:
        """Returns the next token that should come after completing the string (by the language model)
        TODO -V Betere comment """
        ...

    def get_n_next_strs_from_str_sequence(self, string: str, n: int) -> str:
        ...

    def generate_next_input(self, prev_input: MLInputTensor, prev_output: MLOutputTensor, temperature: float) -> MLInputTensor:
        ...

    def convo_to_datapoints(self, window_size: int,  conversation: Conversation) -> List[DataPoint]:
        return ConversationParser().parse(conversation, window_size, self._tokenizer)
    
    def message_do_datapoint(self, window_size: int, message: Message) -> DataPoint:
        return self.convo_to_datapoints(window_size, Conversation([message]))[-1] # TODO is dat de laatste ?
