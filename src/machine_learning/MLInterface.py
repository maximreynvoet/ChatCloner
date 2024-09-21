

from typing import List

from tqdm import tqdm
from datasource.conversation_parser import ConversationParser
from datatypes.Conversation import Conversation
from datatypes.Message import Message
from datatypes.MessageFragment import MessageFragment
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

    def predict_convo_continuation(self, conversation: Conversation, nb_fragments: int, temperature: float, window_size: int) -> 'Conversation':
        "Continues the conversation by doing a number of iterations on the ml model"
        input_dp = self.convo_to_dp(conversation, window_size)
        fragments = self._predict_next_fragments(nb_fragments, input_dp, temperature)
        [conversation.add_message_fragment(f) for f in fragments]
        return conversation

    def _predict_next_fragments(self, nb_passes: int, input_dp: DataPoint, temperature: float) -> List[MessageFragment]:
        input = self._dp_to_model_in(input_dp)
        res = []
        for _ in tqdm(range(nb_passes), "Predicting next fragments of convo"):
            output = self.predict_output(input)
            frag = output.as_message_fragment(temperature)
            res.append(frag)
            input = self._generate_next_input(input, frag)
        return res

    def _dp_to_model_in(self, dp: DataPoint) -> MLInputTensor:
        ...

    def predict_output(self, input: MLInputTensor) -> MLOutputTensor:
        ...
    
    def _generate_next_input(self, prev_input: MLInputTensor, out_fragment: MessageFragment) -> MLInputTensor:
        ...


    #
    # TRANSLATING
    #

    def convo_to_datapoints(self, window_size: int,  conversation: Conversation) -> List[DataPoint]:
        return ConversationParser().parse(conversation, window_size, self._tokenizer)
    
    def message_do_datapoint(self, window_size: int, message: Message) -> DataPoint:
        return self.convo_to_datapoints(window_size, Conversation([message]))[-1] # TODO is dat de laatste ?
    
    def convo_to_dp(self, convo: Conversation, window_size: int) -> DataPoint:
        return self.convo_to_datapoints(window_size, convo)[-1]
