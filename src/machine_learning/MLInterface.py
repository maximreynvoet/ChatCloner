

from typing import List

from tqdm import tqdm
from datasource.conversation_parser import ConversationParser
from datatypes.Conversation import Conversation
from datatypes.Message import Message
from datatypes.MessageFragment import MessageFragment
from datatypes.datapoint import DataPoint
from datatypes.tensors.ml_tensors import MLInputTensor, MLOutputTensor
from machine_learning.TextPredictor import PytorchTextPredictor
from machine_learning.predict_convo_params import PredictConvoParams
from other.tokenizer import Tokenizer

class MLInterface:
    """
    TODO ja ik weet dat dit niet echt in orde is, slechte software-ontwerp en al.
    Ik ben van plan dit op te delen in een iets voor gwn bow models / gewoon subclassen

    -Victor (2024-09-08 08:40)

    TODO ook moeten er hier dingen geimplementeerd worden
    """

    def __init__(self, model: PytorchTextPredictor) -> None:
        self._tokenizer = Tokenizer.get_instance()
        self._nb_tokens = self._tokenizer.get_nb_tokens()
        self._model = model


    def predict_convo_continuation(self, params: PredictConvoParams, conversation: Conversation) -> 'Conversation':
        """Continues the conversation by doing a number of iterations on the ml model
        Returns a new (copy) of a new conversation"""
        conv = conversation.deep_copy()
        input_dp = self.convo_to_dp(conv, params.window_size)
        fragments = self._predict_next_fragments(params.nb_fragments, input_dp, params.temperature)
        
        for f in fragments: 
            conv.add_message_fragment(f)
        return conv

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

    def train_model(self, data_provider: DatapointProvider, num_epochs: int, training_observer: TrainingObserver) -> List[float]:
        "Trains the model and reports the losses from all datapoints"
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.train()  # Set the model to training mode
        losses = []

        for epoch in tqdm(range(num_epochs), "Training epoch"):
            
            for dp in tqdm(data_provider, "Training on datapoint"):
                training_observer.at_new_training_instance(self)
                optimizer.zero_grad()  # Clear the gradients
                
                # Forward pass: compute predicted outputs by passing inputs to the model
                input_tensor = BOWInputTensor.from_datapoint(dp)
                output_tensor = self.forward(input_tensor)
                truth_tensor = BOWOutputTensor.from_datapoint(dp)
                
                loss = BoWModel.loss(output_tensor, truth_tensor)
                loss.backward()
                
                optimizer.step()
                
                # Accumulate the loss for reporting
                losses.append(loss.item())
        return losses
    
    def estimate_loss(self, test_set: DatapointProvider) -> float:
        prev_state_training = self.training
        self.eval()
       
        loss = 0
        for dp in tqdm(test_set, "Evaluating loss"):
            # Forward pass: compute predicted outputs by passing inputs to the model
            input_tensor = BOWInputTensor.from_datapoint(dp)
            output_tensor = self.forward(input_tensor)
            truth_tensor = BOWOutputTensor.from_datapoint(dp)
            
            loss += BoWModel.loss(output_tensor, truth_tensor).item()
            
        self.train(prev_state_training) # Reset state to what it was
        return loss
    
    def predict_output(self, input: CBOWInputTensor) -> CBOWOutputTensor:
        self._model.eval()
        return self._model(input)



    #
    # TRANSLATING
    #

    def convo_to_datapoints(self, window_size: int,  conversation: Conversation) -> List[DataPoint]:
        return ConversationParser().parse(conversation, window_size, self._tokenizer)
    
    def message_do_datapoint(self, window_size: int, message: Message) -> DataPoint:
        return self.convo_to_datapoints(window_size, Conversation([message]))[-1] # TODO is dat de laatste ?
    
    def convo_to_dp(self, convo: Conversation, window_size: int) -> DataPoint:
        return self.convo_to_datapoints(window_size, convo)[-1]
