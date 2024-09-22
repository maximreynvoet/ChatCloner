from typing import Callable, Collection
from tqdm import tqdm
from datasource.datapoint_provider import DatapointProvider
from datatypes.datapoint import DataPoint
from datatypes.tensors.ml_tensors import BOWInputTensor, BOWOutputTensor, MLOutputTensor
from machine_learning.BoWModelInitParam import BoWModelInitParam
from machine_learning.TextPredictor import PytorchTextPredictor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from machine_learning.fully_connected import FullyConnectedModule
from machine_learning.training_observers.train_watcher import TrainingObserver


class BoWModel(PytorchTextPredictor):
    """Simple bag of words model
    We model this by a joint network (or siamese network, whatever term you like most)
    
    TODO what is dropout? Dit implementeren?
    """    

    def __init__(self, init_params: BoWModelInitParam) -> None:
        super(BoWModel, self).__init__()
        self._shape_params = init_params
        
        self._nb_tokens =      init_params.nb_tokens
        self._nb_people =      init_params.nb_people
        self._token_hidden =   FullyConnectedModule(init_params.token_hidden_seq,   init_params.leaky_relu_slope)
        self._people_hidden =  FullyConnectedModule(init_params.people_hidden_seq,  init_params.leaky_relu_slope)
        self._siamese_hidden = FullyConnectedModule(init_params.siamese_hidden_seq, init_params.leaky_relu_slope)
        self._token_out =      FullyConnectedModule(init_params.token_out_seq,      init_params.leaky_relu_slope)
        self._people_out =     FullyConnectedModule(init_params.people_out_seq,     init_params.leaky_relu_slope)

    def get_params(self) -> BoWModelInitParam:
        return self._shape_params
        
    def forward(self, x: BOWInputTensor) -> BOWOutputTensor:
        siamese_tokens = self._token_hidden(x.token_counts).as_subclass(torch.Tensor)
        siamese_people = self._people_hidden(x.talker_tensor).as_subclass(torch.Tensor)
        siamese_in = torch.cat((siamese_tokens, siamese_people), dim=0)
        siamese_out = self._siamese_hidden(siamese_in)
        token_out = self._token_out(siamese_out)
        people_out = self._people_out(siamese_out)
        return BOWOutputTensor(token_out, people_out)

    
    @staticmethod
    def loss(pred_out: MLOutputTensor, true_out: MLOutputTensor) -> torch.Tensor:
        """TODO mss niet in deze class ?
        TODO ook mss andere weights voor token / person error?
            En ook andere soort loss voor token of person (cross entropy, en een distance)
        """
        token_loss = F.cross_entropy(pred_out.token_prob, true_out.token_prob)
        talker_loss= F.cross_entropy(pred_out.talker_prob, true_out.talker_prob)
        return token_loss + talker_loss

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

    def train_model(self, data_provider: DatapointProvider, num_epochs: int, training_observer: TrainingObserver) -> None:
        
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.train()  # Set the model to training mode

        for epoch in tqdm(range(num_epochs), "Training epoch"):
            total_loss = 0.0
            
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
                total_loss += loss.item()

