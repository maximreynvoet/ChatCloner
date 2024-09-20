from tqdm import tqdm
from datasource.datapoint_provider import DatapointProvider
from datatypes.tensors.ml_tensors import BOWInputTensor, BOWOutputTensor, MLOutputTensor
from machine_learning.TextPredictor import PytorchTextPredictor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from machine_learning.fully_connected import FullyConnectedModule


class BoWModel(PytorchTextPredictor):
    """Simple bag of words model
    We model this by a joint network (or siamese network, whatever term you like most)
    
    TODO what is dropout? Dit implementeren?
    """

    def __init__(self,
                 nb_tokens: int,
                 nb_people: int,
                 token_hidden: FullyConnectedModule,
                 people_hidden: FullyConnectedModule,
                 siamese_hidden: FullyConnectedModule,
                 token_out: FullyConnectedModule,
                 people_out: FullyConnectedModule) -> None:
        "TODO de type hinting kan beter zijn (mss FullyConnected -> nn.Module (als dat goed genoeg is))"
        self._nb_tokens = nb_tokens
        self._nb_people = nb_people
        self._token_hidden = token_hidden
        self._people_hidden = people_hidden
        self._siamese_hidden = siamese_hidden
        self._token_out = token_out
        self._people_out = people_out

    def forward(self, x: BOWInputTensor) -> BOWOutputTensor:
        """TODO kan ik gwn fullyConnected callen ?"""
        siamese_tokens = self._token_hidden(x.token_counts)
        siamese_people = self._people_hidden(x.talker_tensor)
        siamese_in = torch.cat(siamese_tokens, siamese_people)
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

    def train_model(self, data_provider: DatapointProvider, num_epochs: int) -> None:
        
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.train()  # Set the model to training mode

        for epoch in tqdm(range(num_epochs), "Training epoch"):
            total_loss = 0.0
            
            for dp in tqdm(data_provider, "Training on datapoint"):
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
