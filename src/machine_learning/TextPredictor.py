from typing import List
from tqdm import tqdm
from datasource.datapoint_provider import DatapointProvider
from datatypes.tensors.ml_tensors import BOWInputTensor, BOWOutputTensor, MLOutputTensor

import torch.optim as optim

    
from machine_learning.training_observers.train_watcher import TrainingObserver

import torch.nn as nn

from datasource.datapoint_provider import DatapointProvider
from datatypes.tensors.ml_tensors import MLInputTensor, MLOutputTensor

"""
TODO ideas voor future models
- Victor 2024-09-08 08:45

Bag of words model (simpelst)

transformer (overfitting / niet genoeg data)

LSTM
RNN
"""


"TODO: Als finetuning gebruiken: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0"

class TextPredictor():
    """TODO type hinting met T Typevar van input, output
    De predictor is eig een model dat kan modelleren 
    WIP -V 2024-09-19
    """

    def predict(self, input: MLInputTensor) -> MLOutputTensor:
        ...

class PytorchTextPredictor(TextPredictor, nn.Module):    
    def estimate_loss(self, test_set: DatapointProvider) -> float:
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
                
                loss = self.loss(output_tensor, truth_tensor)
                loss.backward()
                
                optimizer.step()
                
                # Accumulate the loss for reporting
                losses.append(loss.item())
        return losses