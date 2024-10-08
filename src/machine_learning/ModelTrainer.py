from datasource.datapoint_provider import DatapointProvider
from datatypes.datapoint import DataPoint
from datatypes.tensors.ml_tensors import BOWInputTensor, BOWOutputTensor, CBOWInputTensor, CBOWOutputTensor, MLInputTensor, MLOutputTensor
from machine_learning.BoWModel import BoWModel
from machine_learning.CBowModel import CBowModel
from machine_learning.TextPredictor import PytorchTextPredictor
from machine_learning.training_observers.train_watcher import TrainingObserver


import torch.optim as optim
from tqdm import tqdm


from typing import Callable, List


class ModelTrainer:

    def train_bow_model(self, model: BoWModel, data_provider: DatapointProvider, num_epochs: int, training_observer: TrainingObserver) -> List[float]:
        return self.train_model(model, data_provider, num_epochs, training_observer, 
                                lambda dp: BOWInputTensor.from_datapoint(dp),
                                lambda dp: BOWOutputTensor.from_datapoint(dp))
        
    def train_cbow_model(self, model: CBowModel, data_provider: DatapointProvider, num_epochs: int, training_observer: TrainingObserver) -> List[float]:
        return self.train_model(model, data_provider, num_epochs, training_observer, 
                                lambda dp: CBOWInputTensor(dp.prev_tokens),
                                lambda dp: CBOWOutputTensor.from_datapoint(dp))
    
    def train_model(self, 
                    model: PytorchTextPredictor,
                    data_provider: DatapointProvider, 
                    num_epochs: int, 
                    training_observer: TrainingObserver,

                    dp_to_input:  Callable[[DataPoint], MLInputTensor],
                    dp_to_output: Callable[[DataPoint], MLOutputTensor]
                    
                    ) -> List[float]:
        "Trains the model and reports the losses from all datapoints"
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()  # Set the model to training mode
        losses = []

        for epoch in tqdm(range(num_epochs), "Training epoch"):

            for dp in tqdm(data_provider, "Training on datapoint"):
                training_observer.at_new_training_instance(model)
                optimizer.zero_grad()  # Clear the gradients

                # Forward pass: compute predicted outputs by passing inputs to the model
                input_tensor = dp_to_input(dp)
                output_tensor = model.forward(input_tensor)
                truth_tensor = dp_to_output(dp)

                loss = model.loss(output_tensor, truth_tensor)
                loss.backward()

                optimizer.step()

                # Accumulate the loss for reporting
                losses.append(loss.item())
        return losses