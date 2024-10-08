from datasource.datapoint_provider import DatapointProvider
from datatypes.tensors.ml_tensors import BOWInputTensor, BOWOutputTensor
from machine_learning.TextPredictor import PytorchTextPredictor
from machine_learning.training_observers.train_watcher import TrainingObserver


import torch.optim as optim
from tqdm import tqdm


from typing import List


class ModelTrainer:
    def train_model(self, model: PytorchTextPredictor, data_provider: DatapointProvider, num_epochs: int, training_observer: TrainingObserver) -> List[float]:
        "Trains the model and reports the losses from all datapoints"
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()  # Set the model to training mode
        losses = []

        for epoch in tqdm(range(num_epochs), "Training epoch"):

            for dp in tqdm(data_provider, "Training on datapoint"):
                training_observer.at_new_training_instance(model)
                optimizer.zero_grad()  # Clear the gradients

                # Forward pass: compute predicted outputs by passing inputs to the model
                input_tensor = BOWInputTensor.from_datapoint(dp)
                output_tensor = model.forward(input_tensor)
                truth_tensor = BOWOutputTensor.from_datapoint(dp)

                loss = model.loss(output_tensor, truth_tensor)
                loss.backward()

                optimizer.step()

                # Accumulate the loss for reporting
                losses.append(loss.item())
        return losses