from machine_learning.TextPredictor import PytorchTextPredictor
from machine_learning.training_observers.train_watcher import TrainingObserver
from utils.saving import Saving


import os


class SaveModelObserver(TrainingObserver):
    def __init__(self, activation_frequency: int, save_dir: str) -> None:
        super().__init__()
        self._counter = 0
        self._activation_freq = activation_frequency
        self._save_dir = save_dir

    def at_new_training_instance(self, model: PytorchTextPredictor) -> None:
        self._counter += 1
        if self._counter % self._activation_freq == 0:
            Saving.save_bow_model(model, os.path.join(self._save_dir, f"Model_iter_{self._counter}.pth"))

    @staticmethod
    def get_default_instance(model_name: str, frequency: int) -> 'SaveModelObserver':
        return SaveModelObserver(frequency, f"../output/{model_name}")