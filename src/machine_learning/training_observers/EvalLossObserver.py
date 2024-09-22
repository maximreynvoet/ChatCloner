from machine_learning.BoWModel import BoWModel
from machine_learning.TextPredictor import PytorchTextPredictor
from machine_learning.training_observers.train_watcher import TrainingObserver
from utils.saving import Saving


class EvalLossObserver(TrainingObserver):
    
    
    def __init__(self, frequency: int, test_set: ..., save_file: str) -> None:
        super().__init__()
        self._frequency = frequency
        self._test_set = test_set
        self._save_file = save_file
        self._counter = 0

    def at_new_training_instance(self, model: PytorchTextPredictor) -> None:
        self._counter += 1
        if self._counter % self._frequency == 0:
            loss = model.estimate_loss(self._test_set)
            Saving.write_str_to_file(f"Loss at {self._counter}: {loss}", self._save_file)