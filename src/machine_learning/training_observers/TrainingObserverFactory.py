
import os

from machine_learning.BOWInterface import BOWInterface
from machine_learning.training_observers.EvalLossObserver import EvalLossObserver
from machine_learning.training_observers.SaveModelObserver import SaveModelObserver
from machine_learning.training_observers.TestModelObserver import TestModelObserver
from machine_learning.training_observers.train_watcher import CombinedTrainingsObserver, TrainingObserver


class TrainingObserverFactory:
    @staticmethod
    def get_default_train_observers(model_name: str, interface: BOWInterface, frequency: int) -> TrainingObserver:
        save_dir = f"../output/{model_name}"
        test_path = os.path.join(save_dir, "inferences.txt")
        return CombinedTrainingsObserver([
            TestModelObserver.get_default_instance(test_path, interface, frequency),
            SaveModelObserver.get_default_instance(save_dir, frequency),
            EvalLossObserver.get_default_instance(save_dir, interface, frequency)
        ])