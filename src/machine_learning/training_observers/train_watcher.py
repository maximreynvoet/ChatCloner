
import os
from typing import Collection
from machine_learning.BOWInterface import BOWInterface
from machine_learning.TextPredictor import PytorchTextPredictor
from machine_learning.training_observers.EvalLossObserver import EvalLossObserver
from machine_learning.training_observers.SaveModelObserver import SaveModelObserver
from machine_learning.training_observers.TestModelObserver import TestModelObserver


"""
TODO ik ben niet zoooo blij dat het een pytorchTextPredictor zou moeten zijn als param
"""

class TrainingObserver:
    # TODO elke subclass heeft counter behavior, zet dit in superclass
    def combined_with(self, other: 'TrainingObserver') -> 'TrainingObserver':
        # Return a new observer whose update function calls the first one and then the other one
        return CombinedTrainingsObserver([self, other])

    def at_new_training_instance(self, model: PytorchTextPredictor) -> None:
        """Method that gets called when the model sees a new training instance.
        Subclasses could implement this to for example see what the model would predict on a test-example"""
        pass

    @staticmethod
    def get_default_train_observers(model_name: str, interface: BOWInterface, frequency: int) -> 'TrainingObserver':
        save_dir = f"../output/{model_name}"
        test_path = os.path.join(save_dir, "inferences.txt")
        return CombinedTrainingsObserver([
            TestModelObserver.get_default_instance(test_path, interface, frequency),
            SaveModelObserver.get_default_instance(save_dir, frequency),
            EvalLossObserver.get_default_instance(model_name, interface, frequency)
        ])

class CombinedTrainingsObserver(TrainingObserver):

    def __init__(self, observers: Collection[TrainingObserver]) -> None:
        super().__init__()
        self._observers = observers

    def update(self, model) -> None:
        [m.at_new_training_instance(model) for m in self._observers]

