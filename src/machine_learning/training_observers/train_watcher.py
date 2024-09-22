
from typing import Collection
from machine_learning.BOWInterface import BOWInterface
from machine_learning.TextPredictor import PytorchTextPredictor
from machine_learning.training_observers.SaveModelObserver import SaveModelObserver
from machine_learning.training_observers.TestModelObserver import TestModelObserver


"""
TODO ik ben niet zoooo blij dat het een pytorchTextPredictor zou moeten zijn als param
TODO nog een observer om loss te zien op een mini set
"""

class TrainingObserver:
    def combined_with(self, other: 'TrainingObserver') -> 'TrainingObserver':
        # Return a new observer whose update function calls the first one and then the other one
        return CombinedTrainingsObserver([self, other])

    def at_new_training_instance(self, model: PytorchTextPredictor) -> None:
        """Method that gets called when the model sees a new training instance.
        Subclasses could implement this to for example see what the model would predict on a test-example"""
        pass

    @staticmethod
    def get_default_train_observers(model_name: str, interface: BOWInterface, frequency: int) -> 'TrainingObserver':
        return CombinedTrainingsObserver([
            TestModelObserver.get_default_instance(model_name, interface, frequency),
            SaveModelObserver.get_default_instance(model_name, frequency)
        ])

class CombinedTrainingsObserver(TrainingObserver):

    def __init__(self, observers: Collection[TrainingObserver]) -> None:
        super().__init__()
        self._observers = observers

    def update(self, model) -> None:
        [m.at_new_training_instance(model) for m in self._observers]




