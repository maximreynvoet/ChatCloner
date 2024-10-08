

from typing import Collection
from machine_learning.TextPredictor import PytorchTextPredictor


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

class CombinedTrainingsObserver(TrainingObserver):

    def __init__(self, observers: Collection[TrainingObserver]) -> None:
        super().__init__()
        self._observers = observers

    def at_new_training_instance(self, model: PytorchTextPredictor) -> None:
        [m.at_new_training_instance(model) for m in self._observers]
