
import os
from typing import Collection
from datatypes.Conversation import Conversation
from machine_learning.BOWInterface import BOWInterface
from machine_learning.TextPredictor import PytorchTextPredictor
from machine_learning.predict_convo_params import PredictConvoParams
from utils.examples import Examples
from utils.saving import Saving


"""
TODO ik ben niet zoooo blij dat het een pytorchTextPredictor zou moeten zijn als param
TODO ook ene voor model te saven (checkpointen)
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


class TestModelObserver(TrainingObserver):
    "Class that continues a test conversation every n trainings examples"

    def __init__(self,
                 interface: BOWInterface,
                 activation_frequency: int,
                 save_file: str,
                 test_convo: Conversation,
                 prediction_params: PredictConvoParams) -> None:
        super().__init__()
        self._counter = 0
        self._interface = interface
        self._activation_frequency = activation_frequency
        self._save_file = save_file
        self._test_convo = test_convo
        self._prediction_params = prediction_params

    @staticmethod
    def get_default_instance(model_name: str, interface: BOWInterface, frequency: int) -> 'TestModelObserver':
        return TestModelObserver(interface,
                                     frequency,
                                     f"../output/{model_name}",
                                     Examples.example_convo,
                                     PredictConvoParams.get_default_params())
        ...

    def at_new_training_instance(self, model: PytorchTextPredictor) -> None:
        self._counter += 1
        if self._counter % self._activation_frequency == 0:
            self._test_save_convo(model)

    def _test_save_convo(self, model: PytorchTextPredictor):
        "Tests the model on this conversation, and saves result to file"
        # TODO de model zit al in de interface. Niet goed, niet demure.
        # Zetten we dit als precondition van init ? Een interface die de model niet nodig heeft?
        conv = self._interface.predict_convo_continuation(
            self._prediction_params, self._test_convo)
        s = f"Model at iteration {self._counter}: \n\n\n {str(conv)} \n\n\n{'='*50}"
        Saving.write_str_to_file(s, self._save_file)


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