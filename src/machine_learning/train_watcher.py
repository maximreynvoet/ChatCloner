
from dataclasses import dataclass
from datatypes.Conversation import Conversation
from machine_learning.BOWInterface import BOWInterface
from machine_learning.BoWModel import BoWModel
from machine_learning.TextPredictor import PytorchTextPredictor
from machine_learning.predict_convo_params import PredictConvoParams
from utils.saving import Saving


"""
TODO ik ben niet zoooo blij dat het een pytorchTextPredictor zou moeten zijn als param
TODO ook ene voor model te saven (checkpointen)
TODO ook trainingsobserver combineren (zo kan je in een observer zowel eval als saven (op andere frequency dat is)) 
    Of om ook inference te doen maar op / temperatuur
"""


class TrainingObserver:
    def at_new_training_instance(self, model: PytorchTextPredictor) -> None:
        """Method that gets called when the model sees a new training instance.
        Subclasses could implement this to for example see what the model would predict on a test-example"""
        pass


class ContinueConvoObserver(TrainingObserver):
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

    def at_new_training_instance(self, model: PytorchTextPredictor) -> None:
        self._counter += 1
        if self._counter % self._activation_frequency == 0: self._test_save_convo(model)

    def _test_save_convo(self, model: PytorchTextPredictor):
        "Tests the model on this conversation, and saves result to file"
        # TODO de model zit al in de interface. Niet goed, niet demure. 
        # Zetten we dit als precondition van init ? Een interface die de model niet nodig heeft?
        conv = self._interface.predict_convo_continuation(self._prediction_params, self._test_convo)
        s = f"Model at iteration {self._counter}: \n\n\n {str(conv)} \n\n\n{'='*50}"
        Saving.write_str_to_file(s, self._save_file)


