from datatypes.Conversation import Conversation
from machine_learning.BOWInterface import BOWInterface
from machine_learning.TextPredictor import PytorchTextPredictor
from machine_learning.predict_convo_params import PredictConvoParams
from machine_learning.training_observers.train_watcher import TrainingObserver
from utils.examples import Examples
from utils.saving import Saving


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
    def get_default_instance(save_file_path: str, interface: BOWInterface, frequency: int) -> 'TestModelObserver':
        return TestModelObserver(interface,
                                     frequency,
                                     save_file_path,
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