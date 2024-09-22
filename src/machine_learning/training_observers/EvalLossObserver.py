import os
import random

from datasource.MessageDB import MessageDB
from datasource.conversation_parser import ConversationParser
from datasource.datapoint_provider import DatapointProvider, FixedDatapointProvider
from datatypes.Conversation import Conversation
from machine_learning.BOWInterface import BOWInterface
from machine_learning.TextPredictor import PytorchTextPredictor
from machine_learning.training_observers.train_watcher import TrainingObserver
from utils.examples import Examples
from utils.saving import Saving


class EvalLossObserver(TrainingObserver):
    
    def __init__(self, frequency: int, test_set: DatapointProvider, save_file: str) -> None:
        super().__init__()
        self._frequency = frequency
        self._test_set = test_set
        self._save_file = save_file
        self._counter = 0

    def at_new_training_instance(self, model: PytorchTextPredictor) -> None:
        self._counter += 1
        if self._counter % self._frequency == 0:
            loss = model.estimate_loss(self._test_set)
            Saving.write_str_to_file(f"Loss at {self._counter}: {loss}\n", self._save_file)

    @staticmethod
    def from_params(frequency: int, nb_messages: int, window_size: int, interface: BOWInterface, save_file: str) -> 'EvalLossObserver':
        # TODO te lange init, te veel params. Nood aan tokenizer in de interface -> niet goed
        msgs = Conversation(MessageDB.get_instance().get_test_messages(nb_messages, Examples.RANDOM_SEED))
        dps = ConversationParser().parse(msgs, window_size, interface._tokenizer)
        random.shuffle(dps)
        provider = FixedDatapointProvider(dps)
        return EvalLossObserver(frequency, provider, save_file)
    
    @staticmethod
    def get_default_instance(save_dir: str, interface: BOWInterface, frequency: int) -> 'EvalLossObserver':
        return EvalLossObserver.from_params(frequency, 64, 128, interface, os.path.join(save_dir, "loss.txt"))