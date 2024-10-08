import os

from attr import dataclass
from datatypes.Conversation import Conversation
from machine_learning.BOWInterface import BOWInterface
from machine_learning.BoWModel import BoWModel
from machine_learning.BoWModelInitParam import BoWModelInitParam
from machine_learning.CBowModel import CBowModel
from machine_learning.MLInterface import MLInterface
from machine_learning.ModelTrainer import ModelTrainer
from machine_learning.TextPredictor import PytorchTextPredictor
from machine_learning.predict_convo_params import PredictConvoParams
from machine_learning.training_observers.TrainingObserverFactory import TrainingObserverFactory
from utils.examples import Examples
import random
from datasource.MessageDB import MessageDB
from datasource.conversation_parser import ConversationParser
from datasource.datapoint_provider import DatapointProvider, FixedDatapointProvider
from datatypes.Person import PersonManager
from machine_learning.BOWModelFactory import BOWModelFactory
from other.tokenizer import Tokenizer
from utils.saving import Saving
from utils.utils import Utils


class UseCases:
    @staticmethod
    def test_model_at_temperatures(model: BoWModel):
        conversation = Examples.example_convo
        interface = BOWInterface(model)

        for temperature in [0.01, 0.1, 0.3, 0.5, 0.8, 1, 1.2, 1.5, 1.8, 2, 3]:
            params = PredictConvoParams(64, 128, temperature)
            predicted = interface.predict_convo_continuation(
                params, conversation)
            print(f"Predicting continuation at temperature {temperature}")
            print(predicted)

        print(5)

    @staticmethod
    def get_training_provider(selection_params: 'DatapointSelectorParameters') -> DatapointProvider:
        nb_tokens = Tokenizer.NUMBER_TOKENS
        nb_people = PersonManager.get_nb_persons()
        tokenizer = Tokenizer._generate_tokenizer(nb_tokens)
        window_size = selection_params.window_size

        messages = MessageDB.get_instance().get_message_sample(selection_params.nb_messages)
        conversation = Conversation(messages, "All messages")

        datapoints = ConversationParser().parse(conversation, window_size, tokenizer)
        augmented_datapoints = datapoints if not selection_params.augment_data else Utils.augment_datapoints(datapoints)
        if selection_params.data_after_augment > 0: 
            augmented_datapoints = random.sample(augmented_datapoints, selection_params.data_after_augment)
        
        provider = FixedDatapointProvider(augmented_datapoints, True)

        return provider

    @staticmethod
    def interactive_bow_train(model_params: BoWModelInitParam, save_dir: str, nb_epochs: int, datapoint_provider: DatapointProvider) -> BoWModel:
        "Use case that trains the model and performs intermediate saving and output showing"
        model = BOWModelFactory.get_model_from_params(model_params)
        interface = BOWInterface(model)
        UseCases.interactive_train(interface, save_dir, nb_epochs, datapoint_provider)
        
        Saving.save_bow_model(model, os.path.join(save_dir, f"TrainedModel_{nb_epochs=}.pth"))
        Saving.write_str_to_file(model_params.describe(), os.path.join(save_dir, "ParamDescription.txt"))

        return model

    @staticmethod
    def interactive_train(interface: MLInterface, save_dir: str, nb_epochs: int, datapoint_provider: DatapointProvider) -> PytorchTextPredictor:
        model = interface._model
        nb_params: int = sum(p.numel() for p in model.parameters())
        print(f"Training model with {nb_params=}")

        interface = BOWInterface(model)
        observer = TrainingObserverFactory.get_default_train_observers(save_dir, interface, 25_000)
        
        losses = model.train_model(datapoint_provider, nb_epochs, observer)
        Saving.write_str_to_file("\n".join(map(str, losses)), os.path.join(save_dir, "Losses.txt"))
        return model


@dataclass
class DatapointSelectorParameters:
    window_size: int
    nb_messages: int # -1 if all messages
    augment_data: bool
    data_after_augment: int # -1 if keep maximum

    @staticmethod
    def get_default_instance() -> 'DatapointSelectorParameters':
        return DatapointSelectorParameters(128, 5_000, True, 50_000)
    
    @staticmethod
    def get_large_context_window_instance() -> 'DatapointSelectorParameters':
        return DatapointSelectorParameters(128, 5_000, True, 50_000)
    
    @staticmethod
    def get_tiny_context_window_instance(window: int) -> 'DatapointSelectorParameters':
        return DatapointSelectorParameters(window, -1, True, -1)