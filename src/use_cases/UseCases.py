from datatypes.Conversation import Conversation
from machine_learning.BOWInterface import BOWInterface
from machine_learning.BoWModel import BoWModel
from machine_learning.BoWModelInitParam import BoWModelInitParam
from machine_learning.predict_convo_params import PredictConvoParams
from machine_learning.training_observers.TestModelObserver import TestModelObserver
from machine_learning.training_observers.TrainingObserverFactory import TrainingObserverFactory
from utils.examples import Examples
import random
from datasource.MessageDB import MessageDB
from datasource.conversation_parser import ConversationParser
from datasource.datapoint_provider import FixedDatapointProvider
from datatypes.Person import PersonManager
from machine_learning.BOWModelFactory import BOWModelFactory
from machine_learning.training_observers.train_watcher import TrainingObserver
from other.tokenizer import Tokenizer
from utils.saving import Saving


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
    def interactive_train(model_params: BoWModelInitParam, model_name: str, nb_epochs: int) -> BoWModel:
        "Use case that trains the model and performs intermediate saving and output showing"
        nb_tokens = Tokenizer.NUMBER_TOKENS
        nb_people = PersonManager.get_nb_persons()
        tokenizer = Tokenizer._generate_tokenizer(nb_tokens)
        window_size = 128  # BOW model -> we are able to have a large window

        messages = MessageDB.get_instance().get_messages()
        conversation = Conversation(messages)

        datapoints = ConversationParser().parse(conversation, window_size, tokenizer)
        # augmented_datapoints = Utils.flatmap(datapoints, lambda x: x.get_all_split(1), True)
        augmented_datapoints = datapoints
        random.shuffle(augmented_datapoints)
        provider = FixedDatapointProvider(augmented_datapoints)

        model = BOWModelFactory.get_model_from_params(model_params)
        nb_params: int = sum(p.numel() for p in model.parameters())
        print(f"Training model with {nb_params=}")

        interface = BOWInterface(model)
        observer = TrainingObserverFactory.get_default_train_observers(model_name, interface, 25_000)
        
        model.train_model(provider, nb_epochs, observer)

        return model
