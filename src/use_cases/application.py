import random
from datasource.MessageDB import MessageDB
from datasource.conversation_parser import ConversationParser
from datasource.datapoint_provider import FixedDatapointProvider
from datatypes.Person import PersonManager
from machine_learning.BOWInterface import BOWInterface
from machine_learning.BOWModelFactory import BOWModelFactory
from machine_learning.BoWModelInitParam import BoWModelInitParam
from machine_learning.predict_convo_params import PredictConvoParams
from machine_learning.training_observers.TestModelObserver import TestModelObserver
from other.tokenizer import Tokenizer
from use_cases.UseCases import UseCases
from utils.examples import Examples
from utils.saving import Saving


def generate_model():
    # TODO al die fn in een aparte, iets om gemakkelijker te initializeren
    # TODO ja ik weet geen test / split, maar dat is hier voor de lol
    nb_tokens = Tokenizer.NUMBER_TOKENS
    nb_people = PersonManager.get_nb_persons()
    tokenizer = Tokenizer._generate_tokenizer(nb_tokens)
    window_size = 128 # BOW model -> we are able to have a large window
    
    messages = MessageDB.get_instance().get_messages()
    
    datapoints = ConversationParser().parse(messages, window_size, tokenizer)
    # augmented_datapoints = Utils.flatmap(datapoints, lambda x: x.get_all_split(1), True)
    augmented_datapoints = datapoints
    random.shuffle(augmented_datapoints)
    provider = FixedDatapointProvider(augmented_datapoints)
    
    model = BOWModelFactory.get_default_epmty_instance(nb_tokens, nb_people)

    test_convo = Examples.example_convo
    interface = BOWInterface(model)
    prediction_params = PredictConvoParams(32, 128, 1)
    observer = TestModelObserver(interface, 10_000, "../output/mini_bowwow.txt", test_convo, prediction_params)
    
    model.train_model(provider, 1, observer)

    Saving.save_bow_model(model, "../models/bowwow.pth") # With love that is my hero bowwow

def train_model():
    params = BoWModelInitParam.get_default_param()
    UseCases.interactive_train(params, "SimpleModel", 10)

def test_model(model_path: str):
    model = Saving.load_bow_model(model_path)
    UseCases.test_model_at_temperatures(model)

if __name__ == "__main__":
    # test_model()
    # generate_model()
    train_model()
    # test_model("../output/SimpleModel/Model_iter_100000.pth")
    # test_model("../output/SimpleModel/Model_iter_1000000.pth")