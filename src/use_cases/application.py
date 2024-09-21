from cgi import print_arguments
import random
from datasource.MessageDB import MessageDB
from datasource.conversation_parser import ConversationParser
from datasource.datapoint_provider import DatapointProvider, FixedDatapointProvider
from datatypes.Conversation import Conversation
from datatypes.Person import PersonManager
from machine_learning.BOWInterface import BOWInterface
from machine_learning.BOWModelFactory import BOWModelFactory
from machine_learning.BoWModel import BoWModel
from other.tokenizer import Tokenizer
from utils.examples import Examples
from utils.saving import Saving
from utils.utils import Utils


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
    def tussentijdse_eval(eval_nb):
        if eval_nb % 10_000 == 0: print(interface.predict_convo_continuation(test_convo, 128, 1, 64))
    
    model.train_model(provider, 1, tussentijdse_eval)

    Saving.save_bow_model(model, "../models/bowwow.pth") # With love that is my hero bowwow

def test_model():
    model = Saving.load_bow_model("../models/bowwow.pth")
    conversation = Examples.example_convo
    interface = BOWInterface(model)
    

    for temperature in [0.01, 0.1, 0.3, 0.5, 0.8, 1, 1.2, 1.5, 1.8, 2, 3]:
        predicted = interface.predict_convo_continuation(Conversation(conversation.copy()), 128, temperature, 128)
        print(f"Predicting continuation at temperature {temperature}")
        print(predicted)

    print(5)

    

if __name__ == "__main__":
    # test_model()
    generate_model()