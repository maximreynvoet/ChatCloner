import random
from datasource.MessageDB import MessageDB
from datasource.conversation_parser import ConversationParser
from datasource.datapoint_provider import DatapointProvider, FixedDatapointProvider
from datatypes.Person import PersonManager
from machine_learning.BOWModelFactory import BOWModelFactory
from machine_learning.BoWModel import BoWModel
from other.tokenizer import Tokenizer
from utils.saving import Saving
from utils.utils import Utils


def generate_model():
    # TODO al die fn in een aparte, iets om gemakkelijker te initializeren
    # TODO ja ik weet geen test / split, maar dat is hier voor de lol
    nb_tokens = Tokenizer.NUMBER_TOKENS
    nb_people = PersonManager.get_nb_persons()
    tokenizer = Tokenizer.generate_tokenizer(nb_tokens)
    window_size = 128 # BOW model -> we are able to have a large window
    
    messages = MessageDB.get_instance().get_messages()
    
    datapoints = ConversationParser().parse(messages, window_size, tokenizer)
    # augmented_datapoints = Utils.flatmap(datapoints, lambda x: x.get_all_split(1), True)
    augmented_datapoints = datapoints
    random.shuffle(augmented_datapoints)
    provider = FixedDatapointProvider(augmented_datapoints)
    
    model = BOWModelFactory.get_default_epmty_instance(nb_tokens, nb_people)

    model.train_model(provider, 1)

    Saving.save_bow_model(model, "../models/bowwow.pth") # With love that is my hero bowwow

def test_model():
    model = Saving.load_bow_model("../models.bowwow.pth")
    

if __name__ == "__main__":
    generate_model()