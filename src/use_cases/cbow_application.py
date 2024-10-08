from machine_learning.CBowInterFace import CBowInterFace
from machine_learning.CBowModel import CBowModel
from machine_learning.training_observers.TrainingObserverFactory import TrainingObserverFactory
from other.tokenizer import Tokenizer
from use_cases.UseCases import DatapointSelectorParameters, UseCases
from utils.IntRange import IntRange


def train_see_cbow():
    tokenizer = Tokenizer.get_instance()
    selection_params = DatapointSelectorParameters(10, -1, False, -1)
    provider = UseCases.get_training_provider(selection_params)
    range = IntRange(3,10)
    model = CBowModel(tokenizer.get_nb_tokens(), 64)
    dir = "../output/cbow"
    interface = CBowInterFace(model, range)
    model.train_model(provider, 1, TrainingObserverFactory.get_default_train_observers(dir, interface, 25_000))

def cbow_main():
    train_see_cbow()

if __name__ == "__main__":
    cbow_main()