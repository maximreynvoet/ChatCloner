
from tqdm import tqdm
from datatypes.Person import PersonManager
from machine_learning.BoWInitParamFactory import BoWInitParamFactory
from other.tokenizer import Tokenizer
from use_cases.UseCases import UseCases
from utils.saving import Saving


def train_model():
    params = BoWInitParamFactory.get_default_params()
    UseCases.interactive_train(params, "SimpleModel", 1)

def hyperparam_search():
    nb_tokens = Tokenizer.NUMBER_TOKENS
    nb_people = PersonManager.get_nb_persons()
    params = BoWInitParamFactory.get_hyperparam_options(nb_tokens, nb_people)
    for i, setup in tqdm(enumerate(params), "Testing hyperparameter"):
        UseCases.interactive_train(setup, f"Model_{i}_{setup.approx_size()}_params", 1)

def test_model(model_path: str):
    model = Saving.load_bow_model(model_path)
    UseCases.test_model_at_temperatures(model)

if __name__ == "__main__":
    # test_model()
    # generate_model()
    train_model()
    # test_model("../output/SimpleModel/Model_iter_100000.pth")
    # test_model("../output/SimpleModel/Model_iter_1000000.pth")