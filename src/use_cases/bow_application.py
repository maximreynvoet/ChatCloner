from tqdm import tqdm
from datatypes.Person import PersonManager
from machine_learning.BoWInitParamFactory import BoWInitParamFactory
from other.tokenizer import Tokenizer
from use_cases.UseCases import DatapointSelectorParameters, UseCases
from utils.saving import Saving

def train_model():
    params = BoWInitParamFactory.get_default_params()
    UseCases.interactive_train(params, "SimpleModel", 1)

def hyperparam_search():
    print(Tokenizer.get_instance().get_token_str_mapping_description())
    nb_tokens = Tokenizer.NUMBER_TOKENS
    nb_people = PersonManager.get_nb_persons()
    params = BoWInitParamFactory.get_hyperparam_options(nb_tokens, nb_people)
    print("\n".join([str(x.approx_size()) for x in params]))

    data_selector = DatapointSelectorParameters.get_tiny_context_window_instance(8)
    # data_selector = DatapointSelectorParameters(128, 5_000, True, 50_000)
    provider = UseCases.get_training_provider(data_selector)

    for i, setup in tqdm(enumerate(params), "Testing hyperparameter"):
        UseCases.interactive_train(setup, f"../output/Tiny_Model_{i}_{setup.approx_size()}_params", 1, provider)

def test_model(model_path: str):
    model = Saving.load_bow_model(model_path)
    UseCases.test_model_at_temperatures(model)

def bow_main():
    # test_model()
    # generate_model()
    # train_model()
    hyperparam_search()
    # test_model("../output/Model_0_2112_params/Model_iter_50000.pth")
    # test_model("../output/Model_0_2112_params/Model_iter_25000.pth")
    # test_model("../output/SimpleModel/Model_iter_100000.pth")
    # test_model("../output/SimpleModel/Model_iter_1000000.pth")

if __name__ == "__main__":
    bow_main()
    