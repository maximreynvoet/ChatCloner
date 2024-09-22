from machine_learning.BoWInitParamFactory import BoWInitParamFactory
from machine_learning.BoWModel import BoWModel
from machine_learning.BoWModelInitParam import BoWModelInitParam

class BOWModelFactory:
    @staticmethod
    def get_default_epmty_instance(nb_tokens: int, nb_people: int) -> BoWModel:
        """Returns an instance of a BoWModel
        TODO BOW FActory comment over hyperparam search door de bayesian optimization
            - https://en.wikipedia.org/wiki/Neural_architecture_search
            - Bayesian Optimization
            - random search
            - grid search
            - genetic algorithm (als echt yolo; zeer zeer duur, vele evaluaties nodig)
        """
        return BOWModelFactory.get_model_from_params(
            BoWInitParamFactory.get_default_params()
        )
    
    @staticmethod
    def get_model_from_params(params: BoWModelInitParam) -> BoWModel:
        return BoWModel(params)