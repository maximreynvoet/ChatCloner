from machine_learning.BoWModel import BoWModel
from machine_learning.BoWModelInitParam import BoWModelInitParam
from machine_learning.fully_connected import FullyConnectedModule

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

        nb_tokens: ongeveer 256 ish
        nb_people: 4-8
        """
        params = BoWModelInitParam.get_default_param(nb_people=nb_people, nb_tokens= nb_tokens)
        return BoWModel(params)
    
    
