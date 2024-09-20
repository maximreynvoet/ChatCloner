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
        return BOWModelFactory.get_BOWModel_from_params(params)
    
    @staticmethod
    def get_BOWModel_from_params(params: BoWModelInitParam) -> BoWModel:
        return BoWModel(
            nb_tokens= params.nb_tokens,
            nb_people= params.nb_people,
            
            token_hidden=   FullyConnectedModule(params.token_hidden_seq, params.leaky_relu_slope),
            people_hidden=  FullyConnectedModule(params.people_hidden_seq, params.leaky_relu_slope),
            siamese_hidden= FullyConnectedModule(params.siamese_hidden_seq, params.leaky_relu_slope),
            people_out=     FullyConnectedModule(params.people_out_seq, params.leaky_relu_slope),
            token_out=      FullyConnectedModule(params.token_out_seq, params.leaky_relu_slope),
        )
    
