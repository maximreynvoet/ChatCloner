from machine_learning.BoWModel import BoWModel
from machine_learning.fully_connected import FullyConnectedModule
from utils.sequences import Sequences


class BOWModelFactory:
    @staticmethod
    def get_default_epmty_instance(nb_tokens: int, nb_people: int) -> "BoWModel":
        """Returns an instance of a BoWModel

        TODO -> factory method
        In de factory: gemakkelijk hyperparameters toevoegen (om hyperparameter optimization te doen)
            Deze params hier: mogelijks te veel, mogelijks te weinig, geen idee /shrug
            -V 2024-09-19

        TODO mss ook betere API om fullyconnected te maken 
            - FullyConnected.FromSequence(start, nb_repeats)
            - FullyConnected.FromPowerLaw(stars, end, power, (max_len))

        TODO BOW FActory comment over hyperparam search door de bayesian optimization
            - https://en.wikipedia.org/wiki/Neural_architecture_search
            - Bayesian Optimization
            - random search
            - grid search
            - genetic algorithm (als echt yolo)

        nb_tokens: ongeveer 256 ish
        nb_people: 4-8
        """

        token_siamese_in = 64
        people_siamese_in = nb_people
        people_repeat = 4
        siamese_repeat = 4
        siamese_in = token_siamese_in + people_siamese_in
        siamese_out = siamese_in

        return BoWModel(
                        nb_tokens=nb_tokens,
                        nb_people=nb_people,

                        token_hidden=   FullyConnectedModule(Sequences.power_sequence(nb_tokens, token_siamese_in, 0.75)),
                        people_hidden=  FullyConnectedModule(Sequences.repeat_sequence(nb_people, people_repeat)),
                        siamese_hidden= FullyConnectedModule(Sequences.repeat_sequence(siamese_in, siamese_repeat)),
                        people_out=     FullyConnectedModule(Sequences.power_sequence(siamese_out, nb_people, 0.5)),
                        token_out=      FullyConnectedModule(Sequences.power_sequence(siamese_out, nb_tokens, 0.5)),
                        )