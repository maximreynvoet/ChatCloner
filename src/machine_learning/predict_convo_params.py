from dataclasses import dataclass

from datatypes.Conversation import Conversation


@dataclass
class PredictConvoParams:
    "Simple utility class to bundle parameters for model prediction"
    
    nb_fragments: int
    window_size: int
    temperature: float

    @staticmethod
    def get_default_params() -> 'PredictConvoParams':
        return PredictConvoParams(64, 128, 1)