import numpy as np

from machine_learning.MLFeatures import MLOutputTensor
from machine_learning.MLFeatures import MLInputTensor

"""
TODO ideas voor future models
- Victor 2024-09-08 08:45

Bag of words model (simpelst)

transformer (overfitting / niet genoeg data)

LSTM
RNN
"""


"TODO: Als finetuning gebruiken: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0"

class TextPredictor:
    "Superclass that offers methods to predict token, be able to train etc..."
    
    def __init__(self, nb_features_in: int, nb_features_out: int) -> None:
        self._nb_features_in  = nb_features_in
        self._nb_features_out = nb_features_out

    def train_model(self) -> None:
        raise NotImplementedError()
    
    # TODO hier dinges
    def predict(self, input_tensor: MLInputTensor) -> MLOutputTensor:
        raise NotImplementedError()

    
