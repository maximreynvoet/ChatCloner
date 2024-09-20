import torch.nn as nn
import torch

from machine_learning.BoWModel import BoWModel

class Saving:
    
    @staticmethod
    def save_bow_model(model: BoWModel, path: str) -> None:   
        torch.save(
            {
                "state": model.state_dict(),
                "shape_params": model._shape_params
            },
            path)

    @staticmethod
    def load_bow_model(path: str) -> nn.Module:
        checkpoint = torch.load(path)
        loaded_model = BoWModel(checkpoint['shape_params'])
        loaded_model.load_state_dict(checkpoint['state'])
        
        return loaded_model