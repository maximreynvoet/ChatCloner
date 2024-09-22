import os
import torch.nn as nn
import torch

from machine_learning.BoWModel import BoWModel

def _create_path_if_not_exist(path: str) -> None:
    # Check if the path refers to a file or directory
    if os.path.splitext(path)[1]:  # If path has a file extension
        directory = os.path.dirname(path)  # Extract the directory part of the file path
    else:
        directory = path  # If it's a directory
    
    # Create the directory if it doesn't exist
    if directory:
        os.makedirs(directory, exist_ok=True)


class Saving:
    
    @staticmethod
    def save_bow_model(model: BoWModel, path: str) -> None:
        _create_path_if_not_exist(path)
        torch.save(
            {
                "state": model.state_dict(),
                "shape_params": model._shape_params
            },
            path)

    @staticmethod
    def load_bow_model(path: str) -> BoWModel:
        checkpoint = torch.load(path)
        loaded_model = BoWModel(checkpoint['shape_params'])
        loaded_model.load_state_dict(checkpoint['state'])
        
        return loaded_model
    
    @staticmethod
    def write_str_to_file(string: str, file: str) -> None:
        _create_path_if_not_exist(file)
        with open(file, "a") as f:
            f.write(string)

    @staticmethod
    def clear_file(file: str) -> None:
        try:
            with open(file, "w") as f:
                pass
        except:
            pass # Could not clear a file who does not exist -> no problem
