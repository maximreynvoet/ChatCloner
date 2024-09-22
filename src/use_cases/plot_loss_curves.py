
from dataclasses import dataclass
import os
from typing import List, Optional

from matplotlib import pyplot as plt
import numpy as np

from utils.saving import Saving

output_dir = "../output"

@dataclass
class _ModelLoss:
    model_name: str
    losses: List[float]

    @staticmethod
    def from_dir(dir_path: str) -> '_ModelLoss':
        return _ModelLoss(dir_path, _load_loss_data(dir_path+"/Losses.txt"))

def smooth_data(data: List[float], window_size: int) -> np.ndarray:
    window = np.ones(window_size) / window_size
    # Convolve the data with the window
    smoothed_data = np.convolve(data, window, mode='valid')
    return smoothed_data

def _try_convert_to_float(x: str) -> Optional[float]:
    try: return float(x)
    except: return None

def _load_loss_data(file_path: str) -> List[float]:
    data = [_try_convert_to_float(x) for x in Saving.load_strings(file_path)]
    return [x for x in data if x is not None]

def get_all_losses() -> List[_ModelLoss]:
    return [_ModelLoss.from_dir(output_dir + "/" + d) for d in os.listdir(output_dir)]

def plot_losses(smoothing: int, losses: List[_ModelLoss]):
    for ml in losses:
        data = smooth_data(ml.losses, smoothing)
        plt.plot(np.log(data), label= ml.model_name)
    plt.title(f"Model losses (smoothing={smoothing})")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # plot_losses(3000, get_all_losses())
    plot_losses(25_000, [_ModelLoss.from_dir("../output/Tiny_Model_0_2112_params")])
