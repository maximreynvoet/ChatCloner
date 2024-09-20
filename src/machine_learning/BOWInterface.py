from tqdm import tqdm
from datasource.datapoint_provider import DatapointProvider
from datatypes.Person import Person, PersonManager
from datatypes.datapoint import DataPoint
from datatypes.tensors.pure_tensors import OneHotTensor
from datatypes.tensors.use_case_tensors import TalkerTensor, TokenCountTensor
from datatypes.tensors.ml_tensors import BOWInputTensor, BOWOutputTensor
from machine_learning.BoWModel import BoWModel
from machine_learning.MLInterface import MLInterface

import torch
import torch.optim as optim

from utils.utils import Utils


class BOWInterface(MLInterface):
    """TODO Het is niet omdat nu alle methodes weg zijn dat deze classe voor niets dient. Gebruik het om UI te doen, prompt en van text in -> text uit
    """

    