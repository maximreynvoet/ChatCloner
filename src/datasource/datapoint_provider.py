import random
from typing import Collection, Iterator, List

from attr import dataclass

from datatypes.datapoint import DataPoint
from utils.MixedIterator import MixedIterator

"""TODO to make test / train split: a fold is made via the chat log (just cut off  like 80% of data of message, then translate it to datapoint)

TODO hoeveel is test/train ratio ? 
Voorlopig boeit het niet (denk ik) dat het aan overfitting doet, en dit zou ik zelfs leuker vinden (because funny)

TODO Hmm die moeten een Dataloader van torch implementen denk ik ?

TODO ook de len function implementeren
"""


class DatapointProvider():
    def __iter__(self) -> Iterator[DataPoint]:
        ...

    def __len__(self) -> int:
        ...


class FixedDatapointProvider(DatapointProvider):
    """A datapoint provider that works by first generating all datapoints, and then building an iterator on top of it
    This can be stringent on the memory for bigger datasets"""
    def __init__(self, datapoints: List[DataPoint], shuffle: bool = True) -> None:
        super().__init__()
        self.datapoints = datapoints
        if shuffle: random.shuffle(self.datapoints)

    def __iter__(self) -> Iterator[DataPoint]:
        return iter(self.datapoints)
    
    def __len__(self) -> int:
        return len(self.datapoints)

class SequentialDatapointProvider(DatapointProvider):
    """Provider that does not store the dps explicitly, but generates them on the fly.
    Therefore has lower / no / negligible memory impact.

    Does not support random access
    """


class MixedDatapointProvider(DatapointProvider, MixedIterator):
    """Is a datapoint-provider which is just a random combination of multiple other providers

    Is een aanpak voor random access van datapoints
    """
