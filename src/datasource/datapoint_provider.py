from typing import Iterator

from datatypes.datapoint import DataPoint
from utils.MixedIterator import MixedIterator

"""TODO to make test / train split: a fold is made via the chat log (just cut off  like 80% of data of message, then translate it to datapoint)

TODO hoeveel is test/train ratio ? 
Voorlopig boeit het niet (denk ik) dat het aan overfitting doet, en dit zou ik zelfs leuker vinden (because funny)
"""


class DatapointProvider(Iterator[DataPoint]):
    ...


class FixedDatapointProvider(DatapointProvider):
    """A datapoint provider that works by first generating all datapoints, and then building an iterator on top of it
    This can be stringent on the memory for bigger datasets"""


class SequentialDatapointProvider(DatapointProvider):
    """Provider that does not store the dps explicitly, but generates them on the fly.
    Therefore has lower / no / negligible memory impact.

    Does not support random access
    """


class MixedDatapointProvider(DatapointProvider, MixedIterator):
    """Is a datapoint-provider which is just a random combination of multiple other providers

    Is een aanpak voor random access van datapoints
    """
