
import random
from typing import Iterator, List, TypeVar

T = TypeVar("T")


class MixedIterator(Iterator[T]):
    """This is an iterator that is a collection of multiple iterators.
    When retrieving a value from this iterator: it returns the value of one of its iterators at random (if still available), or returns a StopIteration if all are empty"""
    _iterators: List[Iterator[T]]

    def __next__(self) -> T:
        if not self._iterators:
            raise StopIteration  # All providers are exhausted

        # Randomly select a provider
        while self._iterators:
            provider = random.choice(self._iterators)

            try:
                # Try to get the next datapoint from the chosen provider
                return next(provider)
            except StopIteration:
                # If the provider is exhausted, remove it from the list
                self._iterators.remove(provider)

        # If all providers are exhausted, raise StopIteration
        raise StopIteration
