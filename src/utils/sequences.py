from functools import reduce
from operator import mul
from typing import List


class Sequences:
    """Class that offers utilities for making sequences
    
    TODO ja ik weet dat method names niet de beste zijn
    -V
    """

    @staticmethod
    def power_sequence(start: int, end: int, power: float) -> List[int]:
        """Returns the sequence that starts at "start" and ends at "end", where every value decreases via the harmonic sequence of power "power"
        """
        "TODO ook een max_length van de sequence als argument (zo verbied je te lang en de NN teveel params)"
        assert start > 0 and end > 0
        assert power != 0
        
        res = [start]
        if start > end: # Decreasing sequence
            power = 1 / power if power > 1 else power
            next_val = res[-1] ** power
            while next_val > end:
                res.append(int(next_val))
                next_val = res[-1] ** power
                

        if start < end: # Increasing sequence
            power = 1/power if power < 1 else power
            next_val = res[-1] ** power
            while next_val < end:
                res.append(int(next_val))
                next_val = res[-1] ** power

        
        if res[-1] != end: res.append(end)
        return res
    
    @staticmethod
    def linear_sequence_size(start: int, end: int, size: int) -> List[int]:
        """Returns the sequence of size size, starting with start and ending with end, where every value changes linearly"""
        if size == 1: return [start, end]
        step = (end - start) / (size-1)
        return [int(start + i * step) for i in range(size)]
    
    @staticmethod
    def linear_sequence_step(start: int, end: int, step: int) -> List[int]:
        """Returns a sequence starting with `start` and ending with `end`, where values change linearly by a given `step`."""
        if start < end: step = abs(step)
        else: step = -abs(step)
        if start == end or step == 0: return [start, end]

        sequence = []
        current = start
        if step > 0:
            while current <= end:
                sequence.append(current)
                current += step
        else:
            while current >= end:
                sequence.append(current)
                current += step
        
        if sequence[-1] != end: sequence.append(end)

        return sequence
        
    @staticmethod
    def repeat_sequence(value: int, repeat: int) -> List[int]:
        "Returns the list that is 'value' repeated 'repeat' times "
        return [value] * repeat
    
    @staticmethod
    def encoder_sequence(input_size: int, output_size: int, latent_size: int, length_in: int, length_out: int) -> List[int]:
        input_seq = Sequences.linear_sequence_step(input_size, latent_size, length_in)
        output_seq = Sequences.linear_sequence_step(latent_size, output_size, length_out)
        return input_seq + output_seq[1:] # Do not count the double int in between
    
    @staticmethod
    def sequence_product(sequence: List[int]) -> int:
        return reduce(mul, sequence, 1)
        
def test_reduce_power():
    "TODO waar moeten de testen ?"
    s = Sequences.power_sequence(64, 1, 0.5)
    assert s == [64,32,16,8,4,2,1]

def test_linear():
    assert Sequences.linear_sequence_size(0, 10, 6) == [0,2,4,6,8,10]
    assert Sequences.linear_sequence_size(10, 20, 3) == [10, 15, 20]

    assert Sequences.linear_sequence_size(10, 0, 6) == [10,8,6,4,2,0]
    assert Sequences.linear_sequence_size(20, 10, 3) == [20, 15, 10]

    assert Sequences.linear_sequence_step(0,5,1) == [0,1,2,3,4,5]
    assert Sequences.linear_sequence_step(0,5,2) == [0,2,4,5]
    assert Sequences.linear_sequence_step(0,5,3) == [0,3,5]
    
    assert Sequences.linear_sequence_step(5,0,1) == [5,4,3,2,1,0]
    assert Sequences.linear_sequence_step(5,0,2) == [5,3,1,0]
    assert Sequences.linear_sequence_step(5,0,3) == [5,2,0]
    print("Test succes!")

if __name__ == "__main__":
    test_linear()
