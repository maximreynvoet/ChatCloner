import unittest

from other.tokenizer import Tokenizer

from my_tests.test_examples import TestExamples


def test_tokenizer():
    text = TestExamples.get_example_long_text()
    query = TestExamples.get_example_excerpt()
    # tokenizer = Tokenizer._generate_from(text, 128)
    tokenizer = Tokenizer.get_instance()

    mapping = tokenizer.get_token_str_mapping()
    [print(x[1]) for x in mapping]
    

    print(5)

    

if __name__ == "__main__":
    test_tokenizer()
