from typing import List
from datatypes.Token import Token
from other.TokenizerDB import TokenizerDB

class Tokenizer:
    def sentence_to_tokens(self, sentence: str) -> List[Token]:
        ...

    def token_to_str(self, token: Token) -> str:
        ...


if __name__ == "__main__":
    t = TokenizerDB.generate_instance(1024)
    while 1:
        i = input("Test sentence:    ")
        tokens = t.encode(i)
        reconstructed = t.decode(tokens)
        print(f"{tokens=}\n{reconstructed=}")