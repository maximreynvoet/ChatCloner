from typing import List

from torch import Tensor
from datatypes.Token import Token
from other.TokenizerDB import TokenizerDB

class Tokenizer:
    def sentence_to_tokens(self, sentence: str) -> List[Token]:
        ...

    def token_to_str(self, token: Token) -> str:
        ...

    def get_nb_tokens(self) -> int:
        ...

    @staticmethod
    def get_instance() -> "Tokenizer":
        ...

    "TODO dit w beetje overal accessed, niet goed, maar beste manier tot nu toe om niet overal de tokenizer te moeten laten passeren"
    NUMBER_TOKENS = 256

if __name__ == "__main__":
    t = TokenizerDB.generate_tokenizer(1024)
    while 1:
        i = input("Test sentence:    ")
        tokens = t.encode(i)
        reconstructed = t.decode(tokens)
        print(f"{tokens=}\n{reconstructed=}")