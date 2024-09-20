from typing import List

from datatypes.Token import Token
from tokenizers import Tokenizer as tkTokenizer, models, trainers
from datasource.MessageDB import MessageDB

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

    @staticmethod
    def generate_tokenizer(max_tokens: int) -> 'Tokenizer':
        tokenizer = tkTokenizer(models.BPE())
        trainer = trainers.BpeTrainer(vocab_size=max_tokens)
        texts = MessageDB.get_instance().get_all_text()
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return tokenizer

    "TODO dit w beetje overal accessed, niet goed, maar beste manier tot nu toe om niet overal de tokenizer te moeten laten passeren"
    NUMBER_TOKENS = 256

if __name__ == "__main__":
    t = Tokenizer.generate_tokenizer(1024)
    while 1:
        i = input("Test sentence:    ")
        tokens = t.encode(i)
        reconstructed = t.decode(tokens)
        print(f"{tokens=}\n{reconstructed=}")