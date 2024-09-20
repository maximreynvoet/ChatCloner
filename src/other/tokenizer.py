from typing import List

from attr import dataclass

from datatypes.Token import Token
from tokenizers import Tokenizer as tkTokenizer, models, trainers
from datasource.MessageDB import MessageDB

@dataclass
class Tokenizer:
    _model: tkTokenizer

    "TODO dit w beetje overal accessed, niet goed, maar beste manier tot nu toe om niet overal de tokenizer te moeten laten passeren"
    NUMBER_TOKENS = 256

    def sentence_to_tokens(self, sentence: str) -> List[Token]:
        return self._model.encode(sentence)

    def token_to_str(self, token: Token) -> str:
        return self._model.decode(token)

    def tokens_to_str(self, tokens: List[Token]) -> str:
        return self._model.decode(tokens)
    
    def get_nb_tokens(self) -> int:
        "TODO ezo nie e, retrieve van model zelf"
        return Tokenizer.NUMBER_TOKENS

    @staticmethod
    def get_instance() -> "Tokenizer":
        ...

    @staticmethod
    def generate_tokenizer(max_tokens: int) -> 'Tokenizer':
        tokenizer = tkTokenizer(models.BPE())
        trainer = trainers.BpeTrainer(vocab_size=max_tokens)
        texts = MessageDB.get_instance().get_all_text()
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return Tokenizer(tokenizer)



if __name__ == "__main__":
    t = Tokenizer.generate_tokenizer(1024)
    while 1:
        i = input("Test sentence:    ")
        tokens = t.sentence_to_tokens(i)
        reconstructed = t.tokens_to_str(tokens)
        print(f"{tokens=}\n{reconstructed=}")