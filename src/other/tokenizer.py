from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors

from typing import List
from datasource.MessageDB import MessageDB
from datatypes.Token import Token



class TokenizerDB:
    @staticmethod
    def get_saved_instance(path: str) -> Tokenizer:
        ...

    @staticmethod
    def save_instance(instance: Tokenizer, path: str) -> None:
        ...

    @staticmethod
    def generate_instance(max_tokens: int) -> Tokenizer:
        tokenizer = Tokenizer(models.BPE())
        trainer = trainers.BpeTrainer(vocab_size=max_tokens)
        texts = MessageDB.get_instance().get_all_text()
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return tokenizer


if __name__ == "__main__":
    t = TokenizerDB.generate_instance(1024)
    while 1:
        i = input("Test sentence:    ")
        tokens = t.encode(i)
        reconstructed = t.decode(tokens)
        print(f"{tokens=}\n{reconstructed=}")