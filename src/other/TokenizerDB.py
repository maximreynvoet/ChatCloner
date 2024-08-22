from datasource.MessageDB import MessageDB
from other.tokenizer import Tokenizer


from tokenizers import Tokenizer, models, trainers


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