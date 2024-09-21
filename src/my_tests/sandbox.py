from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers

from my_tests.test_examples import TestExamples

def create_tokenizer(text, max_tokens):
    tokenizer = Tokenizer(models.BPE())

    # Use a different pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation()
    ])

    # Keep the ByteLevel post-processor and decoder
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.decoder = decoders.ByteLevel()

    # Configure the trainer
    trainer = trainers.BpeTrainer(
        vocab_size=max_tokens,
        min_frequency=2,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    # Train the tokenizer
    tokenizer.train_from_iterator(text, trainer=trainer)

    return tokenizer

# Usage
# texts = TestExamples.get_example_long_text()
texts = [TestExamples.get_example_long_text()]
# texts = TestExamples.get_example_long_text().split("\n")
max_tokens = 30000  # Adjust as needed
tokenizer = create_tokenizer(texts, max_tokens)

# Test the tokenizer
test_text = "This is a test sentence to see how subwords are handled."
encoded = tokenizer.encode(test_text)
print(encoded.tokens)