import os
from typing import List
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, DatasetDict
import torch.nn as nn
from datasource.MessageDB import MessageDB
from datatypes.Conversation import Conversation
from datatypes.SerializedConversation import SerializedConversation
from datatypes.SerializedConversationDB import SerializedConversationDB
from utils.saving import Saving
import transformers

class GPTTokenizer:

    @staticmethod
    def get_instance() -> transformers.tokenization_utils_base.PreTrainedTokenizerBase:
        return GPT2Tokenizer.from_pretrained('gpt2')

class GPTModel(nn.Module):
    """TODO inherit van andere interface of maak een heel nieuwe pipeline"""

    @staticmethod
    def get_pretrained():
        return GPT2LMHeadModel.from_pretrained('gpt2')

class GPTDataset:
    @staticmethod
    def from_convo_list(convos: SerializedConversationDB) -> Dataset: # TODO mss param type as simple conversation if needed
        return Dataset.from_dict({"Conversation": c.to_single_string() for c in convos.get_conversations()})

    @staticmethod
    def tokenize_from_convos(convos: SerializedConversationDB, tokenizer) -> Dataset:
        return Dataset.from_dict({"Conversation": tokenizer(c.to_single_string()) for c in convos.get_conversations()})

data_dir = "../data/SerializedConversations"

def _load_ds_text() -> DatasetDict:
    if not Saving.path_exists(data_dir) or Saving.dir_empty(data_dir): _populate_serialized_dir()
    return load_dataset('text', data_dir= data_dir) # type: ignore

def _populate_serialized_dir():
    for c in SerializedConversationDB.get_instance().get_conversations():
        Saving.write_str_to_file(c.to_single_string(), os.path.join(data_dir, c.name + ".txt"))

class GPTTrainer:

    @staticmethod
    def train_model(model: GPTModel, data: SerializedConversationDB):
        tokenizer = GPTTokenizer.get_instance()
        # ds = GPTDataset.from_convo_list(data)
        # tokenized_ds = list(map(tokenizer, ds)) # TODO perhaps not good type, probably goed mss toch wel

        # tokenized_ds = GPTDataset.tokenize_from_convos(data, tokenizer)
        ds = _load_ds_text()

        tokenizer.pad_token = tokenizer.eos_token
        
        def tokenize_function(examples):
            tokenized_inputs = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
            tokenized_inputs['labels'] = tokenized_inputs['input_ids'].copy()  # Add labels
            return tokenized_inputs
        # tokenize_function = lambda examples: tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

        tokenized_ds = ds.map(tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir='../output/results',
            evaluation_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=3,
            weight_decay=0.01,
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds["train"], # TODO HIHI train en eval zijnt zelfde haha
            # eval_dataset=tokenized_ds,  # TODO HIHI train en eval zijnt zelfde haha
        )

        # Fine-tune the model
        trainer.train()
    
if __name__ == "__main__":
    model = GPTModel.get_pretrained()
    ds = SerializedConversationDB.get_instance()
    GPTTrainer.train_model(model, ds)