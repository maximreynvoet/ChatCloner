from typing import List
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch.nn as nn
from datasource.MessageDB import MessageDB
from datatypes.Conversation import Conversation
from datatypes.SerializedConversation import SerializedConversation
from datatypes.SerializedConversationDB import SerializedConversationDB


class GPTTokenizer:

    @staticmethod
    def get_instance():
        return GPT2Tokenizer.from_pretrained('gpt2')

class GPTModel(nn.Module):
    """TODO inherit van andere interface of maak een heel nieuwe pipeline"""

    @staticmethod
    def get_pretrained():
        return GPT2LMHeadModel.from_pretrained('gpt2')

class GPTDataset:
    @staticmethod
    def from_convo_list(convos: SerializedConversationDB) -> Dataset: # TODO mss param type as simple conversation if needed
        return Dataset.from_dict({"Conversation": c.to_single() for c in convos})

class GPTTrainer:

    @staticmethod
    def train_model(model: GPTModel, data: SerializedConversationDB):
        tokenizer = GPTTokenizer.get_instance()
        ds = GPTDataset.from_convo_list(data)
        tokenized_ds = list(map(tokenizer, ds)) # TODO perhaps not good type, probably goed mss toch wel

        training_args = TrainingArguments(
            output_dir='./results',
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
            train_dataset=tokenized_ds, # TODO HIHI train en eval zijnt zelfde haha
            eval_dataset=tokenized_ds, # TODO HIHI train en eval zijnt zelfde haha
        )

        # Fine-tune the model
        trainer.train()
    
if __name__ == "__main__":
    model = GPTModel.get_pretrained()
    ds = SerializedConversationDB.get_instance()
    GPTTrainer.train_model(model, ds)