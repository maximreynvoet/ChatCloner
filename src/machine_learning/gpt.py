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
        """
        TODO in trainer
            - Logging strategy `"steps"`: Logging is done every `logging_steps`.
            - Saven elke X iters
            Lezen wat de params zijn in de template file
            Lezen wat de mogelijke params zijn in de doc
        """

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
    """
    TODO eens de output lezen 
        - 2024-09-28 12:42:31.961535: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
        - 2024-09-28 12:42:32.021214: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
        - To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
        - 2024-09-28 12:42:34.183699: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
        - /home/vico_ptp/.local/lib/python3.8/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
        - warnings.warn(
        - /home/vico_ptp/.local/lib/python3.8/site-packages/transformers/training_args.py:1525: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
        - warnings.warn(
        - Detected kernel version 5.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.

    TODO verstaan wat de code doet (beetje blind gecopy pasted oeioei)
        -V 2024-09-28

    TODO model opslaan en inferences mee doen (beetje zelfde script als BOWModel Trainen)
    TODO GPU gebruiken indien mogelijk
    TODO traint het model door voorbeelden at random?
    TODO traint het model wel efficient door niet telkens van 0 te beginnen, maar door de vorige input verder op te bouwen ?
    """
    model = GPTModel.get_pretrained()
    ds = SerializedConversationDB.get_instance()
    GPTTrainer.train_model(model, ds)