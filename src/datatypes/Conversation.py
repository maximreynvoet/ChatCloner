import copy
from dataclasses import dataclass
from datatypes.Message import Message
from typing import List

from datatypes.MessageFragment import MessageFragment
from other.tokenizer import Tokenizer


@dataclass
class Conversation():
    messages: List[Message]
    name: str

    def get_last_message(self) -> Message:
        return self.messages[-1]
    
    def add_message_fragment(self, fragment: MessageFragment) -> 'Conversation':    
        last_msg = self.get_last_message()
        str_token = Tokenizer.get_instance().token_to_str(fragment.token_id)
        if fragment.is_message_continuation(last_msg): last_msg.add_content(str_token)
        else: self.messages.append(Message(str_token, fragment.talker_as_person().to_str()))
        return self

    def __str__(self) -> str:
        return "\n".join(map(str, self.messages))
    
    def deep_copy(self) -> "Conversation":
        return copy.deepcopy(self)

    @staticmethod
    def generate_name(messages: List[Message]) -> str:
        # TODO change to all participants / constructor
        return messages[0].content[0:50]