import copy
from datatypes.Message import Message
from typing import List

from datatypes.MessageFragment import MessageFragment
from other.tokenizer import Tokenizer


class Conversation(List[Message]):
    

    def get_last_message(self) -> Message:
        return self[-1]
    
    def add_message_fragment(self, fragment: MessageFragment) -> 'Conversation':    
        last_msg = self.get_last_message()
        str_token = Tokenizer.get_instance().token_to_str(fragment.token_id)
        if fragment.is_message_continuation(last_msg): last_msg.add_content(str_token)
        else: self.append(Message(str_token, fragment.talker_as_person().to_str()))
        return self

    def __str__(self) -> str:
        return "\n".join(map(str, self))
    
    def deep_copy(self) -> "Conversation":
        return copy.deepcopy(self)