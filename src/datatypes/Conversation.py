from datatypes.Message import Message


from dataclasses import dataclass
from typing import List

from datatypes.MessageFragment import MessageFragment


class Conversation(List[Message]): # TODO gebruiken als type hint overal waar nodig
    

    def get_last_message(self) -> Message:
        return self[-1]
    
    def add_message_fragment(self, fragment: MessageFragment) -> 'Conversation':    
        last_msg = self.get_last_message()
        if fragment.is_message_continuation(last_msg): last_msg.add_content(fragment.token_as_str())
        else: self.append(Message(fragment.token_as_str(), fragment.talker_as_person().to_str()))
        return self

    def __str__(self) -> str:
        return "\n".join(map(str, self))