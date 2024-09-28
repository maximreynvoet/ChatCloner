from dataclasses import dataclass
from typing import List

from datatypes.Conversation import Conversation
from datatypes.Message import Message
from datatypes.SerializedMessage import SerializedMessage

@dataclass
class SerializedConversation:
    messages: List[SerializedMessage]
    name: str
    
    @staticmethod
    def from_conversation(conversation: Conversation) -> 'SerializedConversation':
        return SerializedConversation([x.serialized() for x in conversation], conversation.name)
    
    def to_single_string(self) -> str:
        return "\n".join(self.messages)
    
