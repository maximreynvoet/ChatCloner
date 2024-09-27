from typing import List

from datatypes.Conversation import Conversation
from datatypes.Message import Message
from datatypes.SerializedMessage import SerializedMessage

class SerializedConversation(List[SerializedMessage]):
    
    @staticmethod
    def from_conversation(conversation: Conversation) -> 'SerializedConversation':
        return SerializedConversation([x.serialized() for x in conversation])
    
    def to_single(self) -> str:
        return "\n".join(self)
    
