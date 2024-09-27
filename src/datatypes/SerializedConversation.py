from typing import List

from datatypes.Conversation import Conversation
from datatypes.Message import Message

class SerializedConversation(List[str]):
    
    @staticmethod
    def from_conversation(conversation: Conversation) -> 'SerializedConversation':
        return SerializedConversation([SerializedConversation.serialize_message(x) for x in conversation])
    
    @staticmethod
    def serialize_message(msg: Message) -> str:
        return f"[{msg.get_talker().to_str()}]: {msg.content}"
