
from typing import List
from datasource.MessageDB import MessageDB
from datatypes.Conversation import Conversation
from datatypes.SerializedConversation import SerializedConversation


class SerializedConversationDB(List[SerializedConversation]):
    
    @staticmethod
    def get_instance() -> 'SerializedConversationDB':
        c = SerializedConversation.from_conversation(
            Conversation(MessageDB.get_instance().get_messages())
        )
        return SerializedConversationDB([c])
        
