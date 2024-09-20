from datasource.JsonRepo import JsonRepo
from datasource.MessageReader import MessengerMessageReader, DiscordMessageReader
from datatypes.Message import Message
from datatypes.Platform import Platform


from typing import List, Set, Optional

from datatypes.Person import Person


class MessageDB:
    _messages: List[Message]
    _INSTANCE = None

    def __init__(self, messages: List[Message]) -> None:
        self._messages = messages

    def get_messages(self) -> List[Message]:
        return self._messages

    @staticmethod
    def get_instance() -> "MessageDB":
        # TODO als nood aan aparte db per platform: generate dat
        if MessageDB._INSTANCE is None: MessageDB._INSTANCE = MessageDB._generate_instance()
        return MessageDB._INSTANCE

    @staticmethod
    def _generate_instance() -> "MessageDB":
        
        messenger_messages = [MessengerMessageReader.read_messages_from_json(f) for f in JsonRepo.get_all_messenger_json_chat_paths()]
        discord_messages = [DiscordMessageReader.read_messages_from_json(f) for f in JsonRepo.get_all_discord_json_chats_paths()]
        all_msgs = messenger_messages + discord_messages
        flattened = [m for sublist in all_msgs for m in sublist]
        return MessageDB(flattened)
    
    def get_all_text(self) -> str:
        "Generates all the text from all messages (useful for making tokenizer)"
        return "\n".join([m.content for m in self._messages])
    
    def get_all_talkers(self, platform : Platform) -> Set[Optional[Person]]:
        return set([Person.string_to_person(m.talker, platform) for m in self._messages])