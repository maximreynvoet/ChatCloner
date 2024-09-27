import random
from datasource.JsonRepo import JsonRepo
from datasource.MessageReader import MessengerMessageReader, DiscordMessageReader
from datatypes.Conversation import Conversation
from datatypes.Message import Message
from datatypes.Platform import Platform


from typing import List, Set, Optional

from datatypes.Person import Person
from utils.examples import Examples


class MessageDB:
    """TODO Functionality to have one conversation per chat message log (ene met enkel Ik + Maxim, ene met enkel Ik + Nick, ene voor groups chat, ...)
    TODO save as list conversations, not simple messages
    """

    _INSTANCE = None

    def __init__(self, messages: List[Message]) -> None:
        self._messages = messages
        self._conversations = ...

    def get_messages(self) -> List[Message]:
        return self._messages
    
    def get_message_sample(self, nb_messages: int) -> List[Message]:
        if nb_messages > 0: return self.get_test_messages(nb_messages, Examples.RANDOM_SEED)
        else: return self.get_messages()
    
    def get_nb_messages(self) -> int:
        return len(self._messages)
    
    def get_test_messages(self, nb_messages: int, random_seed: int) -> List[Message]:
        if nb_messages > self.get_nb_messages(): return self._messages
        r = random.Random(random_seed)
        msg_range = len(self._messages) - nb_messages - 1
        start_pt = r.randint(0, msg_range)
        return self._messages[start_pt:start_pt+nb_messages]

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
    
    def get_all_ascii_text(self) -> str:
        return "\n".join([m.content.encode('ascii', 'ignore').decode('ascii') for m in self._messages])
    
    def get_all_talkers(self, platform : Platform) -> Set[Optional[Person]]:
        return set([Person.string_to_person(m.talker, platform) for m in self._messages])