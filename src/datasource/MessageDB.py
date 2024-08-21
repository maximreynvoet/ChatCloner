from datasource.JsonRepo import JsonRepo
from datasource.MessageReader import MessengerMessageReader
from datatypes.Message import Message


from typing import List


class MessageDB:
    _messages: List[Message]
    _INSTANCE = None

    def __init__(self, messages: List[Message]) -> None:
        self._messages = messages

    def get_messages(self) -> List[Message]:
        return self._messages

    @staticmethod
    def get_instance() -> "MessageDB":
        if MessageDB._INSTANCE is None: MessageDB._INSTANCE = MessageDB._generate_instance()
        return MessageDB._INSTANCE

    @staticmethod
    def _generate_instance() -> "MessageDB":
        messages = [MessengerMessageReader.read_messages_from_json(f) for f in JsonRepo.get_all_messenger_jsons_files()]
        flattened = [m for sublist in messages for m in sublist]
        return MessageDB(flattened)