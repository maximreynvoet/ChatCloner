from datatypes.Message import Message

import json
from typing import List, Union


class MessageReader:
    @staticmethod
    def read_messages_from_json(json_file: str) -> List[Message]:
        raise NotImplemented


class MessengerMessageReader(MessageReader):
    @staticmethod
    def _json_element_to_msg(json_msg: dict) -> Union[Message, None]:
        "Returns a message object based on the json element, if valid (is only text), otherwise None"
        return Message(json_msg["text"], json_msg["senderName"]) if json_msg["type"] == "text" else None

    @staticmethod
    def read_messages_from_json(json_file: str) -> List[Message]:
        with open(json_file, "r") as f:
            d = json.load(f)
            l = [MessengerMessageReader._json_element_to_msg(msg) for msg in d["messages"]]
            return [x for x in l if x is not None]