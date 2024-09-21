from dataclasses import dataclass

from datatypes.Message import Message


@dataclass
class MessageFragment:
    token_id: int
    talker_id: int

    def is_message_continuation(self, message: Message) -> bool:
        return self.talker_id == message.get_talker().to_int()