from dataclasses import dataclass

from datatypes.Message import Message
from datatypes.Person import Person, PersonManager


@dataclass
class MessageFragment:
    token_id: int
    talker_id: int


    def is_message_continuation(self, message: Message) -> bool:
        return self.talker_id == message.get_talker().to_int()
    
    def talker_as_person(self) -> Person:
        return PersonManager.int_to_person(self.talker_id)