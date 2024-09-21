from dataclasses import dataclass

from datatypes.Message import Message
from datatypes.Person import Person, PersonManager
from other.tokenizer import Tokenizer


@dataclass
class MessageFragment:
    token_id: int
    talker_id: int


    def is_message_continuation(self, message: Message) -> bool:
        return self.talker_id == message.get_talker().to_int()
    
    
    def token_as_str(self) -> str:
        """TODO een manier om hieruit str en talker_name te krijgen, op betere manier dan dit (?)
        Geen idee hoe haraam dit is hier?
        """
        return Tokenizer.get_instance().token_to_str(self.token_id)
    
    def talker_as_person(self) -> Person:
        return PersonManager.int_to_person(self.talker_id)