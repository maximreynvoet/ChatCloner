from datatypes.Conversation import Conversation
from datatypes.Message import Message
from datatypes.Person import Person


class Examples:
    "A small db of a pair of examples"

    victor_yo_msg = Message("Yoooo",      Person.VICTOR.to_str())
    maxim_cv = Message("Cava nog mejou?", Person.MAXIM.to_str())
    nick_msg = Message("Ik heb daar zeer veel respect voor xx", Person.NICK.to_str())
    victor_punchline= Message("Op het einde winnen we I guess", Person.VICTOR.to_str())

    example_convo = Conversation([victor_yo_msg, maxim_cv, victor_punchline])
