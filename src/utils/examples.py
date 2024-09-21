from datatypes.Conversation import Conversation
from datatypes.Message import Message
from datatypes.Person import Person


class Examples:
    "A small db of a pair of examples"

    victor_yo_msg = Message("Yoooo", Person.VICTOR.to_str())

    example_convo = Conversation([victor_yo_msg])
