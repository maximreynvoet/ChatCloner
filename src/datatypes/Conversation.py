from datatypes.Message import Message


from dataclasses import dataclass
from typing import List


class Conversation(List[Message]): # TODO gebruiken als type hint overal waar nodig
    ...