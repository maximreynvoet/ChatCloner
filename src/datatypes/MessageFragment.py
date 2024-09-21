from dataclasses import dataclass


@dataclass
class MessageFragment:
    token_id: int
    talker_id: int