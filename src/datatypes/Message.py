from attr import dataclass


@dataclass
class Message:
    content: str
    talker: str # TODO omvormen naar persoon      