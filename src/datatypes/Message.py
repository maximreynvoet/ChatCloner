from dataclasses import dataclass

from datatypes.Person import Person, PersonManager


@dataclass
class Message:
    content: str
    talker: str # TODO omvormen naar persoon

    def get_talker(self) -> Person:
        # TODO untested, niet zeker dat niet altijd unk geeft
        return PersonManager.string_to_person(self.talker)

