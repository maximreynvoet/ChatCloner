from enum import Enum

from tenacity import retry

from datatypes.Platform import Platform

"""
TODO maken van een PersonManager (om puur de enum functionality van translating te scheiden) ?
TODO mss denken om VictorDiscord en VictorMessenger als verschillend te zien?
TODO is het Nick of Nisse die we hebben als persoon in Messenger ?
"""

class Person(Enum):
    MAXIM = "Maxim"
    VICTOR = "Victor"
    NICK = "Nick"
    UNKNOWN = "Unknown"

    def to_int(self)-> int:
        return PersonManager.person_to_int(self)
    
    def to_str(self) -> str:
        return self.value
    
    @staticmethod
    def from_int(i: int) -> 'Person':
        return PersonManager.int_to_person(i)
    
    @staticmethod
    def from_string(sender: str) -> "Person":
        return PersonManager.string_to_person(sender)

class PersonManager:
    _int_mapping = {
        Person.MAXIM:   0,
        Person.VICTOR:  1,
        Person.NICK:    2,
        Person.UNKNOWN: 3,
    }

    @staticmethod
    def get_nb_persons() -> int:
        return len(list(Person))
    
    @staticmethod
    def _get_intvalue_for_unknown() -> int:
        return PersonManager.get_nb_persons()

    _alias_map = {
        "Vico": Person.VICTOR,
        "CrusaderMage": Person.MAXIM,
    }

    @classmethod
    def string_to_person(cls, sender: str) -> Person:
        first_name = sender.split(" ")[0]
        return PersonManager._alias_map.get(first_name) or Person(first_name) or Person.UNKNOWN
        

    @staticmethod
    def person_to_int(person: "Person") -> int:
        return PersonManager._int_mapping.get(person, PersonManager._get_intvalue_for_unknown())
        
    @staticmethod
    def int_to_person(number: int) -> "Person":
        return next((k for k,v in PersonManager._int_mapping.items() if v == number), Person.UNKNOWN)