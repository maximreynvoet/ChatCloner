from enum import Enum

from tenacity import retry

from datatypes.Platform import Platform

"""
TODO add Nick and other people
Ook een methode om dit van en naar getal te translaten (nodig voor one hot in NN)

TODO mss denken om VictorDiscord en VictorMessenger als verschillend te zien?
TODO is het Nick of Nisse die we hebben als persoon in Messenger ?
"""

class Person(Enum):
    MAXIM = "Maxim"
    VICTOR = "Victor"
    NICK = "Nick"
    UNKNOWN = "Unknown"
    "TODO Ook een methode om dit van en naar getal te translaten (nodig voor one hot in NN)"

    @staticmethod
    def _int_mapping() : 
        "Has to be done via method, otherwise this mapping will be seen as a field -> instance of 'a person'"
        return {
            Person.MAXIM:   1,
            Person.VICTOR:  2,
            Person.NICK:    3,
            Person.UNKNOWN: 4,
        }

    @staticmethod
    def get_nb_persons() -> int:
        return len(list(Person))
    
    @staticmethod
    def get_intvalue_for_unknown() -> int:
        return Person.get_nb_persons()

    @classmethod
    def string_to_person(cls, sender : str, platform : Platform) -> 'Person':
        if platform is Platform.DISCORD: # discord
            
            # using nickname
            if sender == "Vico":
                return Person.VICTOR
            elif sender == "CrusaderMage":
                return Person.MAXIM
            else:
                return Person.UNKNOWN
        
        else: # messenger

            # Facebook Messenger uses full names
            firstname = sender.split(" ")[0]
            return (cls(firstname) if firstname in ["Maxim","Victor", "Nick"] else Person.UNKNOWN)

    @staticmethod
    def person_to_int(person: "Person") -> int:
        return Person._int_mapping().get(person, Person.get_intvalue_for_unknown())
        
    @staticmethod
    def int_to_person(number: int) -> "Person":
        return next((k for k,v in Person._int_mapping().items() if v == number), Person.UNKNOWN)