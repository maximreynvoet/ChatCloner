from enum import Enum

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