from enum import Enum

from datatypes.Platform import Platform

class Person(Enum):
    MAXIM = "Maxim"
    VICTOR = "Victor"

    @classmethod
    def string_to_person(cls, sender : str, platform : Platform):
        if platform is Platform.DISCORD: # discord
            
            # using nickname
            if sender == "Vico":
                return cls("Victor")
            elif sender == "CrusaderMage":
                return cls("Maxim")
            else:
                return None
        
        else: # messenger

            # Facebook Messenger uses full names
            firstname = sender.split(" ")[0]
            return (cls(firstname) if firstname in ["Maxim","Victor"] else None)