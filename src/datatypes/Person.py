from enum import Enum

class Person(Enum):
    NONE = "None"
    MAXIM = "Maxim"
    VICTOR = "Victor"

    def __init__(self, sender : str) -> None:
        super().__init__()

    @classmethod
    def string_to_person(cls, sender : str, platform : str):
        if platform == "discord":
            
            # using nickname
            if sender == "Vico":
                return cls("Victor")
            elif sender == "CrusaderMage":
                return cls("Maxim")
            else:
                return cls("None")
        else:

            # Facebook Messenger uses full names
            return cls(sender.split(" ")[0])