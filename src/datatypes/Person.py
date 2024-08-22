from enum import Enum

class Person(Enum):
    NONE = "None"
    MAXIM = "Maxim"
    VICTOR = "Victor"

    @staticmethod
    def string_to_person(sender : str, platform : str):
        if platform == "discord":
            
            # using nickname
            if sender == "Vico":
                return Person("Victor")
            elif sender == "CrusaderMage":
                return Person("Maxim")
            else:
                return Person("None")
        else:

            # Facebook Messenger uses full names
            return Person(sender.split(" ")[0])