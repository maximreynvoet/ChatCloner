from typing import List



class JsonRepo:
    @staticmethod
    def get_all_messenger_jsons_files() -> List[str]:
        return ["/home/vico_ptp/Documents/lelz/Texts/Chats/" + x + ".json" for x in [
            "Hallo De Vlaamse Regering Heeft U Nodig",
            "Victor Corne",
            "Victor CorneandNick Atillinois",
        ]]


