import os
from typing import List

_CHAT_FOLDER = "../data/Chats/"

class JsonRepo:
    @staticmethod
    def get_all_messenger_json_chat_paths() -> List[str]:
        return JsonRepo._get_all_json_files_in_folder(_CHAT_FOLDER + "Messenger")
    
    @staticmethod
    def get_all_discord_json_chats_paths() -> List[str]:
        return JsonRepo._get_all_json_files_in_folder(_CHAT_FOLDER + "Discord")

    @staticmethod
    def _get_all_json_files_in_folder(folder: str) -> List[str]:
        return [os.path.join(folder, x) for x in os.listdir(folder) if x.endswith(".json")]