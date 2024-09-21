"Plot the number of datapoints in the dataset, according to the window size"

from typing import List, Tuple

from tqdm import tqdm
from datasource.MessageDB import MessageDB
from datasource.conversation_parser import ConversationParser
from datatypes.Person import PersonManager
from other.tokenizer import Tokenizer


def get_data() -> List[Tuple[int, int]]:
    nb_tokens = Tokenizer.NUMBER_TOKENS
    tokenizer = Tokenizer._generate_tokenizer(nb_tokens)
    messages = MessageDB.get_instance().get_messages()

    res = []
    
    for window_size in tqdm([1,4,8,16,32,64,128], "Testing window sizes"):
        datapoints = ConversationParser().parse(messages, window_size, tokenizer)
        res.append((window_size, len(datapoints)))
    
    return res

if __name__ == "__main__":
    print(get_data())