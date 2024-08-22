from Message import Message
from dataclasses import dataclass
from typing import List
from enum import Enum

from datatypes.Token import Token

class Person(Enum):
    NONE = 0
    MAXIM = 1
    VICTOR = 2

@dataclass
class DataPoint:
    prev_tokens: List[Token]
    current_token: Token # TODO remember dat current token een user is als er juist een switch gebeurd is (of andere manier om user change te modelleren)
    current_talker: Person
    previous_talker: Person
    time_talked: int

@dataclass
class ConversationParser:
    conversation: List[Message]

    def parse(self, N):
        context_window : List[Token] = []
        conversation_tokens : List[DataPoint] = []
        prev_talker : Person = Person(0)
        curr_talker : Person = Person(0)
        time_talked : int = 0

        for message in self.conversation:

            # set the previous talker
            if message.talker != curr_talker:
                prev_talker = curr_talker
                curr_talker = Person(message.talker)
            else:
                time_talked += 1

            curr_tokens : List[Token] = tokenize(message.content)

            for curr_token in curr_tokens: 
                
                # create datapoint
                dp : DataPoint = DataPoint(context_window, curr_token, curr_talker, prev_talker, time_talked)
                conversation_tokens.append(dp)

                # sorted first to last 
                context_window.append(curr_token)
                if (len(context_window)) >= N:
                    context_window.pop(0)
        
        return conversation_tokens        
                