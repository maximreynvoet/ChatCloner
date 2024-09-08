from Message import Message
from dataclasses import dataclass
from typing import List, Optional

from datatypes.Person import Person
from datatypes.Token import Token
from other.tokenizer import Tokenizer 

@dataclass
class DataPoint:
    prev_tokens: List[Token]
    current_token: Token # TODO remember dat current token een user is als er juist een switch gebeurd is (of andere manier om user change te modelleren)
    current_talker: Optional[Person] # TODO change Optional[Person] -> Person (maak gwn Unknown person die de None verandert); is veel beter voor ML / typing (-V)
    previous_talker: Optional[Person] # TODO change Optional[Person] -> Person (maak gwn Unknown person die de None verandert); is veel beter voor ML / typing (-V)
    time_talked: int

    def is_new_person_talking(self) -> bool:
        return self.time_talked == 0 # TODO klopt dit? Maak test

@dataclass
class ConversationParser:
    conversation: List[Message]

    def parse(self, N : int, tokenizer : Tokenizer) -> List[DataPoint]:
        context_window : List[Token] = []
        conversation_tokens : List[DataPoint] = []
        prev_talker : Optional[Person] = None
        curr_talker : Optional[Person] = None
        time_talked : int = 0

        for message in self.conversation:

            # set the previous talker
            if message.talker != curr_talker:
                prev_talker = curr_talker
                curr_talker = Person(message.talker)
                time_talked = 0
            else:
                time_talked += 1

            curr_tokens : List[Token] = tokenizer.sentence_to_tokens(message.content)

            for curr_token in curr_tokens: 
                
                # create datapoint
                dp : DataPoint = DataPoint(context_window, curr_token, curr_talker, prev_talker, time_talked)
                conversation_tokens.append(dp)

                # sorted first to last 
                context_window.append(curr_token)
                if (len(context_window)) >= N:
                    context_window.pop(0)
        
        # set previous talker of first datapoint to current talker instead of None
        first_dp = conversation_tokens[0]
        first_dp.previous_talker = first_dp.current_talker

        return conversation_tokens        
                