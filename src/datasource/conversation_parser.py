from tqdm import tqdm
from datatypes.Message import Message
from datatypes.Person import Person, PersonManager
from datatypes.Token import Token
from datatypes.datapoint import DataPoint
from other.tokenizer import Tokenizer


from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ConversationParser:

    @staticmethod
    def parse(messages: List[Message], N: int, tokenizer: Tokenizer) -> List[DataPoint]:
        """
        Parses a list of messages to a collection of datapoints.

        args:
            N: the length of the sliding window (hoeveel tokens er in een datapoint mogen)
        ---

        TODO meer uitleg over wat de list is als output. Is het laatste element in de lijst de meest recente message?
            Is input message van oud -> nieuw ?
            Dit allemaal documenteren en kijken dat de callers het goed doen

        TODO (indien reworking nodig): vervangen door functional approach (zie onderstaand messenger bericht):
        Niet ik die in mijn slaap een puur functional approach van map reduce heb gevonden 😅
        Map messages naar tuple van <token, author>, neem sliding window van size n erop, en vanuit die sliding window kan je bepalen wie de auteur is
        Aja fack kan niet weten voor hoe lang de current talker aanwezig was, alhoewel toch wel met een incremental iets
        Update: ja je zou wel incrementeel iets kunnen doen voor duur hoe lang spreker aanwezig is (gwn zijnde vorige count + 1 als talkers zelfde, anders 1)

        """
        context_window: List[Token] = []
        conversation_tokens: List[DataPoint] = []
        prev_talker = Person.UNKNOWN
        curr_talker = Person.UNKNOWN
        time_talked: int = 0

        for message in tqdm(messages, "Parsing messages to datapoint"):

            # set the previous talker
            if message.talker != curr_talker:
                prev_talker = curr_talker
                curr_talker = PersonManager.string_to_person(message.talker)
                time_talked = 0
            else:
                time_talked += 1

            curr_tokens: List[Token] = tokenizer.sentence_to_tokens(message.content)

            for curr_token in curr_tokens.ids:

                # create datapoint
                dp = DataPoint(context_window, curr_token, curr_talker, prev_talker, time_talked)
                conversation_tokens.append(dp)

                # sorted first to last
                context_window.append(curr_token)
                if (len(context_window)) >= N:
                    context_window.pop(0)

        # set previous talker of first datapoint to current talker instead of None
        first_dp = conversation_tokens[0]
        first_dp.previous_talker = first_dp.current_talker

        return conversation_tokens
