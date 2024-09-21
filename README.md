# ChatCloner

## In het kort
1. Basically: steek de json van de chats die je wilt mimicken in de `data/Chats/Messenger` of `data/Chats/Discord` folder
2. Run de `src/use_cases/application.py` (mss moet je wel iets veranderen, idk)
3. ???
4. Enjoy!

Noot: ML is nog zeker niet ok nu. Het urgentste is om nu een echte tokenizer te hebben, en alle todos oplossen, alsook proberen te fixen waarom model niet zo goeie predictions doet (mss normaal dat bag of words op letter niveau niet goed is, maar waarom zoveel repeat van zelfde token ? Wss foutje in de inference pipeline)

## Coding Guidelines
Probeer type hints en een type checker te gebruiken indien je kan.

Als je het niet wist, python biedt een zeer gemakkelijke aanpak om structs te definieren (classes die voornamelijk dienen om data op te slaan (vb Person = record van name: str, age: int))
Je kan dit doen via

```python
from dataclasses import dataclass

@dataclass
class Person:
    age: int
    name: str
```
en zorgt ervoor dat je de lelijke init van velden mag overslaan xx

Als je een static method zou implementeren (zoals in java), vergeet niet dat je jouw code kan / moet annoteren met `@staticmethod` en niet `self` als 1e parameter moet zetten

Het is altijd welgekomen om in elke lus die lang zou duren om uit te voeren (bijvoorbeeld dataset generation, model training, ...) `tqdm` te gebruiken. 
Dit geeft een progress bar in de terminal waarin we kunnen zien / inschatten voor hoe lang het nog zou moeten werken.
tqdm (uitgesproken taqdum (metronoom in het arabisch)) is basically een wrapper voor for lussen

```python
for message in tqdm(messages, "parsing messages"):
    # Do parsing
```
this will give a progress bar for parsing messages.

## Help
In vscode kan je rechts onder (bij python) auto import en type checking activeren
Ook kan je gemakkelijk user snippets genereren

## Folders
- data: jouw data van de chats. Steek daar een Messenger folder en een Discord folder met de data van overeenkomstige platformen
  - ***BELANGRIJK: DEEL DIE ZEKER NIET MET DERDEN !!!***
  - \- V 2024-09-20 19:50
- models: hier slaan we de modellen op