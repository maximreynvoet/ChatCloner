# ChatCloner

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

## Help
In vscode kan je rechts onder (bij python) auto import en type checking activeren
Ook kan je gemakkelijk user snippets genereren