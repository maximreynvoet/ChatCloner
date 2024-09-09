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
