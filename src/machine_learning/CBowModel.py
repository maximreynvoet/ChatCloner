from datatypes.tensors.ml_tensors import BOWInputTensor
from datatypes.tensors.ml_tensors import CBOWInputTensor
from datatypes.tensors.ml_tensors import CBOWOutputTensor
from datatypes.tensors.use_case_tensors import TokenProbabilityTensor
from machine_learning.TextPredictor import PytorchTextPredictor
import torch.nn as nn

"""
TODO organize

TODO Het kan eens kino zijn om cbow model te maken
CBOW = +- BOW maar waarbij je niet "gewoon de word counts" neemt, maar eerst elk woord (in de window) gaat embedden naar een lagere dimensie
    Ik denk dat dit een lineaire operatie kan zijn (lin projectie) naar emb dimensie
Je neemt een aggregate / pooling (sum of avg) van de embeddings van alle vectoren in de window

Dit is eig meer om de representation te leren van een woord (een embedding)
[] Kan je ook de context nemen van een woord om de representation van said word te bepalen?

Hiervoor goed om elk woord een token te nemen
    inb4 de indian character spam en andere singleton tokens deleten

Omdat embedding size niet teee groot -> efficienter om trainen / gebruiken


- Niet zeeer veel moeilijker te trainen (is eigenlijk zeer kino + goeie oefening in NLP)

-V 2024-10-03 23:37


---

Kan ook kino zijn om meer te doen voor word representation learning
- CBOW is cool
- Skip-gram kan ook cool zijn
    Basically taak is leer als woord Y in de context window is van X (anchor)
    En hieruit leer je de representation
- Nog andere 


"""
class CBowModel(PytorchTextPredictor):
    
    def __init__(self, nb_tokens: int, embedding_size: int) -> None:
        super(CBowModel, self).__init__()
        self._nb_tokens = nb_tokens
        self._embedding_size = embedding_size
        self._embedder = nn.Embedding(nb_tokens, embedding_size)
        self._ffc = nn.Linear(self._embedding_size, self._nb_tokens)

    def forward(self, dp: CBOWInputTensor) -> CBOWOutputTensor:
        embeddings = [self._embedder(t) for t in dp.tokens]
        embedding_sum = sum(embeddings)
        token_distribution = self._ffc(embedding_sum)
        return CBOWOutputTensor(TokenProbabilityTensor(token_distribution))