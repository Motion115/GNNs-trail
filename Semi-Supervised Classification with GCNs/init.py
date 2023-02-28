from typing_extensions import Literal, TypedDict
from typing import Callable, List
from torch import Tensor 

LossFn = Callable[[Tensor, Tensor], Tensor] 
Stage = Literal["train", "val", "test"] 

class HistoryDict(TypedDict): 
    training_loss: List[float] 
    training_accuracy: List[float] 
    validation_loss: List[float] 
    validation_acc: List[float] 