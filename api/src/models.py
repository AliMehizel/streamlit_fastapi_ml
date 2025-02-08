from pydantic import BaseModel
from typing import List



class PredictionRequest(BaseModel):
    age: int
    tumor_size: float
    smoking_status: int 


class DataInput(BaseModel):
    X: List[List[float]]
    y: List[float]

class PredictionInput(BaseModel):
    X: List[List[float]]