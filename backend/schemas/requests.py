from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

class RunRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    input: list[FiniteFloat] = Field(min_length=1, max_length=4096)
    model_id: str = 'mnist-mlp'

class LayerSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: Literal['Linear', 'ReLU', 'Sigmoid', 'Tanh', 'Softmax', 'Dropout', 'Flatten']
    weight: list[list[FiniteFloat]] | None = None
    bias: list[FiniteFloat] | None = None
    p: float = Field(default=0.5, ge=0, le=1)

class ModelSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(min_length=1, max_length=80)
    input_features: int = Field(ge=1, le=4096)
    layers: list[LayerSpec] = Field(min_length=1, max_length=24)
