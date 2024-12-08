import torch
import numpy as np
from typing import Self
from torchqc.states import QuantumState
from types import FunctionType
import warnings
import traceback

class Operator:
    dims: int
    matrix: torch.Tensor
    is_hermitian: bool
    is_unitary: bool
    is_dynamic: bool
    is_product: False

    def __init__(self, dims, matrix: torch.Tensor, product_dims = None) -> None:
        self.dims = dims
        self.matrix = matrix
        self.is_product = True
        self.product_dims = product_dims

    def __repr__(self) -> str:
        return f"Operator (dims = {self.dims}, tensor: {self.matrix})"

    def __eq__(self, other: Self):
        return torch.equal(self.matrix, other.matrix)

    def __add__(self, other: Self) -> Self:
        if self.dims != other.dims:
            raise Exception("The two operators dimensions do not match!")

        return Operator(dims=self.dims, matrix=self.matrix + other.matrix)
    
    def __sub__(self, other: Self) -> Self:
        if self.dims != other.dims:
            raise Exception("The two operators dimensions do not match!")

        return Operator(dims=self.dims, matrix=self.matrix - other.matrix)

    def mul(self, vector: QuantumState):
        new_tensor = torch.matmul(self.matrix, vector.state_tensor) 
        return QuantumState(self.dims, new_tensor)
    
    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            new_tensor = other * self.matrix
            return Operator(self.dims, new_tensor)
        elif isinstance(other, Operator):
            return self.opmul(other)
        else:
            return NotImplemented
        
    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            new_tensor = other * self.matrix
            return Operator(self.dims, new_tensor)
        elif isinstance(other, torch.Tensor):
            new_tensor = other * self.matrix
            return Operator(self.dims, new_tensor)
        else:
            return NotImplemented
    
    def opmul(self, operator: Self):
        if self.dims != operator.dims:
            raise Exception("Error: the two operators have no matching dimensions!!")

        new_tensor = torch.matmul(self.matrix, operator.matrix)
        return Operator(self.dims, new_tensor)

    def dagger(self):
        # return Operator(self.dims, torch.conj(torch.transpose(self.matrix, 0, 1)).resize(self.dims, self.dims))
        return Operator(self.dims, torch.conj(torch.transpose(self.matrix, 0, 1)).contiguous().view(self.dims, self.dims))

    def is_hermitian(self):
        return (self == self.dagger())
    
    def is_unitary(self):
        identity = Operator(self.dims, torch.eye(self.dims))
        return (self.opmul(self.dagger()) == identity)
    
class DynamicOperator(Operator):
    def __init__(self, dims, Ht: FunctionType|torch.Tensor|Operator = None, params: list = [], time: np.ndarray|torch.Tensor = None) -> None:
        self.dims = dims

        if isinstance(Ht, (FunctionType, Operator)) and time is None:
            raise Exception("Ht and time must be both defined")
        elif isinstance(Ht, torch.Tensor) and time is not None:
            warnings.warn("If Ht is a torch.Tensor, time parameter is ignored")

        # if time param is np array, then convert it to a torch.Tensor
        if isinstance(time, np.ndarray):
            time_tensor = torch.from_numpy(time).reshape(len(time), 1)
        else:
            time_tensor = time
        
        # Then construct the matrix field of the DynamicOperator instance based on Ht type
        if isinstance(Ht, FunctionType):    
            self.matrix = torch.stack([Ht(t, params) for t in time_tensor])
        elif isinstance(Ht, torch.Tensor):
            self.matrix = Ht
        elif isinstance(Ht, Operator):
            self.matrix = torch.stack([Ht.matrix for t in time_tensor])
        else:
            raise RuntimeError("Error: Ht can be FunctionType|torch.Tensor|Operator types")

    def dagger(self):
        return Operator(self.dims, torch.conj(torch.transpose(self.matrix, 1, 2)).contiguous().view(self.matrix.shape[0], self.dims, self.dims))