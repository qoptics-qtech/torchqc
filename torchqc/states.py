import torch
import numpy as np
from typing import Self
import math

"""
Class that represents a quantum state
"""
class QuantumState:
    dims: int
    state_tensor: torch.Tensor
    is_dual: False
    is_product: False

    def __init__(self, dims: int, vector: torch.Tensor, product_dims = None) -> None:
        r"""
        Initializes a quantum state instance

        Parameters
        ----------
        dims : int
            number of dimensions for the Hilbert space in which the state vector belongs
        vector: torch.Tensor
            a rank-1 tensor representing the vector
        product_dims: list[int]
            a list of product dimension in case the state is a tensor product state

        Returns
        -------
        None
        """
        
        self.dims = dims
        self.state_tensor = vector
        self.is_product = True
        self.product_dims = product_dims

    def __repr__(self) -> str:
        return f"QuantumState (dims = {self.dims} , tensor: {self.state_tensor})"
    
    def __add__(self, other) -> Self:
        if isinstance(other, np.ndarray):
            if self.dims != len(other):
                raise Exception("The two vector do not have the same dims")
            
            self.state_tensor += torch.from_numpy(other)
        elif isinstance(other, QuantumState):
            if self.dims != other.dims:
                raise Exception("The two vector do not have the same dims")
            
            new_state_tensor = self.state_tensor + other.state_tensor

            return QuantumState(self.dims, new_state_tensor)
        
    def __sub__(self, other) -> Self:
        if isinstance(other, np.ndarray):
            if self.dims != len(other):
                raise Exception("The two vector do not have the same dims")
            
            self.state_tensor -= torch.from_numpy(other)
        elif isinstance(other, QuantumState):
            if self.dims != other.dims:
                raise Exception("The two vector do not have the same dims")
            
            new_state_tensor = self.state_tensor - other.state_tensor

            return QuantumState(self.dims, new_state_tensor)
        
    def __mul__(self, other) -> Self:
        if isinstance(other, (int, float, complex)):
            return QuantumState(self.dims, other * self.state_tensor)
        else:
            return NotImplemented
        
    def __rmul__(self, other) -> Self:
        if isinstance(other, (int, float, complex)):
            return QuantumState(self.dims, other * self.state_tensor)
        else:
            return NotImplemented

    def inner_product(self, other: Self) -> torch.Tensor:
        r"""
        QuantumState mehtod that computes and returns the inner product between the self and the other given state vector

        Parameters
        ----------
        self : Operator
            self instance
        other: Operator
            hamiltonian operator

        Returns
        -------
        inner product value : torch.Tensor
        """
        
        # return torch.inner(self.state_tensor.view(self.dims), other.state_tensor.view(other.dims))
        return self.dagger().state_tensor.matmul(other.state_tensor)
    
    def norm(self) -> torch.Tensor:
        # return torch.norm(self.state_tensor)
        return torch.sqrt(torch.real(self.inner_product(self)))
    
    def populations(self) -> torch.Tensor:
        return torch.abs(self.state_tensor) ** 2
    
    def __eq__(self, item: Self) -> bool:
        return (self.dims == item.dims) and (torch.equal(self.state_tensor, item.state_tensor))
    
    def normalize(self) -> Self:
        self.state_tensor /= torch.norm(self.state_tensor)
        return self
    
    @staticmethod
    def basis(dims):
        basis_states = []

        for i in range(dims):
            # basis_state = np.zeros(dims, dtype=np.complex128)
            basis_state = torch.zeros((dims, 1), dtype=torch.complex128)
            basis_state[i] = 1. + 0.j
            # basis_states.append(QuantumState(dims, torch.from_numpy(basis_state)))
            basis_states.append(QuantumState(dims, basis_state))

        return basis_states
    
    @staticmethod
    def coherent(dims, a: complex) -> Self:
        tensor = torch.zeros((dims, 1), dtype=torch.complex128)

        for n in range(dims):
            tensor[n] = math.exp(- abs(a)**2 / 2) * (a**n / math.sqrt(math.factorial(n)))

        return QuantumState(dims, tensor)
    
    def dagger(self) -> Self:
        return QuantumState(self.dims, self.state_tensor.transpose(0, 1).conj())