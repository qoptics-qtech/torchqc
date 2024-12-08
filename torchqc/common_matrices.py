import torch
import numpy as np
from torchqc.operators import Operator
from torchqc.states import QuantumState

def sigmaX(dims=2):
    """
    computes the matrix representation of the x-axis pauli operator

    Parameters
    ----------
    dims: int
        matrix dimensions
   
    Returns
    -------rising
    operator : x pauli operator
    """
    return Operator(dims, torch.from_numpy(np.array([[0.j, 1], [1, 0]])))

def sigmaY(dims=2): 
    """
    computes the matrix representation of the y-axis pauli operator

    Parameters
    ----------
    dims: int
        matrix dimensions
   
    Returns
    -------rising
    operator : y pauli operator
    """
    return Operator(dims, torch.from_numpy(np.array([[0., -1.j], [1.j, 0]])))

def sigmaZ(dims=2):
    """
    computes the matrix representation of the z-axis pauli operator

    Parameters
    ----------
    dims: int
        matrix dimensions
   
    Returns
    -------rising
    operator : z pauli operator
    """
    return Operator(dims, torch.from_numpy(np.array([[1., 0.j], [0., -1.]])))

def sigmaPlus(dims=2):
    """
    computes the matrix representation of the 2-lvl rising operator

    Parameters
    ----------
    dims: int
        matrix dimensions
   
    Returns
    -------rising
    operator : lowering operator
    """
    return Operator(dims, torch.from_numpy(np.array([[0.j, 0.], [1., 0.]])))

def sigmaMinus(dims=2):
    """
    computes the matrix representation of the 2-lvl lowering operator

    Parameters
    ----------
    dims: int
        matrix dimensions
   
    Returns
    -------
    operator : lowering operator
    """
    return Operator(dims, torch.from_numpy(np.array([[0.j, 1.], [0., 0.]])))

def annihilation(dims: int) -> Operator:
    r"""
    computes the matrix representation of the annihilation operator a that appears in the quantum harmonic oscillator and lowers the fock number state by one

    Parameters
    ----------
    dims: int
        matrix dimensions
   
    Returns
    -------
    operator : creation operator
    """

    tensor = torch.zeros((dims, dims), dtype=torch.complex128)

    for i in range(dims):
        for j in range(dims):
            if i + 1 == j:
                tensor[i][j] = np.sqrt(i + 1)

    return Operator(dims, tensor)

def creation(dims: int):
    r"""
    computes the matrix representation of the creation operator a^\dagger that appears in the quantum harmonic oscillator and rises the fock number state by one

    Parameters
    ----------
    dims: int
        matrix dimensions
   
    Returns
    -------
    operator : creation operator
    """

    tensor = torch.zeros((dims, dims), dtype=torch.complex128)

    for i in range(dims):
        for j in range(dims):
            if i == j + 1:
                tensor[i][j] = np.sqrt(j + 1)

    return Operator(dims, tensor)

def eye(dims: int):
    r"""
    computes the matrix representation of the identity operator

    Parameters
    ----------
    dims: int
        matrix dimensions
   
    Returns
    -------
    operator : identity operator
    """

    tensor = torch.eye(dims, dtype=torch.complex128)

    return Operator(dims, tensor)

def displacement_operator(N: int, b: complex) -> Operator:
    r"""
    computes the matrix representation of the displacement operator e^{b a^\dagger - b* a}
    It create the coherent state from the zero fock state
    \ket{a} = D(b) \ket{0},\text{ where }b \in \mathbb{C}

    Parameters
    ----------
    N: int
        fock dimensions
    b: complex:
        complex number assiciated with the displacement operator

    Returns
    -------
    operator : Displacement operator
    """
    a = annihilation(N)
    a_dagger = creation(N)

    return Operator(N, torch.linalg.matrix_exp(b * a_dagger.matrix - np.conjugate(b) * a.matrix))