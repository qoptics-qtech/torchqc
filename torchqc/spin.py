import torch
import numpy as np
from typing import Self
from torchqc.states import QuantumState
from torchqc.common_matrices import eye, sigmaX, sigmaY, sigmaZ
from torchqc.common_functions import binomial_coef
from torchqc.tensor_product import tensor_product_ops_from_array
from torchqc.operators import Operator

def kronecker_delta(a, b):
    if a == b:
        return 1
    else:
        return 0

def get_generalized_spin_matrix(s=1/2, axis='x'):
    """ if s == 1/2:
        if axis == 'x':
            return (1 / 2) * sigmaX()
        elif axis == 'y':
            return (1 / 2) * sigmaY()
        elif axis == 'z':
            return (1 / 2) * sigmaZ()
        else:
            raise RuntimeError("Axis not valid:", axis) """
    
    dim = int(2*s + 1)
    matrix = torch.zeros((dim, dim), dtype=torch.complex128)

    δab = kronecker_delta

    for a in range(1, dim + 1):
        for b in range(1, dim + 1):
            if axis == 'x':
                matrix[a-1, b-1] = (1 / 2) * (δab(a, b+1) + δab(a+1, b)) * np.sqrt((s+1) * (a + b - 1) - a*b)
            elif axis == 'y':
                matrix[a-1, b-1] = (1j / 2) * (δab(a, b+1) - δab(a+1, b)) * np.sqrt((s+1) * (a + b - 1) - a*b)
            elif axis == 'z':
                matrix[a-1, b-1] = (s + 1 - a) * δab(a,b)
            else:
                raise RuntimeError("Axis not valid:", axis)
    
    return Operator(dim, matrix)
        

def get_sigma(ith, s=1/2, axis='x', N=10):
    """
    Returns the tensor product of sigma matrix in the ith position and identity matrices for the other positions

    Parameters
    ----------
    ith: int
    axis: string (axis of the pauli matrix {x, y, z})
    N: number of bodies

    Returns
    -------
    state : QuantumState object of the dicke state
    """

    dims = int(2*s + 1)

    In = eye(dims)

    #Sz = (1 / 2) * sigmaZ()
    #Sx = (1 / 2) * sigmaX()
    #Sy = (1 / 2) * sigmaY()

    Sx = get_generalized_spin_matrix(s, 'x')
    Sy = get_generalized_spin_matrix(s, 'y')
    Sz = get_generalized_spin_matrix(s, 'z')

    args = [In] * N

    if axis == 'z':
        args[ith - 1] = Sz
    elif axis == 'x':
        args[ith - 1] = Sx
    elif axis == 'y':
        args[ith - 1] = Sy
    else:
        raise RuntimeError('operator given not supported', axis)

    return tensor_product_ops_from_array(args)

def get_sigma_total(N=10, s=1/2, axis='z'):
    """
    Returns the sum of sigma matrices (tensor products with identity matrix) for a many body system

    Parameters
    ----------
    axis: string (axis of the pauli matrix {x, y, z})
    s: spin number
    N: number of bodies

    Returns
    -------
    state : QuantumState object of the dicke state
    """

    S_total = None

    for ith in range(N):
        if ith == 0:
            S_total = get_sigma(ith, s, axis, N)
        else:
            S_total += get_sigma(ith, s, axis, N)

    return S_total

def get_dicke_state(s, n, k) -> QuantumState:
    """
    Returns the kth dicke state of n bodies, D_n_k

    Parameters
    ----------
    s: float
    n: int
    k: int

    Returns
    -------
    state : QuantumState object of the dicke state
    """

    # total_dims = 2**n
    total_dims = int((2*s + 1)**n)

    Sx_total = get_sigma_total(n, s, 'x')
    Sy_total = get_sigma_total(n, s, 'y')
    S_minus = Sx_total -1j * Sy_total

    all_spin_up_state = QuantumState.basis(total_dims)[0]

    operators_product = eye(total_dims)

    for i in range(k):
        if i == 0:
            operators_product = S_minus
        else:
            operators_product *= S_minus

    return operators_product.mul(all_spin_up_state).normalize()

def get_dicke_spin_state(n_spins, s, j, m) -> QuantumState:
    """
    Get the dicke spin state, s is the spin number, j is the total spin number and m the z-axis projection |j,m>

    Parameters
    ----------
    s: float
    j: float
    m: float

    Returns
    -------
    operator : QuantumState object of the dicke state
    """
    # n_spins = int(2 * j)

    return get_dicke_state(s, n_spins, int(j - m))

def get_spin_coherent_state(n_spins, s, j, θ, φ):
    """
    Follows definition from https://doi.org/10.1016/j.physrep.2011.08.003
    Get the spin coherent state, j is the total spin number and m the z-axis projection |j,m>

    Parameters
    ----------
    j: float
    θ: float
    φ: float

    Returns
    -------
    operator : QuantumState object of the dicke state
    """

    η = - np.tan(θ / 2) * np.exp(-1j * φ)

    first_term = (1 + np.abs(η)**2)**(-j)
    sum_term = None

    projections = []
    i = -j

    while i <= j:
        projections.append(i)
        i += 1

    for m in projections:
        if sum_term is None:
            sum_term = binomial_coef(int(2 * j), int(j + m)) * (η ** (j + m)) * get_dicke_spin_state(n_spins, s, j, m)
        else:
            sum_term += binomial_coef(int(2 * j), int(j + m)) * (η ** (j + m)) * get_dicke_spin_state(n_spins, s, j, m)

    return first_term * sum_term