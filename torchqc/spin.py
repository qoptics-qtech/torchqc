import torch
import numpy as np
from typing import Self
from torchqc.states import QuantumState
from torchqc.common_matrices import eye, sigmaX, sigmaY, sigmaZ
from torchqc.common_functions import binomial_coef
from torchqc.tensor_product import tensor_product_ops_from_array

def get_sigma(ith, op='Sx', N=10):
    """
    Returns the tensor product of sigma matrix in the ith position and identity matrices for the other positions

    Parameters
    ----------
    ith: int
    op: string (name of the pauli matrix {Sx, Sy, Sz})
    N: number of bodies

    Returns
    -------
    state : QuantumState object of the dicke state
    """

    In = eye(2)
    Sz = (1 / 2) * sigmaZ()
    Sx = (1 / 2) * sigmaX()
    Sy = (1 / 2) * sigmaY()

    args = [In] * N

    if op == 'Sz':
        args[ith - 1] = Sz
    elif op == 'Sx':
        args[ith - 1] = Sx
    elif op == 'Sy':
        args[ith - 1] = Sy
    else:
        raise RuntimeError('operator given not supported', op)

    return tensor_product_ops_from_array(args)

def get_sigma_total(N=10, op='Sz'):
    """
    Returns the sum of sigma matrices (tensor products with identity matrix) for a many body system

    Parameters
    ----------
    op: string (name of the pauli matrix {Sx, Sy, Sz})
    N: number of bodies

    Returns
    -------
    state : QuantumState object of the dicke state
    """

    S_total = None

    for ith in range(N):
        if ith == 0:
            S_total = get_sigma(ith, op, N)
        else:
            S_total += get_sigma(ith, op, N)

    return S_total

def get_dicke_state(n, k) -> QuantumState:
    """
    Returns the kth dicke state of n bodies, D_n_k

    Parameters
    ----------
    n: int
    k: int

    Returns
    -------
    state : QuantumState object of the dicke state
    """

    total_dims = 2**n

    Sx_total = get_sigma_total(n, 'Sx')
    Sy_total = get_sigma_total(n, 'Sy')
    S_minus = Sx_total -1j * Sy_total

    all_spin_up_state = QuantumState.basis(total_dims)[0]

    operators_product = eye(total_dims)

    for i in range(k):
        if i == 0:
            operators_product = S_minus
        else:
            operators_product *= S_minus

    return operators_product.mul(all_spin_up_state).normalize()

def get_dicke_spin_state(j, m) -> QuantumState:
    """
    Get the dicke spin state, j is the total spin number and m the z-axis projection |j,m>

    Parameters
    ----------
    j: float
    m: float

    Returns
    -------
    operator : QuantumState object of the dicke state
    """
    n_spins = int(2 * j)

    return get_dicke_state(n_spins, int(j - m))

def get_spin_coherent_state(j, θ, φ):
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
            sum_term = binomial_coef(int(2 * j), int(j + m)) * (η ** (j + m)) * get_dicke_spin_state(j, m)
        else:
            sum_term += binomial_coef(int(2 * j), int(j + m)) * (η ** (j + m)) * get_dicke_spin_state(j, m)

    return first_term * sum_term