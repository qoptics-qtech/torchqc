import torch
import numpy as np
from torchqc.operators import Operator, DynamicOperator
from torchqc.states import QuantumState
from torchqc.common_matrices import sigmaY, eye
from torchqc.tensor_product import tensor_product_ops

def left_superoperator(op: Operator):
    r"""
    Computes the left superoperator in the Fock-Liouville space (FLS) L(op) of operator op 

    Parameters
    ----------
    op : Operator
        operator

    Method
    ------
    Density matrix is mapped into the Fock-Liouville space (FLS), where it is represented as a vector.
    The left superoperator L(op) is computed by the tensor product: $op \otimes I$.
    The bilinear map Lie bracket can be computed in matrix form by using L(op) and R(op): [H,*] = L(H) - R(H)

    Returns
    -------
    concurrence : float
        float numbers that measures the concurrence.
    """

    return tensor_product_ops(op, eye(op.dims))
                              
def right_superoperator(op: Operator):
    r"""
    Computes the right superoperator in the Fock-Liouville space (FLS) R(op) of operator op 

    Parameters
    ----------
    op : Operator
        operator

    Method
    ------
    Density matrix is mapped into the Fock-Liouville space (FLS), where it is represented as a vector.
    The right superoperator R(op) is computed by the tensor product: $I \otimes op$.
    The bilinear map Lie bracket can be computed in matrix form by using L(op) and R(op): [H,*] = L(H) - R(H)

    Returns
    -------
    concurrence : float
        float numbers that measures the concurrence.
    """

    return tensor_product_ops(eye(op.dims), op.dagger())

def fock_liouville(rho: Operator) -> QuantumState:
    r"""
    Computes the Fock-Liouville vector representation of the density matrix, this vector is element in the Fock-Liouville Space

    Parameters
    ----------
    rho : Operator
        density matrix

    Returns
    -------
    quantum state : QuantumState
        a vector state representation of the density matrix
    """

    return QuantumState(rho.dims ** 2, rho.matrix.flatten())

def lindbladian_operator(H: DynamicOperator|Operator, time: list[float], jump_ops = [], rates = []) -> DynamicOperator|Operator:
    r"""
    Computes the matrix representation of the given operator in the Fock-Liouville Space

    Parameters
    ----------
    H : DynamicOperator|Operator
        operator
    jump_ops: list
        jump operators
    rates: list
        rates for the jump operators

    Returns
    -------
    operator : DynamicOperator|Operator
        the representation of the operator in the FLS
    """

    L = -1j * (left_superoperator(H) - right_superoperator(H))

    for (op, rate) in zip(jump_ops, rates):
        L += rate * (
            left_superoperator(op) * right_superoperator(op.dagger()) - (1 / 2) * (left_superoperator(op.dagger() * op) + right_superoperator(op.dagger() * op))
        ) 
 
    if isinstance(H, DynamicOperator):
        # return DynamicOperator(H.dims * H.dims, L.matrix, time=time)
        return DynamicOperator(H.dims * H.dims, L.matrix)
    else:
        return Operator(H.dims * H.dims, L.matrix)