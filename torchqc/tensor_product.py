import torch
from torchqc.states import QuantumState
from torchqc.operators import Operator
from torchqc.common_matrices import eye
import numpy as np
import math

def tensor_product_states(*states) -> QuantumState:
    r"""
    Constructs the tensor product (Kronecker product) of the tensor associated to the given QuantumState instances

    Parameters
    ----------
    *states : QuantumState arguments

    Returns
    -------
    QuantumState : tensor product state
    """

    if (isinstance(states, tuple)):
        states = list(states)

    if len(states) <= 1:
        raise RuntimeWarning("Error: given states should be a tuple or list of two or more states")
    
    dims_array = [state.dims for state in states]
    dims = math.prod(dims_array)
    product_state_tensor = states[0].state_tensor

    for state in states[1:]:
        product_state_tensor = torch.kron(product_state_tensor, state.state_tensor)

    return QuantumState(dims, product_state_tensor, dims_array)

def tensor_product_ops(*operators) -> Operator:
    r"""
    Constructs the tensor product (Kronecker product) of the tensor associated to the given Operator instances

    Parameters
    ----------
    *operators : Operator arguments

    Returns
    -------
    Operator : tensor product operator
    """

    if (isinstance(operators, tuple)):
        operators = list(operators)

    if len(operators) <= 1:
        raise RuntimeWarning("Error: given operators should be a tuple or list of two or more states")
    
    dims_array = [op.dims for op in operators]
    dims = math.prod(dims_array)
    product_tensor = operators[0].matrix

    for operator in operators[1:]:
        product_tensor = torch.kron(product_tensor, operator.matrix)
    
    return Operator(dims, product_tensor, dims_array)

def partial_trace(product_state: Operator, out_dims: list[int]) -> Operator:
    r"""
    Computes the partial trace of a composite Operator (density matrix) 
    with respect the the given dimentions to trace out

    Parameters
    ----------
    product_state : Operator
    out_dims : list[int]

    Returns
    -------
    Operator : partial trace of the given Operator
    """

    # See https://www.ryanlarose.com/uploads/1/1/5/8/115879647/quic06-states-trace.pdf for more details
    # check of given state is product
    if not product_state.is_product:
        raise RuntimeError("Input state product_state is not a tensor product")

    # get trace out nb of dimentions and prepate reduced state
    trace_out_dims = np.prod([product_state.product_dims[i] for i in out_dims])
    
    reduced_dims = product_state.dims // trace_out_dims

    reduced_state = torch.zeros((reduced_dims, reduced_dims), dtype=torch.complex128)

    basis_states = QuantumState.basis(trace_out_dims)

    left_dims = 0
    right_dims = 0
    put_in_left = True

    # loop traced out dimensions and prepare left and right trace out dimensions (if both existing)
    product_dims_idx = 0
    
    for i in range(len(product_state.product_dims)):
        if i not in out_dims:
            put_in_left = False
            product_dims_idx += 1
            continue

        if put_in_left:
            if left_dims == 0:
                left_dims = product_state.product_dims[product_dims_idx]
            else :
                left_dims *= product_state.product_dims[product_dims_idx]
        else:
            if right_dims == 0:
                right_dims = product_state.product_dims[product_dims_idx]
            else:
                right_dims *= product_state.product_dims[product_dims_idx]

        product_dims_idx += 1

    # construct the I matrices for left and right (if needed)
    I_reduced = eye(reduced_dims).matrix

    # loop trace out all basis states and add individual products in the reduces state tensor
    for basis_state in basis_states:
        bra = basis_state.dagger().state_tensor
        ket = basis_state.state_tensor

        left = None
        right = None

        if left_dims == 0:
            left = I_reduced.kron(bra)
            right = I_reduced.kron(ket)
        else:
            left = bra.kron(I_reduced)
            right = ket.kron(I_reduced)

        reduced_state += left.matmul(product_state.matrix).matmul(right)

    return Operator(reduced_dims, reduced_state)