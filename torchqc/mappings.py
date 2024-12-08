import torch
from torchqc.states import QuantumState
from torchqc.operators import Operator

def Lie_bracket(op1: Operator, op2: Operator) -> Operator:
    r"""
    Computes the Lie bracket billinear operation between two operators [op1, op2] := op1 * op2 - op2 * op1

    Parameters
    ----------
    op1 : Operator
    op2 : Operator

    Returns
    -------
    Operator : Operator constructed by the Lie Bracket operation of the two given operators
    """

    return torch.matmul(op1.matrix, op2.matrix) - torch.matmul(op2.matrix, op1.matrix)

def Lie_bracket_matrices(op1: torch.Tensor, op2: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the Lie bracket billinear operation between two tensors [op1, op2] := op1 * op2 - op2 * op1

    Parameters
    ----------
    op1 : torch.Tensor
    op2 : torch.Tensor

    Returns
    -------
    Operator : torch.Tensor constructed by the Lie Bracket operation of the two given operators
    """
    return torch.matmul(op1, op2) - torch.matmul(op2, op1)

def commutator(op1: Operator, op2: Operator) -> Operator:
    r"""
    Computes the commutator billinear operation between two operators [op1, op2] := op1 * op2 - op2 * op1

    Parameters
    ----------
    op1 : Operator
    op2 : Operator

    Returns
    -------
    Operator : Operator constructed by the commutator operation of the two given operators
    """

    return Lie_bracket(op1, op2)

def anticommutator(op1: Operator, op2: Operator) -> Operator:
    r"""
    Computes the Anticommutator billinear operation between two operators {op1, op2} := op1 * op2 + op2 * op1

    Parameters
    ----------
    op1 : Operator
    op2 : Operator

    Returns
    -------
    Operator : Operator constructed by the anticommutator operation of the two given operators
    """

    tensor = torch.matmul(op1.matrix, op2.matrix) + torch.matmul(op2.matrix, op1.matrix)
    return Operator(op1.dims, tensor)