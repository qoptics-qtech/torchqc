import torch
import numpy as np
from torchqc.operators import Operator, DynamicOperator
from torchqc.states import QuantumState
from torchqc.common_matrices import sigmaY
from torchqc.tensor_product import tensor_product_ops
from torchqc.dynamics import lindblad_equation, lindblad_equation_FLS
from torchqc.fock_liouville import left_superoperator, right_superoperator

def get_density_matrix(state: QuantumState) -> Operator:
    """
    Computes the density matrix of the given state vector

    Parameters
    ----------
    state: QuantumState

    Returns
    -------
    operator : Operator representation of the density matrix
    """

    rho_tensor = torch.outer(state.state_tensor.view(state.dims), state.state_tensor.view(state.dims))

    density_operator =  Operator(state.dims, rho_tensor)

    if state.is_product:
        density_operator.is_product = True
        density_operator.product_dims = state.product_dims

    return density_operator

def expect_val(state: QuantumState|list[QuantumState], operator: Operator):
    """
    Computes the expectation values for a given state or a list of states

    Parameters
    ----------
    state: QuantumState|list[QuantumState]

    Returns
    -------
    expectation value : one or list of expection values for the given operator
    """

    if isinstance(state, QuantumState):
        mapped_state = operator.mul(state)
        if operator.is_hermitian():
            return torch.real(state.inner_product(mapped_state))
        else:
            return state.inner_product(mapped_state)
    elif isinstance(state, list):
        if operator.is_hermitian():
            return torch.tensor([torch.real(st.inner_product(operator.mul(st))) for st in state])
        else:
            return torch.tensor([st.inner_product(operator.mul(st)) for st in state])

def expect_val_dm(state, operator):
    """
    Computes the expectation values for a given state or a list of states

    Parameters
    ----------
    state: Operator|list[Operator]

    Returns
    -------
    expectation value : one or list of expection values for the given operator
    """

    if type(state) == Operator:
       if operator.is_hermitian():
            return torch.real(torch.trace(state.opmul(operator).matrix))
       else:
           return torch.trace(state.opmul(operator).matrix)
    elif isinstance(state, list):
        if operator.is_hermitian():
            # return [torch.real(torch.trace(st.opmul(operator).matrix)) for st in state]
            return torch.tensor([torch.real(torch.trace(st.opmul(operator).matrix)) for st in state])
        else:
            # return [torch.trace(st.opmul(operator).matrix) for st in state]
            return torch.tensor([torch.trace(st.opmul(operator).matrix) for st in state])
        
def outer_product(one: QuantumState, other: QuantumState) -> Operator:
    """
    compute the outer product between two quantum states

    Parameters
    ----------
    one: QuantumState
    other: QuantumState

    Returns
    -------
    operator : Operator with tensor the outer product of the given quantum state tensors
    """

    return Operator(dims=one.dims, matrix=torch.outer(one.state_tensor.view(one.dims), other.state_tensor.view(other.dims)))

def basis_operators(dims: int):
    """
    function that returns the basis operators, transition and projector operators

    Parameters
    ----------
    dims: int

    Returns
    -------
    operators : tuple of operators and labels
        three element tuple with projector, transition operators and labels (projectors, transitions, labels)
    """

    basis_states = QuantumState.basis(dims)
    
    projectors = []
    transitions = []
    labels = []

    for i in range(dims):
        proj = outer_product(basis_states[i], basis_states[i])
        projectors.append(proj)
        labels.append(f"proj{i+1}")

        for j in range(i+1, dims):
           trans = outer_product(basis_states[i], basis_states[j])
           transitions.append(trans)
           labels.append(f"trans{i+1}{j+1}")
    
    return (projectors, transitions, labels)

def bell_states():
    """
    function that creates the bell states of a two qubit composite system

    Parameters
    ----------
    NONE

    Returns
    -------
    bell_states : list[QuantumState]
        A list of the bell states
    """

    basis_states = QuantumState.basis(dims=4)

    bell1 = (basis_states[0] + basis_states[3]).normalize()
    bell2 = (basis_states[0] - basis_states[3]).normalize()
    bell3 = (basis_states[1] + basis_states[2]).normalize()
    bell4 = (basis_states[1] - basis_states[2]).normalize()

    states = [bell1, bell2, bell3, bell4]

    return states

def fidelity(state1: QuantumState|Operator, state2: QuantumState|Operator):
    """
    function that computes the fidelity between two quantum states

    Parameters
    ----------
    state1: QuantumState|Operator 
    state2: QuantumState|Operator

    Returns
    -------
    fidelity : float
        float number that measures the fideluty, takes values between 0 and 1
    """

    if type(state1) != type(state2):
        raise Exception(f"Error: the two given tensors are not the same type: {type(state1)} != {type(state2)}")
    
    if isinstance(state1, QuantumState):
        return torch.abs(state1.inner_product(state2)) ** 2
    elif isinstance(state1, Operator):
        return torch.abs(torch.trace(state2.opmul(state1).matrix))
    else:
        raise Exception(f"Error: unknown state type: {type(state1)}")
    
def herm_operator_function(operator: Operator, func):
    eigenvals, eigenvecs = torch.linalg.eig(operator.matrix)
    
    # apply the function in the eigenvalues
    eigenvals = func(eigenvals)

    # use spectral theorem to compute function of operator
    return Operator(operator.dims, eigenvecs.matmul(torch.diag(eigenvals)).matmul(eigenvecs.transpose(0, 1).conj()))

def concurrence(state: QuantumState|Operator) -> float:
    """
    function that computes the entanglement metric concurrence of a given quantum state.

    Parameters
    ----------
    state : QuantumState|Operator
        A product quantum state (state vector or density matrix)

    Returns
    -------
    concurrence : float
        float numbers that measures the concurrence.
    """

    operator = tensor_product_ops(sigmaY(), sigmaY())

    if isinstance(state, QuantumState):
        tilde_state = operator.mul(QuantumState(state.dims, state.state_tensor.conj()))

        return torch.abs(state.inner_product(tilde_state))
    elif isinstance(state, Operator): # TODO: check if correct
        conj_operator = Operator(state.dims, state.matrix.conj())
        tilde_state = operator.opmul(conj_operator).opmul(operator)

        # get sqrt of density operator
        sqrt_rho = herm_operator_function(state, torch.sqrt)

        # construct Hermitian matrix R
        R = sqrt_rho.opmul(tilde_state).opmul(sqrt_rho)
        R = herm_operator_function(R, torch.sqrt)

        eigenvals = torch.linalg.eigvals(R.matrix)
        eigenvals_sorted = torch.sort(torch.real(eigenvals), descending=True).values

        eigvals_minus = eigenvals_sorted[0]
        for val in eigenvals_sorted[1:]:
            eigvals_minus -= val

        # return max between subtract value of eigenvals and zero
        return torch.max(torch.stack((eigvals_minus, torch.tensor(0.))))

    # return max between subtract value of eigenvals and zero
    return torch.max(torch.stack((eigvals_minus, torch.tensor(0.))))

def trace_distance(rho: Operator, sigma: Operator):
    return (1 / 2) * torch.trace(torch.sqrt(((rho - sigma).dagger().opmul(rho - sigma)).matrix))

def steadystate(H: Operator, jump_ops: list[Operator], rates: list[float], method='SVD', initial_state: Operator=None) -> Operator:
    r"""
    \frac{d\hat{\rho}_{ss}}{dt}=\mathcal{L}\hat{\rho}_{ss}=0.
    """

    # Solve Lψ = 0
    # Construct superoperator
    # we need to get the matrix representation of the superoperator
    # Example: Super operator of the hamiltonian involced in the Lie bracket:  
    #          L(H) = (left_superoperator(H) - right_superoperator(H))[rho] + left_superoperator(H)[rho]
    L = -1j * (left_superoperator(H) - right_superoperator(H.dagger()))

    for (op, rate) in zip(jump_ops, rates):
        L += rate * (
            left_superoperator(op) * right_superoperator(op.dagger()) 
            - (1 / 2) * (
                left_superoperator(op.dagger() * op) + right_superoperator(op.dagger() * op)
            )
        )

    if method == "SVD":
        # Perform SVD in the lindbladian matrix
        U, S, Vh = torch.linalg.svd(L.matrix)
        V = torch.transpose(torch.adjoint(Vh), 0, 1)

        # steady state is the last vector of V matrix, normalized by trace
        rho = V[-1].reshape(H.dims, H.dims)
        rho_ss = rho / rho.trace()

        return Operator(H.dims, rho_ss)
    elif method == "eigen":
        raise NotImplementedError("Not implemented yet")
        """ A = L.dagger() * L

        vals, vecs = torch.linalg.eig(A.matrix)

        min_index = torch.argmin(torch.abs(vals))
        rho_ss = vecs[min_index]

        rho_ss = rho_ss.reshape((H.dims, H.dims))

        rho_ss = rho_ss / rho_ss.trace()

        return Operator(H.dims, rho_ss) """
    elif method == "inf":

        if initial_state is None:
            initial_state = get_density_matrix(QuantumState.basis(H.dims)[0])

        rho0 = initial_state
        
        rho = rho0
        Dt = 100
        time = np.arange(0, 2 * Dt, Dt, dtype = np.float32)
        trace_dist = 100.
        states = None

        while trace_dist > 1e-5:
            rho0 = rho
            hamiltonian = DynamicOperator(H.dims, H, time=time)
            _, states = lindblad_equation_FLS(rho, hamiltonian, time, Dt, jump_ops, rates)
            rho = states[-1]
            trace_dist = torch.real(trace_distance(rho, rho0)).numpy()

        return rho

    else:
        raise RuntimeError(f"Method {method} not implemented!")