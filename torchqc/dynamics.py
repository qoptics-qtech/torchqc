import torch
import numpy as np
from torchqc.states import QuantumState
from torchqc.operators import Operator, DynamicOperator
from torchqc.markovian import SuperOperatorActed
from torchqc.mappings import Lie_bracket_matrices
from torchqc.common_matrices import sigmaZ
from torchqc.fock_liouville import fock_liouville, lindbladian_operator
from torchqc.runge_kutta_methods import runge_kutta4, runge_kutta45
from torchqc.adams_bashforth_methods import adams_bashforth_2step, adams_bashforth_4step, adams_bashforth_12step
from types import FunctionType

import numpy as np

def TDSE(initial_state: QuantumState, hamiltonian: Operator, time: np.ndarray, Dt: float) -> list:
    r"""
    TDSE(initial_state: QuantumState, hamiltonian: Operator, time: np.array, Dt: float) -> list
    
    Returns a list of states as the qunatum system evolves in time
    
    The shapes of the :attr:`initial_state` and the :attr:`hamiltonian` tensor need
    to match.
    
    Args:
        initial_state (QuantumState): the initial quantum state.
        hamiltonian  (Operator): hamiltonian matrix or matrices
        time (np.ndarray): time in discrete time steps
        Dt (float): time step duration
    """

    time_tensor = torch.from_numpy(time).reshape(len(time), 1)
    time_tensor.requires_grad_(True)

    current_state = initial_state
    states = [initial_state]

    for i in range(len(time_tensor) - 1):
        Ht = hamiltonian.matrix[i]
        
        current_state_tensor = torch.matmul(torch.linalg.matrix_exp(-1j * Ht * Dt), current_state.state_tensor)
        
        current_state = QuantumState(current_state.dims, current_state_tensor)
        states.append(current_state)

    # if initial state is product state
    if initial_state.is_product:
        product_dims = initial_state.product_dims

        for state in states:
            state.is_product = True
            state.product_dims = product_dims

    return states

def TDSE_numeric(initial_state: QuantumState, hamiltonian: Operator, time: np.ndarray, Dt: float) -> list:
    r"""
    TDSE_numeric(initial_state: QuantumState, hamiltonian: Operator, time: np.array, Dt: float) -> list
    
    Returns a list of states as the qunatum system evolves in time
    
    The shapes of the :attr:`initial_state` and the :attr:`hamiltonian` tensor need
    to match.
    
    Args:
        initial_state (QuantumState): the initial quantum state.
        hamiltonian  (Operator): hamiltonian matrix or matrices
        time (np.ndarray): time in discrete time steps
        Dt (float): time step duration
    """

    time_tensor = torch.from_numpy(time).reshape(len(time), 1)
    time_tensor.requires_grad_(True)

    def tdse_problem(t, psi: QuantumState):
        step = int(t / Dt)
        Ht = hamiltonian.matrix[step]

        dpsi_dt = -1j * Ht.matmul(psi.state_tensor)

        return QuantumState(dims=psi.dims, vector=dpsi_dt)
    
    _, state_vectors = runge_kutta45(tdse_problem, 0, initial_state, time, Dt)

    # if initial state is product state
    if initial_state.is_product:
        product_dims = initial_state.product_dims

        for state in state_vectors:
            state.is_product = True
            state.product_dims = product_dims

    return state_vectors

def von_neumann(initial_state: Operator, hamiltonian: Operator, time: np.ndarray, Dt: float):
    r"""
    Simulates the dynamnics with von-Neumann equation

    Parameters
    ----------
    initial_state : Operator
        density matrix
    hamiltonian: Operator
        hamiltonian operator
    time: np.ndarray
        array of times as an ndarray
    Dt: float
        time step duration

    Returns
    -------
    states : list[Operator]
        list of density matrices at each time slot
    """

    time_tensor = torch.from_numpy(time).reshape(len(time), 1)
    time_tensor.requires_grad_(True)

    current_density_matrix = initial_state
    states = [current_density_matrix]

    for i in range(len(time_tensor)):
        Ht = hamiltonian.matrix[i]

        rho_t = torch.matmul(torch.matmul(torch.linalg.matrix_exp(-1j * Ht * Dt), current_density_matrix.matrix), torch.linalg.matrix_exp(1j * Ht * Dt))
        
        new_state = Operator(current_density_matrix.dims, rho_t)

        states.append(new_state)
        current_density_matrix = new_state

    return states

def lindblad_equation(initial_state: Operator, hamiltonian: Operator, time: np.ndarray, Dt: float, jump_operators: list = [], damp_rates = [], method = 'rk4'):
    r"""
    Simulates the dynamnics with lindblad master equation with optinal jump operators

    Parameters
    ----------
    initial_state : Operator
        density matrix
    hamiltonian: Operator
        hamiltonian operator
    time: np.ndarray
        array of times as an ndarray
    Dt: float
        time step duration
    jump_operators: list[Operator]
        list of jump operators
    damp_rates: list[float]
        list of jump rates

    Returns
    -------
    (time_tensor, states) : Tuple[torch.tensor, list[Operator]]
        tuple of time tensor and list of density matrices
    """

    if len(jump_operators) != len(damp_rates):
        raise Exception("jump_operators and damp_rates must have the same length!")

    time_tensor = torch.from_numpy(time).reshape(len(time), 1)
    time_tensor.requires_grad_(True)

    def ode_problem(t, rho: Operator):
        step = int(t / Dt)
        Ht = hamiltonian.matrix[step]

        drho_dt = -1j * Lie_bracket_matrices(Ht, rho.matrix)

        if len(jump_operators) > 0:
            for oper, gamma in zip(jump_operators, damp_rates):
                gamma_val = None

                if isinstance(gamma, FunctionType):
                    gamma_val = gamma(t)
                else:
                    gamma_val = gamma

                super_op_acted = SuperOperatorActed(gamma=gamma_val, jump_operator=oper, rho=rho)
                drho_dt += super_op_acted.matrix

        return Operator(dims=rho.dims, matrix=drho_dt)

    if method == 'rk4':
        time_tensor, density_matrices = runge_kutta4(ode_problem, 0, initial_state, time, Dt)
    elif method == 'rk45':
        time_tensor, density_matrices = runge_kutta45(ode_problem, 0, initial_state, time, Dt)
    elif method == "adams2":
        time_tensor, density_matrices = adams_bashforth_2step(ode_problem, 0, initial_state, time, Dt)
    elif method == "adams4":
        time_tensor, density_matrices = adams_bashforth_4step(ode_problem, 0, initial_state, time, Dt)
    elif method == "adams12":
        time_tensor, density_matrices = adams_bashforth_12step(ode_problem, 0, initial_state, time, Dt)
    else:
        raise RuntimeError(f"given numerical method: {method} not available")

    # if initial state is product state
    if initial_state.is_product:
        product_dims = initial_state.product_dims

        for state in density_matrices:
            state.is_product = True
            state.product_dims = product_dims

    return (time_tensor, density_matrices)

def lindblad_equation_FLS(initial_state: Operator, hamiltonian: Operator, time: np.ndarray, Dt: float, jump_operators: list = [], damp_rates = []):
    r"""
    Simulates the dynamnics with lindblad master equation with optinal jump operators in the Fock-Liouville space
    It maps state in the Fock-Liouiville Space and then solve numerically the equation d|rho>> / dt = L |rho>>

    Parameters
    ----------
    initial_state : Operator
        density matrix
    hamiltonian: Operator
        hamiltonian operator
    time: np.ndarray
        array of times as an ndarray
    Dt: float
        time step duration
    jump_operators: list[Operator]
        list of jump operators
    damp_rates: list[float]
        list of jump rates

    Returns
    -------
    (time_tensor, states) : Tuple[torch.tensor, list[Operator]]
        tuple of time tensor and list of density matrices
    """

    L = lindbladian_operator(hamiltonian, time, jump_operators, damp_rates)
    
    rho_FLS = fock_liouville(initial_state)

    time_tensor = torch.from_numpy(time).reshape(len(time), 1)
    time_tensor.requires_grad_(True)

    current_state = rho_FLS
    states = [current_state]

    for i in range(len(time_tensor) - 1):
        Lt = L.matrix[i]
        
        current_state_tensor = torch.matmul(torch.linalg.matrix_exp(Lt * Dt), current_state.state_tensor)
        
        current_state = QuantumState(current_state.dims, current_state_tensor)
        states.append(current_state)

    density_matrices = [Operator(hamiltonian.dims, state.state_tensor.reshape(hamiltonian.dims, hamiltonian.dims)) for state in states]

    if initial_state.is_product:
        product_dims = initial_state.product_dims

        for state in density_matrices:
            state.is_product = True
            state.product_dims = product_dims
    
    return (time_tensor, density_matrices)

def QMCT_NOT_TESTED(initial_state: QuantumState, hamiltonian: DynamicOperator, time: np.ndarray, Dt: float, jump_operators: list, damp_rates: list, nb_tranj: int) -> list:
    r"""
    QMCT(initial_state: QuantumState, hamiltonian: Operator, time: np.array, Dt: float, jump_operators: list = [], damp_rates = []) -> list
    
    Returns a list of states as the qunatum system evolves in time
    
    The shapes of the :attr:`initial_state` and the :attr:`hamiltonian` tensor need
    to match.
    
    Args:
        initial_state (QuantumState): the initial quantum state.
        hamiltonian  (Operator): hamiltonian matrix or matrices
        time (np.ndarray): time in discrete time steps
        Dt (float): time step duration
        jump_operators: list[Operator]
            list of jump operators
        damp_rates: list[float]
            list of jump rates
    """

    time_tensor = torch.from_numpy(time).reshape(len(time), 1)
    time_tensor.requires_grad_(True)

    tranjectories = []

    # while traj_index < nb_tranj:
    for _ in range(nb_tranj):
        current_state = initial_state
        states = [initial_state]

        # generate random number r
        norm_lb = torch.rand(1)

        for i in range(len(time_tensor) - 1):
            Ht = hamiltonian.matrix[i]

            # construct effective hamiltonian
            Heff = Ht
            for rate, jump_op in zip(damp_rates, jump_operators):
                Heff += (- 1j / 2) * rate * (jump_op.dagger() * jump_op).matrix
            
            current_state_tensor = torch.matmul(torch.linalg.matrix_exp(-1j * Heff * Dt), current_state.state_tensor)

            current_state = QuantumState(current_state.dims, current_state_tensor)

            # calculate norm
            norm_squared = current_state.norm() ** 2
            
            # when the norm drops the lb, then perform a quantum jump
            if norm_squared <= norm_lb:
                # print("Quantum jump")
                # randomly sample one jump operator from list of jump ops based on certain weights
                weights = np.array([torch.abs(rate * current_state.inner_product((op.dagger() * op).mul(current_state))) for op, rate in zip(jump_ops, damp_rates)])
                weights_exp = np.exp(weights)
                weights_distr = weights_exp / weights_exp.sum()

                random_jump_op = np.random.choice(jump_operators, p=weights_distr)

                # apply random jump op to the state and then normalize it
                current_state = random_jump_op.mul(current_state)

                if current_state.norm() > 0:
                    current_state = current_state.normalize()

                # print("Norm after jump: ", current_state.norm())

                # generate new random norm lower bound
                norm_lb = torch.rand(1)
            
            states.append(current_state)

        # if initial state is product state
        if initial_state.is_product:
            product_dims = initial_state.product_dims

            for state in states:
                state.is_product = True
                state.product_dims = product_dims

        tranjectories.append(states)

    return tranjectories