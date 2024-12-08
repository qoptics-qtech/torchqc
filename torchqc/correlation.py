import torch
import numpy as np
from torchqc.operators import Operator, DynamicOperator
from torchqc.dynamics import lindblad_equation
from torchqc.common_functions import expect_val_dm, steadystate

def correlation_fn_2op_2time(H: Operator, initial_state: Operator, time: np.ndarray|None, tau: np.ndarray|None, Dt: float, A: Operator, B: Operator, jump_ops: list[Operator], rates: list[float]):
    r"""
    function that computes the two time correlation of two operators : \langle A(t + τ)B(t) \rangle
    
    Parameters
    ----------
    H: Operator
    initial_state: Operator
    time: np.ndarray|None
    tau: np.ndarray|None
    Dt: float
    A: Operator
    B: Operator
    jump_ops: list[Operator]
    rates: list[float]
    
    Methodology:
    ------------
    1. Compute B V(t, 0){rho(0)}
    2. Compute A V(t + τ, t) {B V(t, 0){rho(0)}}
    """

    if time is None:
        state = B.opmul(initial_state)
        _, states_for_coherence = lindblad_equation(state, H, tau, Dt, jump_ops, rates)
        correlation = np.array([torch.trace(A.opmul(state).matrix).numpy() for state in states_for_coherence])
    else:
        _, states = lindblad_equation(initial_state, H, time, Dt, jump_ops, rates)
        states = [A.opmul(state) for state in states]

        # correlation = np.zeros([np.size(time), np.size(tau)], dtype=complex)
        correlation = torch.zeros((np.size(time), np.size(tau)), dtype=torch.complex128)

        for i, rho_i in enumerate(states):
            t = time[i]
            _, states_for_coherence = lindblad_equation(rho_i, H, t + tau, Dt, jump_ops, rates)
            correlation[i, :] = torch.tensor([torch.trace(A.opmul(state).matrix) for state in states_for_coherence])

    return correlation

def correlation_fn_2op_1time(H: Operator, state: Operator|None, tau: np.ndarray|None, Dt: float, A: Operator, B: Operator, jump_ops: list[Operator], rates: list[float]):
    r"""
    function that computes the one time correlation of two operators : \langle A(t)B(t) \rangle
    
    Parameters
    ----------
    H: Operator
    initial_state: Operator
    tau: np.ndarray|None
    Dt: float
    A: Operator
    B: Operator
    jump_ops: list[Operator]
    rates: list[float]
    """

    if state is None:
        state = steadystate(H, jump_ops, rates)

    state = B.opmul(state)

    ham_tensor = H.matrix.expand(len(tau), -1, -1).type(torch.complex128)
    hamiltonian = Operator(H.dims, ham_tensor)

    _, states_for_coherence = lindblad_equation(state, hamiltonian, tau, Dt, jump_ops, rates)
    correlation = torch.tensor([torch.trace(A.opmul(state).matrix) for state in states_for_coherence])

    return correlation

def correlation_fn_3op_1time(H: DynamicOperator|Operator, state: Operator|None, tau: np.ndarray|None, Dt: float, A: Operator, B: Operator, C: Operator, jump_ops: list[Operator], rates: list[float]):
    r"""
    function that computes the one time correlation of three operators : \langle A(0)B(τ)C(0) \rangle
    
    Parameters
    ----------
    H: Operator
    initial_state: Operator
    tau: np.ndarray|None
    Dt: float
    A: Operator
    B: Operator
    C: Operator
    jump_ops: list[Operator]
    rates: list[float]
    """

    if state is None:
        state = steadystate(H, jump_ops, rates)

    if tau is not None:
        state = C.opmul(state.opmul(A))

        if isinstance(H, Operator):
            ham_tensor = H.matrix.expand(len(tau), -1, -1).type(torch.complex128)
            hamiltonian = Operator(H.dims, ham_tensor)
        else:
            hamiltonian = H

        _, states_for_coherence = lindblad_equation(state, hamiltonian, tau, Dt, jump_ops, rates)
        correlation = torch.tensor([torch.trace(B.opmul(state).matrix) for state in states_for_coherence])
    else:
        correlation = torch.real(torch.trace((A * B * C * state).matrix))

    return correlation

def fist_order_coherence_fn(H: Operator, initial_state: Operator, tau: np.ndarray, Dt: float, A: Operator, B: Operator, jump_ops: list[Operator], rates: list[float]):
    r"""
    g^{(1)}(\tau) = \frac{\langle A(\tau)B(0)\rangle}{\sqrt{\langle A(\tau)B(\tau)\rangle\langle A(0)B(0)\rangle}}

    For a coherent state |g(1)(τ)|=1, and for a completely incoherent (thermal) state g(1)(τ)=0.

    Parameters
    ----------
    H: Operator
        Hamiltonian
    initial_state: Operator
        density matrix
    tau: list[float]
        discrete time steps
    Dt: float
        time step duration
    A: operator
        first operator in the correlation function
    B: operator
        second operaor in the correlation function
    jump_ops: list[Operator]
        jump or Libland operators for the dissipation
    rates: list[float]
        dump rates for the corresponding jump_ops

    Returns:
    --------
    g1: first order coherence function values for the given timings
    """

    _, denominator_states = lindblad_equation(initial_state, H, tau, Dt, jump_ops, rates)

    G1 = correlation_fn_2op_2time(H, initial_state, None, tau, Dt, A, B, jump_ops, rates)

    AtBt = expect_val_dm(denominator_states, A.opmul(B))

    g1 = G1 / np.sqrt(AtBt * AtBt[0])

    return g1

def second_order_coherence_fn(H: Operator, state: Operator, taus: np.ndarray|None, Dt: float, A: Operator, jump_ops: list[Operator], rates: list[float]):
    r"""
    Compute the second order coherence function for the given timings tau
    g^{(2)}(\tau) = \frac{\langle A(0)A(\tau)B(\tau)B(0)\rangle}{\langle A(0)B(0) A(t)B(t)\rangle}

    Parameters
    ----------
    H: Operator
        Hamiltonian
    state: Operator
        density matrix
    taus: list[float]
        discrete time steps
    Dt: float
        time step duration
    A: operator
        first operator in the correlation function
    B: operator
        second operaor in the correlation function
    jump_ops: list[Operator]
        jump or Libland operators for the dissipation
    rates: list[float]
        dump rates for the corresponding jump_ops

    Returns:
    --------
    g2: second order coherence function values for the given timings
    """

    B = A * A.dagger()
    C = A.dagger()

    # numerator
    G2 = correlation_fn_3op_1time(H, state, taus, Dt, A, B, C, jump_ops, rates)

    # denominator
    ham_tensor = H.matrix.expand(len(taus), -1, -1).type(torch.complex128)
    hamiltonian = Operator(H.dims, ham_tensor)

    _, states = lindblad_equation(state, hamiltonian, taus, Dt, jump_ops, rates)
    AtCt = expect_val_dm(states, A * C)

    g2 = G2 / (AtCt[0] * AtCt)

    return g2

def spectrum(corr, time):
    """
    Computes the emission spectum of the given correlation values

    Parameters
    ----------
    corr: correlation values
    time: list of times in discrete steps

    Returns
    -------
    (freq, power spectrum) : tuple of frequencies and power spectrum
    """

    steps = len(time)
    Dt = time[-1] / steps

    F = torch.fft.fft(corr)
    f = torch.fft.fftfreq(steps, Dt)

    # first put negative and next positive values indices
    neg_idx = torch.where(f < 0)[0]
    pos_idx = torch.where(f >= 0)[0]
    indices = torch.cat((neg_idx, pos_idx), 0)

    return (2 * np.pi * f[indices], 2 * Dt * torch.real(F[indices]))