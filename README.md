# TorchQC: Quantum Dynamics and Machine Learning

## Description
TorchQC is a Python library that is based on the PyTorch deep learning library. Quantum operations and dynamics are entirely implemented using the PyTorch tensor mechanism. This makes all generated data ready for use in Deep Learning models.

## Contributors
[Dimitris Koutromanos](https://link-url-here.org) [Dionisis Stefanatos](https://github.com/diostef) [Emmanuel Paspalakis](https://github.com/paspalak)


## Repository structure

- examples: a folder containing jupyter notebooks demonstrating how the library can be used
    - composite_system_dynamics.ipynb
    - correlation_functions.ipynb
    - coupled_qubits_plus_resonator.ipynb
    - coupled_qubits_resonator_BBNN_extra_costs.ipynb
    - coupled_qubits_resonator_BBNN.ipynb
    - coupled_spins_3lvl_system.ipynb
    - demonstrations.ipynb
    - Dynamical_blockade_coupled_bosonic_modes.ipynb
    - Dynamical_blockade_single_mode_bosonic_system.ipynb
    - Dynamical_blockade_single_mode_gaussian_pulses.ipynb
    - emission_spectum.ipynb
    - Jaynes_Cummings_model_cat_state.ipynb
    - Jaynes_Cummings_model_coherent_state.ipynb
    - Jaynes_Cummings_model.ipynb
    - Jaynes_Cummings_model_photon_number_parity_oscillations.ipynb
    - lambda_3lvl_system_BBNN.ipynb
    - lambda_3lvl_system_STIRAP.ipynb
    - Lossy_Jaynes_Cummings_model_coherent_state.ipynb
    - Lossy_Jaynes_Cummings_model.ipynb
    - optomechanical_system.ipynb
    - Quantum_Rabi_model.ipynb
    - qubit_rabi_oscillations.ipynb
    - single_bosonic_mode.ipynb
    - single_qubit_dynamics.ipynb
    - single_qubit_markovian_master_equation_fock_liouville.ipynb
    - single_qubit_markovian_master_equation_num_methods.ipynb
    - steadystate.ipynb
    - Two_operator_two_time_correlation_function.ipynb

- LICENSE: license file
- README.md: Readme Markdown file
- requirements.txt: file with Python package requirements for running the library
- torchqc: main library directory with the core source code
    - adams_bashforth_methods.py: Adams numerical methods for solving ordinary differential equations
    - common_functions.py: contains common useful functions
    - common_matrices.py: matrix representations of common quantum operators
    - correlation.py: methods to calculate correlation functions
    - dynamics.py: methods to solve quantum dynamics (i.e. Schrodinger equation, Lindblad master equation)
    - fock_liouville.py: methods to calculate operators useful for quantum dynamics in Fock-Liouville space
    - mappings.py: Lie algebra operations (commutator, anticommutator)
    - markovian.py: method to apply Lindblad dissipative operator
    - operators.py: Operator / DynamicOperator Python classes that represent arbitrary quantum operators (density matrix, Hamiltonian)
    - runge_kutta_methods.py: Runge-Kutta numerical methods for solving ordinary differential equations
    - states.py: QuamtumState class representing quantum state vector
    - tensor_product.py: methods that calculate tensor products (Kronecker products) of states or operators

## Installation instructions
* Clone GitHub repository
```sh
git clone https://github.com/qoptics-qtech/torchqc.git
```

* Setup Python virtual environment (Python 3 required)
```sh
python -m venv my_env
```

* Activate venv 
```sh
source my_env/bin/activate
```

* Install required packages
```sh
pip install -r requirements.txt
```

* Install package (until it is officially released in Python Package Index PyPI)
```sh
pip install .
```

## Examples execution
The examples folder contains several examples that can be added into three main categories
* Closed and Open Quantum system simulations 
* Quantum optimal control with machine learning
* Quantum optics

All examples are implemented in jupyter notebooks and can be executed by any program that can execute a jupyter notebook, such as Visual Code Studio

## Usage instructions
To use the library, after it is successfully installed, it is required to be added to the program scope using commands such as:

```python
import torchqc
```

Or directly including the submodules of interest:

```python
from torchqc.states import QuantumState
```

After the needed object is in scope, one can use it to perform quantum operations. The following code defines the ground state of a qubit system:

```python
qubit_state = QuantumState.basis(2)[0]
```