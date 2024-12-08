import torch
import numpy as np
from torchqc.states import QuantumState
from torchqc.operators import Operator
import numpy as np

def runge_kutta4(problem, t0, y0: Operator|QuantumState, time: np.ndarray, h: float):
    y = y0
    y_values = [y0]
    time_tensor = torch.from_numpy(time).reshape(len(time), 1)

    for i in range(len(time_tensor) - 1):
        k1 = h * problem(t0, y)
        k2 = h * problem(t0 + 0.5 * h, y + 0.5 * k1)
        k3 = h * problem(t0 + 0.5 * h, y + 0.5 * k2)
        k4 = h * problem(t0 + h, y + k3)
 
        # Update next value of y
        y = y + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        y_values.append(y)
 
        # Update next value of x
        t0 = t0 + h

    return (time_tensor, y_values)

def runge_kutta45(problem, t0, y0: Operator|QuantumState, time: np.ndarray, h: float):
    y = y0
    y_values = [y0]
    time_tensor = torch.from_numpy(time).reshape(len(time), 1)

    for i in range(len(time_tensor) - 1):
        k1 = h * problem(t0, y)
        k2 = h * problem(t0 + (1 / 4) * h, y + (1 / 4) * k1)
        k3 = h * problem(t0 + (3 / 8) * h, y + (3 / 32) * k1 + (9 / 32) * k2)
        k4 = h * problem(t0 + (12 / 13) * h, y + (1932 / 2197) * k1 + (-7200 / 2197) * k2 + (7296 / 2197) * k3)
        k5 = h * problem(t0 + h, y + (439/216) * k1 + (-8) * k2 + (3680/513) * k3 + (-845/4104) * k4)
        k6 = h * problem(t0 + (1 / 2) * h, y + (-8/27) * k1 + (2) * k2 + (-3544/2565) * k3 + (1859/4104) * k4 + (-11/40) * k5)
 
        # Update next value of y
        y = y + (16 / 135) * k1 + 0 * k2 + (6656 / 12825) * k3 + (28561 / 56430) * k4 + (- 9 / 50) * k5 + (2 / 55) * k6

        y_values.append(y)
 
        # Update next value of x
        t0 = t0 + h

    return (time_tensor, y_values)