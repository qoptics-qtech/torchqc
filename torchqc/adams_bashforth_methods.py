import torch
import numpy as np
from torchqc.states import QuantumState
from torchqc.operators import Operator
import numpy as np

def adams_bashforth_2step(problem, t0, y0: Operator|QuantumState, time: np.ndarray, h: float):
    y = y0
    y_values = [y0]
    time_tensor = torch.from_numpy(time).reshape(len(time), 1)

    y1 = y0 + h * problem(t0, y0)
    t1 = t0 + h

    y_values.append(y1)

    for i in range(len(time_tensor) - 1):

        y2 = y1 + h * ((3 / 2) * problem(t1, y1) - (1 / 2) * problem(t0, y0))

        y_values.append(y2)

        y0 = y1
        y1 = y2
 
        # Update next value of x
        t0 = t1
        t1 = t1 + h

    return (time_tensor, y_values)

def adams_bashforth_4step(problem, t0, y0: Operator|QuantumState, time: np.ndarray, h: float):
    y = y0
    y_values = [y0]
    time_tensor = torch.from_numpy(time).reshape(len(time), 1)

    y1 = y0 + h * problem(t0, y0)
    t1 = t0 + h
    y_values.append(y1)

    y2 = y1 + h * ((3 / 2) * problem(t1, y1) - (1 / 2) * problem(t0, y0))
    t2 = t1 + h
    y_values.append(y2)

    y3 = y2 + h * ((23 / 12) * problem(t2, y2) - (16 / 12) * problem(t1, y1) + (5 / 12) * problem(t0, y0))
    y_values.append(y3)
    t3 = t2 + h

    for i in range(len(time_tensor) - 3):

        y4 = y3 + h * ((55 / 24) * problem(t3, y3) - (59 / 24) * problem(t2, y2) + (37 / 24) * problem(t1, y1) - (9 / 24) * problem(t0, y0))

        y_values.append(y4)

        y0 = y1
        y1 = y2
        y2 = y3
        y3 = y4
 
        # Update next value of x
        t0 = t1
        t1 = t1 + h
        t2 = t2 + h
        t3 = t3 + h

    return (time_tensor, y_values)

def adams_4step_pc(problem, t0, y0: Operator|QuantumState, time: np.ndarray, h: float):
    y_values = [y0]
    time_tensor = torch.from_numpy(time).reshape(len(time), 1)

    P1 = y0 + h * problem(t0, y0) # predictor
    t1 = t0 + h
    y1 = P1 + h * problem(t1, P1) # corrector
    y_values.append(y1)

    P2 = y1 + h * ((3 / 2) * problem(t1, y1) - (1 / 2) * problem(t0, y0)) # predictor
    t2 = t1 + h
    y2 = y1 + (1 / 2) * h * (problem(t2, P2) + problem(t1, y1)) # corrector
    y_values.append(y2)

    P3 = y2 + h * ((23 / 12) * problem(t2, y2) - (16 / 12) * problem(t1, y1) + (5 / 12) * problem(t0, y0))
    t3 = t2 + h
    y3 = y2 + h * ((5 / 12) * problem(t3, P3) + (8 / 12) * problem(t2, y2) - (1 / 12) * problem(t1, y1))
    y_values.append(y3)

    for i in range(len(time_tensor) - 3):
        P4 = y3 + h * ((55 / 24) * problem(t3, y3) - (59 / 24) * problem(t2, y2) + (37 / 24) * problem(t1, y1) - (9 / 24) * problem(t0, y0))
        t4 = t3 + h
        y4 = y3 + h * ((9 / 24) * problem(t4, P4) + (19 / 24) * problem(t3, y3) - (5 / 24) * problem(t2, y2) + (1 / 24) * problem(t1, y1))
        y_values.append(y4)

        y0 = y1
        y1 = y2
        y2 = y3
        y3 = y4
 
        # Update next value of x
        t0 += h
        t1 += h
        t2 += h
        t3 += h
        t4 += h

    return (time_tensor, y_values)

def adams_bashforth_5step(problem, t0, y0: Operator|QuantumState, time: np.ndarray, h: float):
    y_values = [y0]
    time_tensor = torch.from_numpy(time).reshape(len(time), 1)

    y1 = y0 + h * problem(t0, y0)
    t1 = t0 + h
    y_values.append(y1)

    y2 = y1 + h * ((3 / 2) * problem(t1, y1) - (1 / 2) * problem(t0, y0))
    t2 = t1 + h
    y_values.append(y2)

    y3 = y2 + h * ((23 / 12) * problem(t2, y2) - (16 / 12) * problem(t1, y1) + (5 / 12) * problem(t0, y0))
    y_values.append(y3)
    t3 = t2 + h

    y4 = y3 + h * ((55 / 24) * problem(t3, y3) - (59 / 24) * problem(t2, y2) + (37 / 24) * problem(t1, y1) - (9 / 24) * problem(t0, y0))
    y_values.append(y4)
    t4 = t3 + h

    for i in range(len(time_tensor) - 4):
        y5 = y4 + h * ((1901 / 720) * problem(t4, y4) - (2774 / 720) * problem(t3, y3) + (2616 / 720) * problem(t2, y2) - (1274 / 720) * problem(t1, y1) + (251 / 720) * problem(t0, y0))

        y_values.append(y5)

        y0 = y1
        y1 = y2
        y2 = y3
        y3 = y4
        y4 = y5
 
        # Update next value of t
        t0 = t1
        t1 += h
        t2 += h
        t3 += h
        t4 += h

    return (time_tensor, y_values)

def adams_bashforth_12step(problem, t0, y0: Operator|QuantumState, time: np.ndarray, h: float):
    y = y0
    y_values = [y0]
    time_tensor = torch.from_numpy(time).reshape(len(time), 1)

    y1 = y0 + h * problem(t0, y0)
    t1 = t0 + h
    y_values.append(y1)

    y2 = y1 + h * ((3 / 2) * problem(t1, y1) - (1 / 2) * problem(t0, y0))
    t2 = t1 + h
    y_values.append(y2)

    y3 = y2 + h * ((23 / 12) * problem(t2, y2) - (16 / 12) * problem(t1, y1) + (5 / 12) * problem(t0, y0))
    y_values.append(y3)
    t3 = t2 + h

    y4 = y3 + h * ((55 / 24) * problem(t3, y3) - (59 / 24) * problem(t2, y2) + (37 / 24) * problem(t1, y1) - (9 / 24) * problem(t0, y0))
    y_values.append(y4)
    t4 = t3 + h

    y5 = y4 + h * ((55 / 24) * problem(t4, y4) - (59 / 24) * problem(t3, y3) + (37 / 24) * problem(t2, y2) - (9 / 24) * problem(t1, y1))
    y_values.append(y5)
    t5 = t4 + h

    y6 = y5 + h * ((55 / 24) * problem(t5, y5) - (59 / 24) * problem(t4, y4) + (37 / 24) * problem(t3, y3) - (9 / 24) * problem(t2, y2))
    y_values.append(y6)
    t6 = t5 + h

    y7 = y6 + h * ((55 / 24) * problem(t6, y6) - (59 / 24) * problem(t5, y5) + (37 / 24) * problem(t4, y4) - (9 / 24) * problem(t3, y3))
    y_values.append(y7)
    t7 = t6 + h

    y8 = y7 + h * ((55 / 24) * problem(t7, y7) - (59 / 24) * problem(t6, y6) + (37 / 24) * problem(t5, y5) - (9 / 24) * problem(t4, y4))
    y_values.append(y8)
    t8 = t7 + h

    y9 = y8 + h * ((55 / 24) * problem(t8, y8) - (59 / 24) * problem(t7, y7) + (37 / 24) * problem(t6, y6) - (9 / 24) * problem(t5, y5))
    y_values.append(y9)
    t9 = t8 + h

    y10 = y9 + h * ((55 / 24) * problem(t9, y9) - (59 / 24) * problem(t8, y8) + (37 / 24) * problem(t7, y7) - (9 / 24) * problem(t6, y6))
    y_values.append(y10)
    t10 = t9 + h

    y11 = y10 + h * ((55 / 24) * problem(t10, y10) - (59 / 24) * problem(t9, y9) + (37 / 24) * problem(t8, y8) - (9 / 24) * problem(t6, y6))
    y_values.append(y11)
    t11 = t10 + h

    for i in range(len(time_tensor) - 11):

        y12  = y11 + (h / 958003200) * (4527766399 * problem(t11, y11) - 19433810163 * problem(t10, y10) + 61633227185 * problem(t9, y9))
        y12 += (h / 958003200) * (-135579356757 * problem(t8, y8) + 214139355366 * problem(t7, y7) - 247741639374 * problem(t6, y6))
        y12 += (h / 958003200) * (211103573298 * problem(t5, y5) - 131365897290 * problem(t4, y4) + 58189107627 * problem(t3, y3))
        y12 += (h / 958003200) * (-17410248271 * problem(t2, y2) + 3158642445 * problem(t1, y1) - 262747265 * problem(t0, y0))

        y_values.append(y12)

        y0 = y1
        y1 = y2
        y2 = y3
        y3 = y4
        y4 = y5
        y5 = y6
        y6 = y7
        y7 = y8
        y8 = y9
        y9 = y10
        y10 = y11
        y11 = y12
 
        # Update next value of x
        t0 += h
        t1 += h
        t2 += h
        t3 += h
        t4 += h
        t5 += h
        t6 += h
        t7 += h
        t8 += h
        t9 += h
        t10 += h
        t11 += h

    return (time_tensor, y_values)