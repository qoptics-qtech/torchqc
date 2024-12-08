import torch
import numpy as np
from typing import Self
from torchqc.operators import Operator
from torchqc.mappings import anticommutator

class SuperOperatorActed(Operator):
    def __init__(self, gamma, jump_operator: Operator, rho: Operator) -> None:
        super_tensor = gamma * (jump_operator.opmul(rho).opmul(jump_operator.dagger()).matrix + (- 1 / 2) * anticommutator(jump_operator.dagger().opmul(jump_operator), rho).matrix)

        super().__init__(jump_operator.dims, super_tensor)