import numpy as np

from dataclasses import dataclass, field


@dataclass
class History:
    weights_history: list[np.ndarray] = field(default_factory=list)
    bias_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)

    def update(self, weights: np.ndarray, bias: float, loss: float):
        self.weights_history.append(weights)
        self.bias_history.append(bias)
        self.loss_history.append(loss)
