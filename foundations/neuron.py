import numpy as np
from numpy.typing import NDArray

class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # Step 1: Compute pre-activation (weighted sum + bias)
        z = np.dot(x, w) + b

        # Step 2: Apply activation function
        if activation == "sigmoid":
            output = 1 / (1 + np.exp(-z))
        elif activation == "relu":
            output = max(0.0, z)
        else:
            raise ValueError("Unsupported activation. Use 'sigmoid' or 'relu'.")

        # Step 3: Round to 5 decimal places
        return round(float(output), 5)

