from dataclasses import dataclass

import numpy as np


@dataclass
class WeibullFitParams():
    scale: np.float32  # In the lit Lambda
    shape: np.float32  # In the lit k
