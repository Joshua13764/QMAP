from dataclasses import dataclass

import numpy as np


@dataclass
class PowerLawFitParams():
    q: np.float32
    g: np.float32
