from dataclasses import dataclass


@dataclass
class WeibullFitParams():
    lambda_: float  # In the lit Lambda
    k: float  # In the lit k
