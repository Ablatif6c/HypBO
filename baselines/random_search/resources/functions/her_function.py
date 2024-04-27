import os
import pickle

import numpy as np


class HER:
    """
    This model uses an sklearn GP object from a pickle as a black box function.

    """

    def __init__(
            self,
            noisy=False,
            ):
        # Model
        path = os.path.join(
            "resources",
            "functions",
            "Burger et al. 2020",
            "virtual_models",
            "virtual_model_combined.pkl",
        )

        self.var_map = {'AcidRed871_0gL': 0,
                        'L-Cysteine-50gL': 1,
                        'MethyleneB_250mgL': 2,
                        'NaCl-3M': 3,
                        'NaOH-1M': 4,
                        'P10-MIX1': 5,
                        'PVP-1wt': 6,
                        'RhodamineB1_0gL': 7,
                        'SDS-1wt': 8,
                        'Sodiumsilicate-1wt': 9}
        self.dim = len(self.var_map.values())
        self.lb = np.full(self.dim, 0)
        self.lb[5] = 1  # P10-MIX1
        self.ub = np.full(self.dim, 5)

        with open(path, 'rb') as file:
            gp = pickle.load(file)
        self.gp = gp
        self.noisy = noisy

        # optimum value(s)
        self.xopt = np.array([0,	0.98,	0,	1.04,	0.75,	3.96,	0,	0,	0,	0.75])
        self.yopt = self.__call__(self.xopt)

    def __call__(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if self.noisy:
            return self.gp.predict(
                x.reshape(1, -1))[0] + np.random.uniform(-.5, .5)
        else:
            return self.gp.predict(x.reshape(1, -1))[0]
