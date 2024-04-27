"""
This script virtualizes the Photocatalytic Hydrogen Production from the
experimental data provided by the authors of Burger et al. 2020.
"""

import os
import pickle

import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                              WhiteKernel)


def get_exp_data(path: str):
    """
    Get the experimental data from 2020 Nature and drop all unuseful
    information.
    :param data_dir: data storage direction.
    :return:
    """
    data = pd.read_csv(path)
    data.drop(
        columns=[data.columns[0]],
        inplace=True,
    )   # Drop Index column
    return data


dim = 10
random_state = 1
length_scale = [0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.25, 0.25, 0.25, 0.25]
kernel = Matern(length_scale=length_scale,
                length_scale_bounds=(1e-01, 1e4),
                nu=2.5)
kernel *= ConstantKernel(1.0, (0.5, 5))
kernel += WhiteKernel(
    noise_level=0.1,
    noise_level_bounds=(5e-02, 7e-1)
)
model = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,
    normalize_y=False,
    n_restarts_optimizer=10 * dim,
    random_state=random_state,
)

# Get X and y data
suffix = "sc"
exp_data = get_exp_data(path=os.path.join("data", f"her_data_{suffix}.csv"))
X_train = exp_data.iloc[:, :dim]
y_train = exp_data.iloc[:, -1]
model.fit(X=X_train, y=y_train)

# open a file, where you ant to store the data
model_path = os.path.join(
    "virtual_models",
    f"virtual_model_{suffix}.pkl",
)
file = open(model_path, "wb")

# dump information to that file
pickle.dump(model, file)

# close the file
file.close()
