# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import numpy as np

from hypothesis import Hypothesis


def from_unit_cube(
    samples: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> np.ndarray:
    """Fits a sampled point from a unit cub within a lower and an upper bounds.

    Args:
        point (np.ndarray): sampled point from a unit cube.
        lb (np.ndarray): lower bounds.
        ub (np.ndarray): upper bounds.

    Returns:
        np.ndarray: _description_
    """
    assert np.all(
        lb <= ub
    ), "The lower bounds must be less than the upper \
        bounds."
    assert lb.ndim == 1
    assert ub.ndim == 1
    assert samples.ndim == 2
    new_samples = samples * (ub - lb) + lb
    return new_samples


def latin_hypercube(
    n: int,
    dim: int,
    seed: int = 0,
) -> np.ndarray:
    """Generate latin hyper cube sampling with random noise.

    Args:
        n (int): number of samples to generate.
        dims (int): dimension of the sample.
        seed (int, optional): seed for the random generator. Defaults to 0.

    Returns:
        _type_: _description_
    """
    points = np.zeros((n, dim))
    centers = 1.0 + 2.0 * np.arange(0.0, n)
    centers = centers / float(2 * n)
    np.random.seed(seed)
    for i in range(0, dim):
        points[:, i] = centers[np.random.permutation(n)]

    perturbation = np.random.uniform(-1.0, 1.0, (n, dim))
    perturbation = perturbation / float(2 * n)
    points += perturbation
    return points


def sample_region(
    count: int,
    lb: np.ndarray,
    ub: np.ndarray,
    avoid: Optional[Hypothesis] = None,
    seed: int = 0,
) -> np.ndarray:
    samples = latin_hypercube(
        n=count,
        dim=len(lb),
        seed=seed,
    )
    samples = from_unit_cube(
        samples,
        lb,
        ub,
    )

    if avoid:
        new_samples = []
        for s in samples:
            if not avoid.apply(input=s):
                new_samples.append(s)
        samples = np.array(new_samples)

    return samples
