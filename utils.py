import inspect
import os
from multiprocessing import Pool, Process
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import resources.functions.standard_test_functions as fcts
from hypothesis import Hypothesis
from resources.functions.her_function import HER
from resources.functions.test_problem import TestProblem


def get_poor_hypothesis(
    lb: np.ndarray,
    ub: np.ndarray,
    size: np.ndarray,
    sol: np.ndarray,
) -> Hypothesis:
    """
    Get a poor hypothesis.

    Args:
        lb (np.ndarray): Lower bounds.
        ub (np.ndarray): Upper bounds.
        size (np.ndarray): Size.
        sol (np.ndarray): Solution.

    Returns:
        Hypothesis: The poor hypothesis.
    """
    print("Getting a poor hypothesis...")
    assert lb.shape == ub.shape, "The lower and upper\
          bounds must be of same shape."
    assert lb.shape == size.shape, "The size and the bounds \
        must be of the same shape."
    assert (
        lb.shape == sol.shape
    ), "The solution and the bounds must be of the same shape."

    dim = len(lb)

    coeff_equalities = None
    const_equalities = None

    coeff_inequalities = []
    for i in range(dim):
        _ub = [0] * dim
        _ub[i] = 1
        _lb = [0] * dim
        _lb[i] = -1
        coeff_inequalities.append(_ub)
        coeff_inequalities.append(_lb)
    coeff_inequalities = np.array(coeff_inequalities)

    # Getting the hypothesis bounds.
    hyp_lb = lb.copy()
    hyp_ub = lb.copy()
    for i in range(dim):
        diffs = [
            sol[i] - lb[i],
            ub[i] - sol[i],
        ]
        max_size = max(diffs)
        assert (
            max_size > size[i]
        ), f"The hypothesis size at index {i} is too big that it \
            contains the solution."

        side = np.argmax(diffs)
        # hypothesis must be placed between the lower bound and the solution.
        if side == 0:
            hyp_lb[i] = lb[i]
            hyp_ub[i] = hyp_lb[i] + size[i]
        # hypothesis must be placed between the solution and the upper bound.
        else:
            hyp_ub[i] = ub[i]
            hyp_lb[i] = hyp_ub[i] - size[i]

    const_inequalities = []
    for j in range(dim):
        const_inequalities.append(hyp_ub[j])
        const_inequalities.append(-hyp_lb[j])
    const_inequalities = np.array(const_inequalities)

    hypothesis = Hypothesis(
        name="Poor",
        lb=hyp_lb,
        ub=hyp_ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )
    print(f"\tsolution bounds = {hypothesis.sol_bounds}")
    return hypothesis


def get_weak_hypothesis(
    lb: np.ndarray,
    ub: np.ndarray,
    size: np.ndarray,
    sol: np.ndarray,
) -> Hypothesis:
    """
    Get a weak hypothesis based on the given lower bounds, upper bounds,
    size, and solution.

    Args:
        lb (np.ndarray): The lower bounds.
        ub (np.ndarray): The upper bounds.
        size (np.ndarray): The size of the hypothesis.
        sol (np.ndarray): The solution.

    Returns:
        Hypothesis: The weak hypothesis.

    Raises:
        AssertionError: If the lower bounds, upper bounds, size, and solution
            are not of the same shape, or if the hypothesis size is too big
            that it contains the solution.
    """
    print("Getting a weak hypothesis...")

    assert lb.shape == ub.shape, "The lower and upper bounds must be of same shape."
    assert lb.shape == size.shape, "The size and the bounds must be of the same shape."
    assert lb.shape == sol.shape, "The solution and the bounds must be of the same shape."

    dim = len(lb)
    coeff_equalities = None
    const_equalities = None

    coeff_inequalities = []
    for i in range(dim):
        _ub = [0] * dim
        _ub[i] = 1
        _lb = [0] * dim
        _lb[i] = -1
        coeff_inequalities.append(_ub)
        coeff_inequalities.append(_lb)
    coeff_inequalities = np.array(coeff_inequalities)

    # Getting the hypothesis bounds.
    hyp_lb = lb.copy()
    hyp_ub = lb.copy()
    for i in range(dim):
        diffs = [
            sol[i] - lb[i],
            ub[i] - sol[i],
        ]
        max_size = max(diffs)
        assert max_size > size[i], f"The hypothesis size at index {i} is too big that it contains the solution."

        side = np.argmax(diffs)
        # hypothesis must be placed between the lower bound and the solution.
        if side == 0:
            hyp_ub[i] = sol[i] - 0.2 * max_size
            hyp_lb[i] = hyp_ub[i] - size[i]
        # hypothesis must be placed between the solution and the upper bound.
        else:
            hyp_lb[i] = lb[i] + 0.2 * max_size
            hyp_ub[i] = hyp_lb[i] + size[i]

    const_inequalities = []
    for j in range(dim):
        const_inequalities.append(hyp_ub[j])
        const_inequalities.append(-hyp_lb[j])
    const_inequalities = np.array(const_inequalities)

    hypothesis = Hypothesis(
        name="Weak",
        lb=hyp_lb,
        ub=hyp_ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )
    print(f"\t solution bounds = {hypothesis.sol_bounds}")
    return hypothesis


def get_good_hypothesis(
    lb: np.ndarray,
    ub: np.ndarray,
    size: np.ndarray,
    sol: np.ndarray,
) -> Hypothesis:
    """
    Generate a good hypothesis based on the given lower bounds, upper bounds,
    size, and solution.

    Args:
        lb (np.ndarray): The lower bounds of the problem.
        ub (np.ndarray): The upper bounds of the problem.
        size (np.ndarray): The size of the problem.
        sol (np.ndarray): The solution of the problem.

    Returns:
        Hypothesis: The generated hypothesis.

    Raises:
        AssertionError: If the lower bounds, upper bounds, size, and solution
            arrays are not of the same shape, or if the size is not smaller
            than the problem bounds.
    """
    print("Getting a good hypothesis...")

    assert lb.shape == ub.shape, "The lower and upper bounds must be \
        of same shape."
    assert lb.shape == size.shape, "The size and the bounds must be \
        of the same shape."
    assert (
        lb.shape == sol.shape
    ), "The solution and the bounds must be of the same shape."
    assert np.all(
        ub - lb - size > 0
    ), "The size must be smaller than the problem bounds."
    dim = len(lb)

    coeff_equalities = None
    const_equalities = None

    coeff_inequalities = []
    for i in range(dim):
        _ub = [0] * dim
        _ub[i] = 1
        _lb = [0] * dim
        _lb[i] = -1
        coeff_inequalities.append(_ub)
        coeff_inequalities.append(_lb)
    coeff_inequalities = np.array(coeff_inequalities)

    # Getting the hypothesis bounds.
    hyp_lb = lb.copy()
    hyp_ub = lb.copy()
    for i in range(dim):
        hyp_lb[i] = sol[i] - size[i] / 2
        hyp_ub[i] = hyp_lb[i] + size[i]

    const_inequalities = []
    for j in range(dim):
        const_inequalities.append(hyp_ub[j])
        const_inequalities.append(-hyp_lb[j])
    const_inequalities = np.array(const_inequalities)

    hypothesis = Hypothesis(
        name="Good",
        lb=hyp_lb,
        ub=hyp_ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )
    print(f"\t solution bounds = {hypothesis.sol_bounds}")
    return hypothesis


def get_function(
    name: str,
    dim: int = 4,
) -> TestProblem:
    """
    Returns the functions from 'standard_test_problems.py' \
    which take a dimension parameter.

    Args:
        dim (int, optional): Dimension of the function. Defaults to 4.

    Returns:
        List[Callable]: List of the functions.
    """
    # HER is a special case.
    if name == "HER":
        func = test_problem = TestProblem(
            problem=HER(),
            maximise=True,
        )
        return func

    # Get the function from the standard_test_functions.py file.
    members = inspect.getmembers(fcts)
    for f_name, f in members:
        f_args = inspect.signature(f.__init__).parameters
        if inspect.isclass(f) and f_name != "TestProblem":
            if f_name.lower() == name.lower():
                test_problem = None
                # If the function takes a dimension parameter.
                if "d" in f_args:
                    test_problem = TestProblem(
                        problem=f(d=dim),
                        maximise=False,
                    )
                else:
                    test_problem = TestProblem(
                        problem=f(),
                        maximise=False,
                    )
                return test_problem
            else:
                continue

    raise ValueError(f"Function '{name}' not found!")


def _get_HER_hypotheses() -> List[Hypothesis]:
    """
    Get HER hypotheses.

    Returns:
        List[Hypothesis]: List of HER hypotheses.
    """
    # Features of the HER problem.
    feature_names = [
        'AcidRed871_0gL',
        'L-Cysteine-50gL',
        'MethyleneB_250mgL',
        'NaCl-3M',
        'NaOH-1M',
        'P10-MIX1',
        'PVP-1wt',
        'RhodamineB1_0gL',
        'SDS-1wt',
        'Sodiumsilicate-1wt',
    ]

    dim = len(feature_names)
    lb = np.full(dim, 0, dtype=float)
    lb[5] = 1
    ub = np.full(dim, 5, dtype=float)

    # --------------------------------- Scenario 1: Perfect Hindsight
    name = "Perfect Hindsight"
    coeff_equalities = np.array(
        [
            [0, 0,  1,  0, 0, 0,   0,  0,  0,  0, ],    # MB = 0
            [1, 0,  0,  0, 0, 0,   0,  0,  0,  0, ],    # AR87 = 0
            [0, 0,  0,  0, 0, 0,   0,  1,  0,  0, ],    # RB = 0
            [0, 0,  0,  0, 0, 0,   1,  0,  0,  0, ],    # SDS = 0
            [0, 0,  0,  0, 0, 0,   0,  0,  1,  0, ],    # PVP = 0
        ]
    )
    const_equalities = np.array(
        [
            0,  # MB = 0
            0,  # RB = 0
            0,  # SDS = 0
            0,  # SDS = 0
            0,  # PVP = 0
        ]
    )

    coeff_inequalities = np.array([
        [0, 0,  0,  0, 0, -1,   0,  0,  0,  0, ],   # P10 >= 3.5 mg

        [0, -1,  0, 0, 0,  0,   0,  0,  0,  0, ],   # CYS >= 1 mL
        [0, 1,  0, 0, 0,  0,   0,  0,  0,  0, ],    # CYS <= 3 mL

        [0, 0,  0,  0, -1,  0,   0,  0,  0,  0, ],   # NaOH >= 0.5 mL
        [0, 0,  0,  0, 1,  0,   0,  0,  0,  0, ],    # NaOH <= 2 mL

        [0, 0,  0,  0, 0,  0,   0,  0,  0,  -1, ],   # NaDS >= 0 mL
        [0, 0,  0,  0, 0,  0,   0,  0,  0,  1, ],    # NaDS <= 1.5 mL

        [0, -1,  0,  0, -1,  0,   0,  0,  0,  -1, ],  # Cys + NaOH + NaDS >= 2 mL
        # Cys + NaOH + NaDS <= 4.5 mL
        [0, 1,  0,  0, 1,  0,   0,  0,  0,  1, ],

        [0, 0,  0,  -1, -1,  0,   0,  0,  0,  -1, ],  # NaCl + NaOH + NaDS >= 1 mL
        # NaCl + NaOH + NaDS <= 2.75mL
        [0, 0,  0,  1, 1,  0,   0,  0,  0,  1, ],

        [0, 0,  0, 1, 0,  0,   0,  0,  0,  0, ],    # CYS <= 2.5 mL
    ])
    const_inequalities = np.array([
        -3.5,   # P10 >= 3.5 mg

        -1,     # CYS >= 1 mL
        3.5,    # CYS <= 3 mL

        -0.5,   # NaOH >= 0.5 mL
        2,      # NaOH <= 2 mL

        0,      # NaDS >= 0 mL
        1.5,    # NaDS <= 1.5 mL

        -2,     # Cys + NaOH + NaDS >= 2 mL
        4.5,    # Cys + NaOH + NaDS <= 4.5 mL

        -1,     # NaCl + NaOH + NaDS >= 1 mL
        2.75,   # NaCl + NaOH + NaDS <= 2.75 mL

        2.5,    # NaCL <= 2.5 mL
    ])

    perfect_hindsight = Hypothesis(
        name=name,
        feature_names=feature_names,
        lb=lb,
        ub=ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )

    # -------------------------------------- Scenario 2: What they knew
    name = "What they knew"
    coeff_equalities = np.array(
        [
            [0, 0,  0,  0, 0, 1,   0,  0,  0,  0, ],    # P10 = 5
        ]
    )
    const_equalities = np.array(
        [
            5,  # P10 = 5
        ]
    )

    coeff_inequalities = np.array([
        [0, -1,  0, 0, 0,  0,   0,  0,  0,  0, ],   # CYS >= 1 mL
        [0, 1,  0, 0, 0,  0,   0,  0,  0,  0, ],    # CYS <= 4 mL

        [0, 0,  1, 0, 0,  0,   0,  0,  0,  0, ],    # MB <= 0.5 mL

        [1, 0,  0, 0, 0,  0,   0,  0,  0,  0, ],    # AR87 <= 1 mL

        [0, 0,  0, 0, 0,  0,   0,  1,  0,  0, ],    # RB <= 0.5 mL

        [0, 0,  0, 0, 1,  0,   0,  0,  0,  0, ],    # NaOH <= 3 mL

        [0, 0,  0, 1, 0,  0,   0,  0,  0,  0, ],    # NaCl <= 3 mL

        [0, 0,  0, 0, 0,  0,   0,  0,  1,  0, ],    # SDS <= 1 mL

        [0, 0,  0, 0, 0,  0,   1,  0,  0,  0, ],    # PVP <= 2 mL

        [0, 0,  0, 0, 0,  0,   0,  0,  0,  1, ],    # NaDS <= 4 mL
    ])
    const_inequalities = np.array([
        -1,     # CYS >= 1 mL

        4,      # CYS <= 4 mL

        0.5,    # MB <= 0.5 mL

        1,      # AR87 <= 1 mL

        0.5,    # RB <= 0.5 mL

        3,      # NaOH <= 3 mL

        3,      # NaCl <= 3 mL

        1,      # SDS <= 1 mL

        2,      # PVP <= 2 mL

        4,      # NaDS <= 4 mL
    ])

    what_we_kew = Hypothesis(
        name=name,
        feature_names=feature_names,
        lb=lb,
        ub=ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )

    # ---------------------------------------- Scenario 3: Dye-sceptic
    name = "Dye Sceptic"
    coeff_equalities = np.array(
        [
            [0, 0,  1,  0, 0, 0,   0,  0,  0,  0, ],    # MB = 0

            [1, 0,  0,  0, 0, 0,   0,  0,  0,  0, ],    # AR87 = 0

            [0, 0,  0,  0, 0, 0,   0,  1,  0,  0, ],    # RB = 0
        ]
    )
    const_equalities = np.array(
        [
            0,  # MB = 0

            0,  # AR87 = 0

            0,  # RB = 0
        ]
    )

    coeff_inequalities = None
    const_inequalities = None

    dye_sceptic = Hypothesis(
        name=name,
        feature_names=feature_names,
        lb=lb,
        ub=ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )

    # ------------------------------------- Scenario 4: Dye-advocate
    name = "Dye Fanatic"
    coeff_equalities = None
    const_equalities = None

    coeff_inequalities = np.array([
        [-1, 0,  -1, 0, 0,  0,   0,  -1,  0,  0, ],   # MB + AR87 + RB >= 3 mL
    ])
    const_inequalities = np.array([
        -3,     # MB + AR87 + RB >= 3 mL
    ])

    dye_advocate = Hypothesis(
        name=name,
        feature_names=feature_names,
        lb=lb,
        ub=ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )

    # ------------------------------------- Scenario 5: AR87-advocate
    name = "AR87 Obsessed"
    coeff_equalities = None
    const_equalities = None

    coeff_inequalities = np.array([
        [-1, 0,  0, 0, 0,  0,   0,  0,  0,  0, ],   # AR87 >= 3 mL

        [0, 0,  1, 0, 0,  0,   0,  0,  0,  0, ],   # MB <= 0.5 mL

        [0, 0,  0, 0, 0,  0,   0,  1,  0,  0, ],   # RB <= 0.5 mL
    ])
    const_inequalities = np.array([
        -3,     # AR87 >= 3 mL

        0.5,    # MB <= 0.5 mL

        0.5,    # RB <= 0.5 mL
    ])

    ar87_advocate = Hypothesis(
        name=name,
        feature_names=feature_names,
        lb=lb,
        ub=ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )

    # ----------------------------------- Scenario 6: Surfactant-sceptic
    name = "Surfactant Sceptic"
    coeff_equalities = np.array(
        [
            [0, 0,  0,  0, 0, 0,   0,  0,  1,  0, ],    # SDS = 0

            [0, 0,  0,  0, 0, 0,   1,  0,  0,  0, ],    # PVP = 0

        ]
    )
    const_equalities = np.array(
        [
            0,  # SDS = 0

            0,  # PVP = 0
        ]
    )

    coeff_inequalities = None
    const_inequalities = None

    surfactant_sceptic = Hypothesis(
        name=name,
        feature_names=feature_names,
        lb=lb,
        ub=ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )

    # ---------------------------------- Scenario 7: Scavenger-Obsessive
    name = "Scavenger Obsessive"
    coeff_equalities = None
    const_equalities = None

    coeff_inequalities = np.array([
        [0, -1,  0, 0, 0,  0,   0,  0,  0,  0, ],   # CYS >= 4 mL
    ])
    const_inequalities = np.array([
        -4,     # CYS >= 4 mL
    ])

    scavenger_obsessive = Hypothesis(
        name=name,
        feature_names=feature_names,
        lb=lb,
        ub=ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )

    # --------------------------------------- Scenario 8: pH-Obsessive
    name = "pH Fanatic"
    coeff_equalities = None
    const_equalities = None

    coeff_inequalities = np.array([
        [0, 0,  0, 0, -1,  0,   0,  0,  0,  -1, ],   # NaOH + NaDS > 3.5 mL
    ])
    const_inequalities = np.array([
        -3.5,     # NaOH + NaDS > 3.5 mL
    ])

    ph_obsessive = Hypothesis(
        name=name,
        feature_names=feature_names,
        lb=lb,
        ub=ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )

    # -------------------------------- Scenario 9: H-bonding Obsessive
    name = "H-bond Lover"
    coeff_equalities = None
    const_equalities = None

    coeff_inequalities = np.array([
        [0, 0,  0, 0, 0,  0,   0,  0,  0,  -1, ],   # NaDS > 3.5 mL
    ])
    const_inequalities = np.array([
        -3.5,     # NaDS > 3.5 mL
    ])

    h_bonding_obsessive = Hypothesis(
        name=name,
        feature_names=feature_names,
        lb=lb,
        ub=ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )

    # ------------------------------------ Scenario 10: Halophile
    name = "Halophile"
    coeff_equalities = None
    const_equalities = None

    coeff_inequalities = np.array([
        # NaOH + NaDS + NaCl > 3.5 mL
        [0, 0,  0, -1, -1,  0,   0,  0,  0,  -1, ],
    ])
    const_inequalities = np.array([
        -3.5,     # NaOH + NaDS + NaCl > 3.5 mL
    ])

    halophile = Hypothesis(
        name=name,
        feature_names=feature_names,
        lb=lb,
        ub=ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )

    # ---------------------------------------- Scenario 11: Halophobe
    name = "Halophobe"
    coeff_equalities = np.array([
        [0, 0,  0, 1, 0,  0,   0,  0,  0,  0, ],   # NaCl = 0 mL
    ])
    const_equalities = np.array([
        0,     # NaCl = 0 mL
    ])

    coeff_inequalities = None
    const_inequalities = None

    halophobe = Hypothesis(
        name=name,
        feature_names=feature_names,
        lb=lb,
        ub=ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )

    # --------------------------------- Scenario 12: Bizarro World
    name = "Bizarro World"
    coeff_equalities = np.array(
        [
            [0, 0,  0,  0, 0, 1,   0,  0,  0,  0, ],    # P10 = 1

            [0, 1,  0,  0, 0, 0,   0,  0,  0,  0, ],    # CYS = 0

            [0, 0,  0,  0, 1, 0,   0,  0,  0,  0, ],    # NaOH = 0

            [0, 0,  0,  0, 0, 0,   0,  0,  0,  1, ],    # NaDS = 0
        ]
    )
    const_equalities = np.array(
        [
            1,  # P10 = 1

            0,  # CYS = 0

            0,  # NaOH = 0

            0,  # NaDS = 0
        ]
    )

    coeff_inequalities = np.array([
        [0, 0,  -1,  0, 0, 0,   0,  0,  0,  0, ],   # MB >= 0.5 mg

        [-1, 0,  0, 0, 0,  0,   0,  0,  0,  0, ],   # AR87 >= 0.5 mL

        [0, 0,  0, 0, 0,  0,   0,  -1,  0,  0, ],    # RB <= 0.5 mL

        [0, 0,  0,  -1, 0,  0,   0,  0,  0,  0, ],   # NaCL >= 0.5 mL

        [0, 0,  0,  0, 0,  0,   0,  0,  -1,  0, ],    # SDS >= 0.5 mL

        [0, 0,  0,  0, 0,  0,   -1,  0,  0,  0, ],   # PVP >= 0.5 mL
    ])
    const_inequalities = np.array([
        -0.5,   # MB >= 0.5 mg

        -0.5,   # AR87 >= 0.5 mL

        -0.5,   # RB >= 0.5 mL

        -0.5,   # NaCl >= 0.5 mL

        -0.5,   # SDS >= 0.5 mL

        -0.5,   # PVP >= 0.5 mL
    ])

    bizarro_world = Hypothesis(
        name=name,
        feature_names=feature_names,
        lb=lb,
        ub=ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )

    # --------------------------------- Scenario 13: Crowd
    virtual_chemists_team = [
        dye_sceptic,
        dye_advocate,
        ar87_advocate,
        surfactant_sceptic,
        scavenger_obsessive,
        ph_obsessive,
        h_bonding_obsessive,
        halophile,
        halophobe,
    ]

    scenarios_list = [
        [perfect_hindsight],
        [what_we_kew],
        [bizarro_world],
        virtual_chemists_team
    ]

    return scenarios_list


def _get_branin_hypotheses() -> List[List[Hypothesis]]:
    """
    Get a list of scenarios for the Branin function.

    Returns:
        scenarios (List[List[Hypothesis]]): A list of scenarios, where each
            scenario is a list of Hypothesis objects.
    """
    func = get_function("Branin", dim=2)
    lb = func.problem.lb
    ub = func.problem.ub
    coeff_equalities = None
    const_equalities = None

    coeff_inequalities = np.array([
        [-1, 0],   # x1 >= 4
        [1,  0],   # x1 <= 6

        [0, -1],   # x2 >= 13
        [0,  1],   # x2 <= 15
    ])

    const_inequalities = np.array([
        -4,  # x1 >= 4
        6,  # x1 <= 6

        -13,
        15,
    ])

    hyp_poor = Hypothesis(
        name="Poor",
        feature_names=func.input_columns,
        lb=lb,
        ub=ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )
    poor = [hyp_poor]

    # Weak
    coeff_equalities = None
    const_equalities = None

    coeff_inequalities = np.array([
        [-1, 0],   # x1 >= 3
        [1,  0],   # x1 <= 5

        [0, -1],   # x2 >= 5
        [0,  1],   # x2 <= 7
    ])
    const_inequalities = np.array([
        -3,  # x1 >= 3
        5,  # x1 <= 5

        -5,
        7,
    ])

    hyp_bad = Hypothesis(
        name="Weak",
        feature_names=func.input_columns,
        lb=lb,
        ub=ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )
    bad = [hyp_bad]

    # Good
    coeff_equalities = None
    const_equalities = None

    coeff_inequalities = np.array([
        [-1, 0],   # x1 >=
        [1,  0],   # x1 <=

        [0, -1],   # x2 >=
        [0,  1],   # x2 <=
    ])
    const_inequalities = np.array([
        -np.pi + 0.5,  # x1 >= 3
        np.pi + 0.5,  # x1 <= 5

        -2.275 + 0.5,
        2.275 + 0.5,
    ])

    hyp_good = Hypothesis(
        name="Good",
        feature_names=func.input_columns,
        lb=lb,
        ub=ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )
    good = [hyp_good]

    scenarios = [
        poor,
        bad,
        good,
        poor + good,
        bad + good,
        poor + bad,
    ]
    return scenarios


def get_scenarios(
        func_name: str,
        dim: int = 4,
) -> List[List[Hypothesis]]:
    """
    Get the scenarios for a given function.

    Args:
        func_name (str): The name of the function.
        dim (int, optional): The dimension of the function. Defaults to 4.

    Returns:
         List[List[Hypothesis]]: A list of scenario, where each
            scenario is a list of Hypothesis objects.

    Raises:
        ValueError: If the function name is not recognized.
    """
    # HER is a special case.
    if func_name == "HER":
        scenarios = _get_HER_hypotheses()
        return scenarios

    # Branin is a special case.
    if func_name.lower() == "branin":
        scenarios = _get_branin_hypotheses()
        return scenarios

    # Compute the different hypotheses.
    func = get_function(
        name=func_name,
        dim=dim,
    )
    hypothesis_size = 2
    if func_name.lower() == "styblinskitang":
        hypothesis_size = 0.4
    hypothesis_size = np.full(func.problem.dim, hypothesis_size, dtype=float)
    sol = func.problem.xopt

    poor_hypothesis = get_poor_hypothesis(
        lb=func.problem.lb,
        ub=func.problem.ub,
        size=hypothesis_size,
        sol=sol)

    weak_hypothesis = get_weak_hypothesis(
        lb=func.problem.lb,
        ub=func.problem.ub,
        size=hypothesis_size,
        sol=sol)

    good_hypothesis = get_good_hypothesis(
        lb=func.problem.lb,
        ub=func.problem.ub,
        size=hypothesis_size,
        sol=sol)

    scenarios = [
        [poor_hypothesis],
        [weak_hypothesis],
        [good_hypothesis],
        [good_hypothesis, poor_hypothesis],
        [good_hypothesis, weak_hypothesis],
        [weak_hypothesis, poor_hypothesis],
    ]
    return scenarios


def get_scenario_name(scenario: List[List[Hypothesis]]) -> str:
    """
    Returns the name of a scenario based on the given list of hypotheses.

    Args:
        scenario (List[List[Hypothesis]]): A list of scenario, where each
            scenario is a list of Hypothesis objects.

    Returns:
        str: The name of the scenario.

    Raises:
        None
    """
    if scenario == []:
        return "No hypothesis"
    elif len(scenario) > 3:
        return "Mixed Starting Hypotheses"
    name = '_'.join([s.name for s in scenario])
    return name


def process_scenario(
    path: str,
    scenario_name: str,
    func_name: str,
    objective: float,
    regret: bool = True,
) -> Dict[int, pd.DataFrame]:
    """
    Process a scenario by reading dataset files, calculating the best so far
    values, and storing them in a dictionary.

    Args:
        path (str): The path to the scenario folder.
        scenario_name (str): The name of the scenario.
        func_name (str): The name of the function.
        objective (float): The objective value.
        regret (bool, optional): Whether to calculate regret. Defaults to True.

    Returns:
        dict: A dictionary containing the best so far values for each seed.
    """
    seeds = {}
    # Get the dataset files
    scenario_path = os.path.join(
        path,
        scenario_name,
    )

    # If the scenario does not exist, return
    if not os.path.exists(scenario_path):
        raise FileNotFoundError(
            f"Scenario {scenario_name} does not exist in {path}")

    # For each file in the scenario folder
    for file in os.listdir(scenario_path):
        # print(f"\t\t...file: {file}")
        if file.startswith(f"dataset_{func_name}_{scenario_name}"):
            # Fetch the seed integer in the filename
            file_name_wo_ext = file.split(".")[0]
            seed_str = next(filter(str.isdigit, file_name_wo_ext.split("_")))
            seed = int(seed_str)

            # Create a DataFrame to save all data from
            # all scenarios of that seed
            if seed not in seeds.keys():
                seeds[seed] = pd.DataFrame()

            # Get the Target column
            target = pd.read_csv(
                os.path.join(
                    scenario_path,
                    file,
                )
            )["Target"]
            best_so_far = pd.DataFrame(columns=[scenario_name])

            # Calculate the best so far
            for i in target.index:
                if regret:
                    # TODO: Turn HypBO into a minimization problem
                    best_so_far.loc[i] = objective - max(target[: i + 1])
                else:
                    best_so_far.loc[i] = max(target[: i + 1])

            # Save the best so far column
            seeds[seed][scenario_name] = best_so_far[scenario_name]

    return seeds


def save_all_best_so_far(
    func_name: str,
    dir_name: str = "data",
    regret: bool = False,
):
    """
    Generate and save the best so far datasets for a given function
    and scenarios.

    Args:
        func_name (str): The name of the function.
        dir_name (str, optional): Directory name to save the datasets.
            Defaults to "data".
        regret (bool, optional): Flag indicating whether to calculate regret.
            Defaults to False.
    """
    print(
        f"{dir_name}: generating the best so far datasets for {func_name}..."
    )
    seeds = {}
    path = os.path.join(dir_name, func_name)

    scenario_names = [name for name in os.listdir(
        path) if os.path.isdir(os.path.join(path, name))]
    objective = 0
    # Get the true objective value for the function if regret is True
    if regret:
        if func_name == "HER":
            objective = 28
        else:
            func = get_function(name=func_name.split("_")[0])
            objective = func.problem.yopt

    # For each scenario in the list
    with Pool(processes=len(scenario_names)) as pool:
        args = [
            (path, scenario_name, func_name, objective, regret)
            for scenario_name in scenario_names
        ]
        results = pool.starmap(process_scenario, args)
        for res in results:
            if res is None:
                continue
            for seed, series in res.items():
                if seed not in seeds.keys():
                    seeds[seed] = pd.DataFrame()
                scenario_name = series.columns[0]
                seeds[seed][scenario_name] = series

    # Save the best so far data
    for seed, df in seeds.items():
        file_path = os.path.join(path, f"all_{func_name}_bsf_{seed}.csv")
        df.to_csv(file_path)
        # print(f"Data generated and saved in {file_path}")


def save_avg_and_std_best_so_far(
    func_name: str,
    dir_name: str = "data",
):
    """
    Generate and save the average and standard deviation datasets for a given
    function.

    Args:
        func_name (str): The name of the function.
        dir_name (str, optional): Directory name to save the datasets.
            Defaults to "data".
    """
    print(f"\tGenerating the average and std datasets for {func_name}...")
    path = os.path.join(
        dir_name,
        func_name,
    )
    scenario_names = [name for name in os.listdir(
        path) if os.path.isdir(os.path.join(path, name))]
    df_avg = pd.DataFrame(columns=scenario_names)
    df_std = pd.DataFrame(columns=scenario_names)

    for scenario_name in scenario_names:
        # Get the bsf files
        df_scenario = pd.DataFrame()
        for file in os.listdir(path):
            if file.startswith(f"all_{func_name}_bsf_"):
                seed = file.split(".")[0]
                seed = seed.split("_")[-1]
                df = pd.read_csv(
                    os.path.join(
                        path,
                        file,
                    )
                )
                if scenario_name in df.columns:
                    target = df[scenario_name]
                    df_scenario[f"Target_{seed}"] = target
        df_avg[scenario_name] = df_scenario.mean(axis=1)
        df_std[scenario_name] = df_scenario.std(axis=1)

    average_file_path = os.path.join(path, f"{func_name}_average.csv")
    df_avg.to_csv(average_file_path)
    print(f"\tData generated and saved in {average_file_path}")

    std_file_path = os.path.join(path, f"{func_name}_std.csv")
    df_std.to_csv(std_file_path)
    print(f"\tData generated and saved in {std_file_path}")


def process_function_data(
    func_name: str,
    dim: int,
    method_dir: str,
    regret: bool = True,
):
    """
    Processes the data for a specific function and saves the best so far data.

    Args:
        func_name (str): Name of the function.
        dim (int): Dimensionality of the function.
        method_dir (str): Directory to save the data.
        method (str, optional): Method name. Defaults to "HypBO".
        regret (bool, optional): Whether to calculate regret or not.
            Defaults to True.

    Returns:
        None
    """
    print(f"\tProcessing {func_name}...")
    if "HER" not in func_name:
        # Append dimension to function name if not already present
        func_name = f"{func_name}_d{dim}"

    # Save the best so far data
    save_all_best_so_far(
        func_name=func_name,
        dir_name=method_dir,
        regret=regret,
    )

    # Generate and save average and std datasets
    save_avg_and_std_best_so_far(
        func_name=func_name,
        dir_name=method_dir,
    )


def multiprocess_all_function_data(
    test_functions: List[Tuple[str, int]],
    method_dir: str,
    regret: bool = True,
):
    """
    Processes the data for multiple functions in parallel and saves
    the best so far data.

    Args:
        test_functions (List[Tuple[str, int]]): A list of tuples containing
            the name of the function and its dimensionality.
        method_dir (str): The directory to save the data.
        method (str, optional): The name of the method. Defaults to "HypBO".
        regret (bool, optional): Whether to calculate regret or not.
            Defaults to True.

    Returns:
        None
    """
    processes = []  # List of processes to run in parallel.

    # For each function, generate the best so far data.
    for func_name, dim in test_functions:
        process = Process(
            target=process_function_data,
            args=(func_name, dim, method_dir, regret),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()  # Wait for all processes to finish.
