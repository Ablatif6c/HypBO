import inspect

import numpy as np

import resources.functions.standard_test_functions as fcts
from Hypothesis import Hypothesis
from resources.functions.her_function import HER
from resources.functions.test_problem import TestProblem


def get_poor_hypothesis(
    lb: np.ndarray,
    ub: np.ndarray,
    size: np.ndarray,
    sol: np.ndarray,
):
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

    conj = Hypothesis(
        name="Poor",
        lb=hyp_lb,
        ub=hyp_ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )
    print(f"\tsolution bounds = {conj.sol_bounds}")
    return conj


def get_weak_hypothesis(
    lb: np.ndarray,
    ub: np.ndarray,
    size: np.ndarray,
    sol: np.ndarray,
):
    print("Getting a weak hypothesis...")

    assert lb.shape == ub.shape, "The lower and upper bounds must \
        be of same shape."
    assert lb.shape == size.shape, "The size and the bounds must \
        be of the same shape."
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

    conj = Hypothesis(
        name="Weak",
        lb=hyp_lb,
        ub=hyp_ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )
    print(f"\t solution bounds = {conj.sol_bounds}")
    return conj


def get_good_hypothesis(
    lb: np.ndarray,
    ub: np.ndarray,
    size: np.ndarray,
    sol: np.ndarray,
):
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

    conj = Hypothesis(
        name="Good",
        lb=hyp_lb,
        ub=hyp_ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )
    print(f"\t solution bounds = {conj.sol_bounds}")
    return conj


def get_function(
    name: str,
    dim: int = 4,
        ) -> TestProblem:
    """Returns the functions from 'standard_test_problems.py' \
        which takes a dimension parameter.

    Args:
        dim (int, optional): dimension of the function. Defaults to 4.

    Returns:
        list: list of the functions.
    """
    if name == "HER":
        func = test_problem = TestProblem(
                                    problem=HER(),
                                    maximise=True,
                                )
        return func

    members = inspect.getmembers(fcts)
    for f_name, f in members:
        f_args = inspect.signature(f.__init__).parameters
        if inspect.isclass(f) and f_name != "TestProblem":
            if f_name == name:
                test_problem = None
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


def __get_HER_scenarios():
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
        [0, 1,  0,  0, 1,  0,   0,  0,  0,  1, ],    # Cys + NaOH + NaDS <= 4.5 mL

        [0, 0,  0,  -1, -1,  0,   0,  0,  0,  -1, ],  # NaCl + NaOH + NaDS >= 1 mL
        [0, 0,  0,  1, 1,  0,   0,  0,  0,  1, ],    # NaCl + NaOH + NaDS <= 2.75mL

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
    name = "Dye-sceptic"
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
    name = "Dye-advocate"
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
    name = "AR87-advocate"
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
    name = "Surfactant-sceptic"
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
    name = "Scavenger-Obsessive"
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
    name = "pH-Obsessive"
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
    name = "H-bonding Obsessive"
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
        [0, 0,  0, -1, -1,  0,   0,  0,  0,  -1, ],   # NaOH + NaDS + NaCl > 3.5 mL
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


def __get_branin_scenarios():
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
        ):
    if func_name == "HER":
        scenarios_list = __get_HER_scenarios()
        return scenarios_list

    if func_name == "Branin":
        scenarios_list = __get_branin_scenarios()
        return scenarios_list

    func = get_function(
            name=func_name,
            dim=dim,
            )
    size = 2
    if func_name == "StyblinskiTang":
        size = 0.4
    size = np.full(func.problem.dim, size, dtype=float)
    sol = func.problem.xopt

    hyp_poor = get_poor_hypothesis(
        lb=func.problem.lb,
        ub=func.problem.ub,
        size=size,
        sol=sol,
    )
    Poor = [hyp_poor]

    hyp_bad = get_weak_hypothesis(
        lb=func.problem.lb,
        ub=func.problem.ub,
        size=size,
        sol=sol,
    )
    bad = [hyp_bad]

    hyp_good = get_good_hypothesis(
        lb=func.problem.lb,
        ub=func.problem.ub,
        size=size,
        sol=sol,
        )
    good = [hyp_good]

    scenarios = [
        Poor,
        bad,
        good,
        Poor + good,
        Poor + bad,
        bad + good,
        ]
    return scenarios


def get_scenarios_name(scenarios):
    if scenarios == []:
        return "No hypothesis"
    elif len(scenarios) > 3:
        return "Mixed Starting Hypotheses"
    name = '_'.join([s.name for s in scenarios])
    return name
