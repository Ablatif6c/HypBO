import inspect

import numpy as np

from resources.functions import standard_test_functions as fcts
from resources.functions.test_problem import TestProblem
from resources.Hypothesis import Hypothesis


def get_poor_hypothesis(
    lb: np.ndarray,
    ub: np.ndarray,
    size: np.ndarray,
    sol: np.ndarray
) -> Hypothesis:
    """
    Get a poor hypothesis.

    Args:
        lb (np.ndarray): Lower bounds.
        ub (np.ndarray): Upper bounds.
        size (np.ndarray): Size of the hypothesis.
        sol (np.ndarray): Solution.

    Returns:
        Hypothesis: The poor hypothesis.
    """
    print("Getting a poor hypothesis...")
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
    hypothesis_lb = lb.copy()
    hypothesis_ub = lb.copy()
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
            hypothesis_lb[i] = lb[i]
            hypothesis_ub[i] = hypothesis_lb[i] + size[i]
        # hypothesis must be placed between the solution and the upper bound.
        else:
            hypothesis_ub[i] = ub[i]
            hypothesis_lb[i] = hypothesis_ub[i] - size[i]

    const_inequalities = []
    for j in range(dim):
        const_inequalities.append(hypothesis_ub[j])
        const_inequalities.append(-hypothesis_lb[j])
    const_inequalities = np.array(const_inequalities)

    hypothesis = Hypothesis(
        name="Poor",
        lb=hypothesis_lb,
        ub=hypothesis_ub,
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
    sol: np.ndarray
) -> Hypothesis:
    """
    Get a weak hypothesis.

    Args:
        lb (np.ndarray): Lower bounds.
        ub (np.ndarray): Upper bounds.
        size (np.ndarray): Size of the hypothesis.
        sol (np.ndarray): Solution.

    Returns:
        Hypothesis: The weak hypothesis.
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
    hypothesis_lb = lb.copy()
    hypothesis_ub = lb.copy()
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
            hypothesis_ub[i] = sol[i] - 0.2 * max_size
            hypothesis_lb[i] = hypothesis_ub[i] - size[i]
        # hypothesis must be placed between the solution and the upper bound.
        else:
            hypothesis_lb[i] = lb[i] + 0.2 * max_size
            hypothesis_ub[i] = hypothesis_lb[i] + size[i]

    const_inequalities = []
    for j in range(dim):
        const_inequalities.append(hypothesis_ub[j])
        const_inequalities.append(-hypothesis_lb[j])
    const_inequalities = np.array(const_inequalities)

    hypothesis = Hypothesis(
        name="Weak",
        lb=hypothesis_lb,
        ub=hypothesis_ub,
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
):
    """
    Get a good hypothesis.

    Args:
        lb (np.ndarray): Lower bounds.
        ub (np.ndarray): Upper bounds.
        size (np.ndarray): Size of the hypothesis.
        sol (np.ndarray): Solution.

    Returns:
        Hypothesis: The good hypothesis.
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
    hypothesis_lb = lb.copy()
    hypothesis_ub = lb.copy()
    for i in range(dim):
        hypothesis_lb[i] = sol[i] - size[i] / 2
        hypothesis_ub[i] = hypothesis_lb[i] + size[i]

    const_inequalities = []
    for j in range(dim):
        const_inequalities.append(hypothesis_ub[j])
        const_inequalities.append(-hypothesis_lb[j])
    const_inequalities = np.array(const_inequalities)

    hypothesis = Hypothesis(
        name="Good",
        lb=hypothesis_lb,
        ub=hypothesis_ub,
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
        which takes a dimension parameter.

    Args:
        name (str): Name of the function.
        dim (int, optional): Dimension of the function. Defaults to 4.

    Returns:
        TestProblem: The test problem.
    """
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


def _get_branin_hypotheses():
    """
    Get the hypotheses for the Branin function.

    Returns:
        list: List of hypotheses.
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

    poor_hypothesis = Hypothesis(
        name="Poor",
        feature_names=func.input_columns,
        lb=lb,
        ub=ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )

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

    weak_hypothesis = Hypothesis(
        name="Weak",
        feature_names=func.input_columns,
        lb=lb,
        ub=ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )

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

    good_hypothesis = Hypothesis(
        name="Good",
        feature_names=func.input_columns,
        lb=lb,
        ub=ub,
        coeff_inequalities=coeff_inequalities,
        const_inequalities=const_inequalities,
        coeff_equalities=coeff_equalities,
        const_equalities=const_equalities,
    )

    scenarios = [
        [poor_hypothesis],
        [weak_hypothesis],
        [good_hypothesis],
        [good_hypothesis, poor_hypothesis],
        [good_hypothesis, weak_hypothesis],
        [weak_hypothesis, poor_hypothesis],
    ]
    return scenarios


def get_scenarios(
        func_name: str,
        dim: int = 4,
):
    """
    Get the scenarios for a given function.

    Args:
        func_name (str): Name of the function.
        dim (int, optional): Dimension of the function. Defaults to 4.

    Returns:
        list: List of hypotheses.
    """
    scenarios = None
    if func_name == "Branin":
        scenarios = _get_branin_hypotheses()
    else:
        # Compute the different hypotheses.
        func = get_function(
            name=func_name,
            dim=dim,
        )
        hypothesis_size = 2
        if func_name == "StyblinskiTang":
            hypothesis_size = 0.4
        hypothesis_size = np.full(
            func.problem.dim, hypothesis_size, dtype=float)
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


def get_scenario_name(scenario):
    if scenario == []:
        return "No hypothesis"
    elif len(scenario) > 3:
        return "Mixed Starting Hypotheses"
    name = '_'.join([s.name for s in scenario])
    return name
