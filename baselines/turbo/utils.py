"""
This module provides utility functions for working with test problems in the
TuRBO package.

The main function in this module is `get_function`, which retrieves a
`TestProblem` object representing a specific test function from the
`standard_test_functions` module.

"""

import inspect

from resources.functions import standard_test_functions as fcts
from resources.functions.test_problem import TestProblem


def get_function(
    name: str,
    dim: int = 4,
) -> TestProblem:
    """
    Returns the TestProblem object from 'standard_test_problems.py'
    that matches the given name and dimension.

    Args:
        name (str): The name of the function.
        dim (int, optional): The dimension of the function. Defaults to 4.

    Returns:
        TestProblem: The TestProblem object representing the function.

    Raises:
        ValueError: If the function with the given name is not found.
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
