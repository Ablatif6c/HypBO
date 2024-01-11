from typing import List, Tuple

import numpy as np
from scipy.optimize import linprog


class Hypothesis:
    def __init__(
        self,
        name: str,
        lb: np.array,
        ub: np.array,
        feature_names: List[str] = [],
        coeff_inequalities: np.array = None,
        const_inequalities: np.array = None,
        coeff_equalities: np.array = None,
        const_equalities: np.array = None,
    ) -> None:

        if (
            (coeff_inequalities is None)
            and (const_inequalities is None)
            and (coeff_equalities is None)
            and (const_equalities is None)
        ):
            raise Exception("one must be non null")
        if coeff_inequalities is not None:
            assert const_inequalities is not None
        if coeff_equalities is not None:
            assert const_equalities is not None

        self.name = name
        self.feature_names = feature_names
        if len(self.feature_names) != len(lb):
            self.feature_names = []
            for i in range(len(lb)):
                self.feature_names.append(f"x{i}")
        self.lb = lb
        self.ub = ub
        self.coeff_inequalities = coeff_inequalities
        self.const_inequalities = const_inequalities
        self.coeff_equalities = coeff_equalities
        self.const_equalities = const_equalities
        self.samples = []
        self.sol_bounds = self.__get_sol_bounds()

    def __get_sol_bounds(self) -> Tuple[np.array, np.array]:
        """Returns the new bounds of the features satisfying the hypotheses"""
        dims = len(self.lb)
        bounds = [(self.lb[i], self.ub[i]) for i in range(dims)]
        sol_lb = np.array([])
        sol_ub = np.array([])

        for i in range(dims):
            # Find the minimum value of the feature
            coefficients_min = [0] * dims
            coefficients_min[i] = 1
            res = linprog(
                coefficients_min,
                A_ub=self.coeff_inequalities,
                b_ub=self.const_inequalities,
                A_eq=self.coeff_equalities,
                b_eq=self.const_equalities,
                bounds=bounds,
            )
            sol_lb = np.append(sol_lb, res.fun)

            # Find the maximal value of the feature
            # = minimal value of -feature
            coefficients_max = [0] * dims
            coefficients_max[i] = -1
            res = linprog(
                coefficients_max,
                A_ub=self.coeff_inequalities,
                b_ub=self.const_inequalities,
                A_eq=self.coeff_equalities,
                b_eq=self.const_equalities,
                bounds=bounds,
            )
            # max = opposite of value of -feature
            sol_ub = np.append(sol_ub, -res.fun)

        if sol_lb == np.array([]) or sol_ub == np.array([]):
            return None, None

        return sol_lb, sol_ub

    def apply(self, input) -> bool:
        # Check the boundaries.
        for i, v in enumerate(input):
            if not (self.lb[i] <= v <= self.ub[i]):
                return False

        # Check the hypothesis.
        if self.coeff_inequalities is not None:
            inequalities_state = (
                self.coeff_inequalities.dot(input) <= self.const_inequalities
            ).all()
            if not inequalities_state:
                return False

        if self.coeff_equalities is not None:
            equalities_state = (
                self.coeff_equalities.dot(input) == self.const_equalities
            ).all()
            if not equalities_state:
                return False

        return True

    def is_valid(self):
        lb_OK = self.sol_bounds[0] is not None
        ub_OK = self.sol_bounds[1] is not None
        if lb_OK and ub_OK:
            return True
        return False

    def stringify(self, ge=False,):
        constraints = []

        ge = 1 if ge is False else -1
        # Inequalities
        if self.coeff_inequalities is None:
            return []

        for i in range(len(self.coeff_inequalities)):
            coeffs = []
            # Add the + signs for the positive coeffs
            for c in self.coeff_inequalities[i, :]:
                if str(ge * c).startswith("-"):
                    coeffs.append(str(ge * c))
                else:
                    coeffs.append(f"+{ge * c}")
            # Make sure you remove the + sign for the first coeff
            if coeffs[0].startswith("+"):
                coeffs[0] = coeffs[0][1:]

            constraint = ""
            for c, f in zip(coeffs, self.feature_names):
                constraint += f"{c}*{f}"
            constraint += f"-({ge * self.const_inequalities[i]})"
            constraints.append(constraint)
        # Equalities
        return constraints

    def clear_samples(self):
        self.samples = []