import torch
from typing import Any, Dict, Optional
from botorch.test_functions.synthetic import (
    Ackley,
    Beale,
    Branin,
    Bukin,
    Cosine8,
    DixonPrice,
    DropWave,
    EggHolder,
    Griewank,
    Hartmann,
    HolderTable,
    Levy,
    Michalewicz,
    Powell,
    Rastrigin,
    Rosenbrock,
    Shekel,
    SixHumpCamel,
    StyblinskiTang,
    ThreeHumpCamel,
    # Constrained functions
    ConstrainedGramacy,
    ConstrainedHartmann,
    ConstrainedHartmannSmooth,
    PressureVessel,
    TensionCompressionString,
    SpeedReducer,
)


# Set device and dtype
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


class ExperimentTemplate:
    def __init__(self, base_experiment: Any, is_synthetic: bool = True):
        """
        Initialize the experiment template.

        Args:
            base_experiment: An object defining the experiment's constraints and feasibility method.
            is_synthetic: Flag to indicate if the experiment is synthetic.
        """
        self.base_experiment = base_experiment
        self.is_synthetic = is_synthetic

    @property
    def discretization_steps(self):
        if hasattr(self.base_experiment, "discretization_steps"):
            return self.base_experiment.discretization_steps
        return None

    @property
    def bounds(self):
        """
        Retrieve the bounds of the experiment.

        Returns:
            A tensor of shape (2, D) where D is the number of dimensions.
            The first row contains lower bounds and the second row contains upper bounds.
        """
        return self.base_experiment.bounds

    @property
    def dim(self):
        """
        Retrieve the number of dimensions of the experiment.

        Returns:
            An integer representing the number of dimensions.
        """
        return self.base_experiment.dim

    @property
    def feature_names(self):
        """
        Retrieve the names of the experiment's features.

        Returns:
            A list of strings containing the names of the experiment's features.
        """
        feature_names = []
        if hasattr(self.base_experiment, "feature_names"):
            feature_names = self.base_experiment.feature_names
        else:
            for i in range(self.dim):
                feature_names.append(f"x{i+1}")

        return feature_names

    @property
    def pbounds(self):
        feature_names = self.feature_names
        pbounds = {
            feature_names[i]: (
                self.bounds[0, i].item(),
                self.bounds[1, i].item(),
                None,
            )
            for i in range(self.dim)
        }
        if self.discretization_steps is not None:
            for k, v in pbounds.items():
                pbounds[k] = (v[0], v[1], self.discretization_steps)
        return pbounds

    @property
    def optimums(self):
        """
        Retrieve the optimums for the experiment.

        Returns:
            A list of optimizers.
        """
        if hasattr(self.base_experiment, "_optimizers"):
            return self.base_experiment._optimizers
        return None

    def get_all_constraints(self) -> Dict[str, Optional[Any]]:
        """
        Retrieve constraints and an initial condition generator if applicable.

        Returns:
            A dictionary with the following keys:
                - "linear_eq_constraints": Linear equality constraints (if any).
                - "linear_ineq_constraints": Linear inequality constraints (if any).
                - "nonlinear_constraints": Nonlinear constraints function (if any).
                - "ic_generator": Function to generate feasible initial conditions (if applicable).
        """
        constraints = {
            "linear_eq_constraints": None,
            "linear_ineq_constraints": None,
            "nonlinear_constraints": None,
            "ic_generator": None,
            "is_feasible": None,
        }

        if not self.is_synthetic:
            constraints = self.base_experiment.get_all_constraints()
        elif (
            self.is_synthetic
            and hasattr(self.base_experiment, "num_constraints")
            and self.base_experiment.num_constraints > 0
        ):
            # Assign the nonlinear constraints function if available.
            constraints["nonlinear_constraints"] = (
                self.base_experiment.evaluate_slack_true
            )
            constraints["is_feasible"] = self.base_experiment.is_feasible

            def _ic_generator(
                acq_function: Any, bounds: torch.Tensor, q: int, **kwargs
            ) -> torch.Tensor:
                """
                Generate a batch of feasible initial conditions.

                Args:
                    acq_function: Acquisition function (unused in this implementation).
                    bounds: A tensor of shape (2, D) where D is the number of dimensions. The first row
                            is lower bounds and second row is upper bounds.
                    q: The number of initial conditions to generate.
                    **kwargs: Additional keyword arguments (not used).

                Returns:
                    A tensor of shape (q, D) containing feasible initial conditions.

                Raises:
                    RuntimeError: If a feasible sample cannot be found within 200 attempts.
                """
                batch = torch.zeros((q, bounds.size(1)), **tkwargs)
                for i in range(q):
                    for attempt in range(200):
                        sample = (
                            torch.rand(1, bounds.size(1), **tkwargs)
                            * (bounds[1] - bounds[0])
                            + bounds[0]
                        )
                        if self.base_experiment.is_feasible(sample):
                            batch[i] = sample
                            break
                    else:
                        raise RuntimeError(
                            "Could not generate feasible initial conditions."
                        )
                return batch

            constraints["ic_generator"] = _ic_generator

        return constraints

    def __call__(self, *args, **kwds):
        """
        Call the base experiment with the given arguments.

        Args:
            *args: Positional arguments to be passed to the base experiment.
            **kwds: Keyword arguments to be passed to the base experiment.

        Returns:
            The result of calling the base experiment.
        """
        return self.base_experiment(*args, **kwds)


# Synthetic function template library
library = [
    Ackley,
    Beale,
    Branin,
    Bukin,
    Cosine8,
    DixonPrice,
    DropWave,
    EggHolder,
    Griewank,
    Hartmann,
    HolderTable,
    Levy,
    Michalewicz,
    Powell,
    Rastrigin,
    Rosenbrock,
    Shekel,
    SixHumpCamel,
    StyblinskiTang,
    ThreeHumpCamel,
    ConstrainedGramacy,
    ConstrainedHartmann,
    ConstrainedHartmannSmooth,
    PressureVessel,
    TensionCompressionString,
    SpeedReducer,
]
library = {func.__name__: ExperimentTemplate(func(negate=True)) for func in library}

if __name__ == "__main__":
    func_test = Ackley()
    exp_test = ExperimentTemplate(func_test)
    constraints = exp_test.get_all_constraints()
    if constraints["ic_generator"]:
        initial_conditions = constraints["ic_generator"](
            None,
            func_test.bounds,
            10,
        )
        print(initial_conditions)
    else:
        print("No initial condition generator available.")
