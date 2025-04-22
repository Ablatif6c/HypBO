import torch
import uuid
from botorch.acquisition import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Dict, List, Callable, Optional, Tuple

# Set torch device and dtype
tkwargs = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float32,
}


class Model:
    """
    A Model class that handles constraints, sampling, Gaussian Process modeling,
    candidate generation, and recommendations using BoTorch.
    """

    def __init__(
        self,
        name: str,
        pbounds: Dict[str, Tuple[float, float, float]],
        linear_eq_constraints: Optional[List[Tuple[torch.Tensor, torch.Tensor, float]]],
        linear_ineq_constraints: Optional[
            List[Tuple[torch.Tensor, torch.Tensor, float]]
        ],
        nonlinear_constraints: Optional[List[Tuple[torch.Tensor, torch.Tensor, float]]],
        ic_generator: Optional[Callable],
        is_feasible: Optional[Callable],
        num_restarts: int = 10,
        mc_samples: int = 256,
        raw_samples: int = 512,
    ):
        """
        Initialize the Model.

        Args:
            name (str): Name of the model.
            pbounds (Dict[str, Tuple[float, float, float]]): Dictionary mapping parameter names
                to a tuple (lower_bound, upper_bound, step_size).
            linear_eq_constraints (Optional[List[Tuple[torch.Tensor, torch.Tensor, float]]]):
                List of linear equality constraints.
            linear_ineq_constraints (Optional[List[Tuple[torch.Tensor, torch.Tensor, float]]]):
                List of linear inequality constraints.
            nonlinear_constraints (Optional[List[Tuple[torch.Tensor, torch.Tensor, float]]]):
                List of nonlinear constraints.
            ic_generator (Optional[Callable]): Function to generate initial conditions.
            is_feasible (Optional[Callable]): Function to check feasibility of a candidate.
            num_restarts (int, optional): Number of restarts during optimization. Defaults to 10.
            mc_samples (int, optional): Number of Monte Carlo samples for acquisition function.
                Defaults to 256.
            raw_samples (int, optional): Number of raw samples for acquisition optimization.
                Defaults to 512.
        """
        self.name = name
        self.id = uuid.uuid4()

        # Set parameter bounds and discretization steps if provided
        self.bounds = torch.tensor([v[:2] for k, v in pbounds.items()], **tkwargs).T
        self.discretization_steps = self.get_discretization_steps(pbounds)

        # Configure sampler and optimization parameters
        self.num_restarts = num_restarts
        self.mc_samples = mc_samples
        self.raw_samples = raw_samples
        self.sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.mc_samples]),
        )

        # Initialize constraints and related functions
        self.constraints = {}
        self.init_constraints(
            linear_eq_constraints,
            linear_ineq_constraints,
            nonlinear_constraints,
            ic_generator,
            is_feasible,
        )

        # Initialize training data attributes
        self.train_x: torch.Tensor = None
        self.train_y: torch.Tensor = None

        # Gaussian Process model placeholder
        self.gp: SingleTaskGP = None

    @property
    def has_constraints(self):
        """
        Check if any constraints are defined.

        Returns:
            bool: True if any constraints are present.
        """
        return (
            self.has_linear_eq_constraints
            or self.has_linear_ineq_constraints
            or self.has_nonlinear_constraints
        )

    @property
    def has_linear_eq_constraints(self):
        """
        Check if linear equality constraints are defined.

        Returns:
            bool: True if linear equality constraints exist.
        """
        return self.constraints.get("linear_eq_constraints", None) is not None

    @property
    def has_linear_ineq_constraints(self):
        """
        Check if linear inequality constraints are defined.

        Returns:
            bool: True if linear inequality constraints exist.
        """
        return self.constraints.get("linear_ineq_constraints", None) is not None

    @property
    def has_nonlinear_constraints(self):
        """
        Check if nonlinear constraints are defined.

        Returns:
            bool: True if nonlinear constraints exist.
        """
        return self.constraints.get("nonlinear_constraints", None) is not None

    @property
    def has_is_feasible(self):
        """
        Check if a feasibility checking function is provided.

        Returns:
            bool: True if feasibility check function exists.
        """
        return self.constraints.get("is_feasible", None) is not None

    def get_discretization_steps(
        self, pbounds: Dict[str, Tuple[float, float, float]]
    ) -> Optional[torch.Tensor]:
        """
        Get discretization steps from the provided parameter bounds.

        Args:
            pbounds (Dict[str, Tuple[float, float, float]]): Parameter bounds.

        Returns:
            Optional[torch.Tensor]: Discretization steps if valid, else None.
        """
        discretization_steps: Optional[torch.Tensor] = None
        if all(len(p) == 3 for p in pbounds.values()) and all(
            isinstance(p[2], (float, int)) for p in pbounds.values()
        ):
            discretization_steps = torch.tensor(
                [p[2] for p in pbounds.values()], **tkwargs
            )
        return discretization_steps

    def init_constraints(
        self,
        linear_eq_constraints,
        linear_ineq_constraints,
        nonlinear_constraints,
        ic_generator,
        is_feasible,
    ):
        """
        Initialize constraints and related functions.

        Args:
            linear_eq_constraints: Linear equality constraints.
            linear_ineq_constraints: Linear inequality constraints.
            nonlinear_constraints: Nonlinear constraints.
            ic_generator: Function to generate initial conditions.
            is_feasible: Function to check feasibility of candidates.

        Raises:
            ValueError: If constraints are set but either ic_generator or is_feasible is not provided.
        """
        # Add each constraint or function to the constraints dictionary if provided.
        for c_name, c in [
            ("linear_eq_constraints", linear_eq_constraints),
            ("linear_ineq_constraints", linear_ineq_constraints),
            ("nonlinear_constraints", nonlinear_constraints),
            ("ic_generator", ic_generator),
            ("is_feasible", is_feasible),
        ]:
            if c:
                self.constraints[c_name] = c

        # Ensure both initial condition generator and feasibility checker are provided
        # when constraints are present.
        if self.has_constraints and (
            self.constraints.get("ic_generator", None) is None
            or self.constraints.get("is_feasible", None) is None
        ):
            raise ValueError(
                "IC generator and is_feasible must be provided when constraints are present."
            )

        # Generate starting initial conditions if constraints exist.
        if self.has_constraints:
            self.constraints["batch_initial_conditions"] = ic_generator(
                None,
                self.bounds,
                self.num_restarts,
                **tkwargs,
            )

    def discretize_if_necessary(self, points):
        """
        Discretize points based on provided discretization steps.

        Args:
            points (torch.Tensor): The input points.

        Returns:
            torch.Tensor: Discretized points.
        """
        if self.discretization_steps is not None:
            points = (
                torch.round(points / self.discretization_steps)
                * self.discretization_steps
            )
        return points

    def generate_random_candidates(self, n) -> torch.Tensor:
        """
        Generate random candidate points.

        Args:
            n (int): Number of candidates to generate.

        Returns:
            torch.Tensor: Generated candidate points.
        """
        if self.has_constraints:
            # Use the custom initial condition generator if constraints are defined.
            candidates = self.constraints["ic_generator"](
                None,
                self.bounds,
                n,
                **tkwargs,
            )
        else:
            # Uniform random sampling within bounds.
            candidates = (
                torch.rand(n, self.bounds.shape[1], **tkwargs)
                * (self.bounds[1, :] - self.bounds[0, :])
                + self.bounds[0, :]
            )

        # Discretize the candidates if discretization steps are provided.
        if self.discretization_steps is not None:
            candidates = self.discretize_if_necessary(candidates)
        return candidates

    def filter_data(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter input data x and corresponding output data y.
        Only points within bounds and satisfying feasibility constraints (if defined) are kept.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Output data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing filtered x and y.
        """
        # Create a mask that is True for points within the bounds.
        mask = torch.all(
            (x >= self.bounds[0, :]) & (x <= self.bounds[1, :]),
            dim=1,
        )
        # Further mask by feasibility if applicable.
        if self.has_is_feasible:
            mask_constraints = self.constraints["is_feasible"](x)
            mask = mask & mask_constraints
        masked_x = x[mask]
        masked_y = y[mask]
        return masked_x, masked_y

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update the model with new training data and refit the Gaussian Process.

        Args:
            x (torch.Tensor): Input training points.
            y (torch.Tensor): List of function evaluations at the training points.
        """
        self.train_x = x
        self.train_y = y
        # Initialize the GP model with standardized outcomes and normalized inputs.
        self.gp = SingleTaskGP(
            self.train_x,
            self.train_y.view(-1, 1),
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(d=self.train_x.shape[1]),
        )
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        # Fit the Gaussian Process model.
        fit_gpytorch_mll(mll)

    def recommend(
        self, batch_size: int, best_f: Optional[float]
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Recommend new candidate points based on the acquisition function.

        Args:
            batch_size (int): Number of candidates to recommend.
            best_f (Optional[float]): Best observed function value so far.

        Returns:
            Tuple[torch.Tensor, List[float]]: Tuple containing the recommended batch of points and
            their corresponding acquisition values.

        Raises:
            ValueError: If the model hasn't been updated with training data.
        """
        if self.train_x.numel() == 0 or self.train_y.numel() == 0:
            raise ValueError("Model has not been updated with training data.")

        if self.has_nonlinear_constraints:
            batch, acq_values = self._fantasy_method_batch(batch_size)

        else:
            # Define the acquisition function using qLogExpectedImprovement.
            acq_func = qLogExpectedImprovement(
                self.gp,
                best_f=best_f if best_f is not None else self.train_y.max().item(),
                sampler=self.sampler,
            )

            # Optimize the acquisition function to get a batch of new candidates.
            constraints_excluding = {
                k: v for k, v in self.constraints.items() if k != "is_feasible"
            }
            batch, acq_values = optimize_acqf(
                acq_function=acq_func,
                bounds=self.bounds,
                q=batch_size,
                num_restarts=10,
                raw_samples=512,
                return_best_only=True,
                sequential=True,
                **constraints_excluding,
            )

        # Discretize the obtained batch if needed.
        batch = self.discretize_if_necessary(batch)

        return batch, acq_values

    def _fantasy_method_batch(self, batch_size: int):
        """
        Build a batch of size `batch_size` by picking one candidate at a time,
        'fantasizing' its outcome, and updating the GP before choosing the
        next one.
        """
        candidates = []
        acq_values = []
        train_x = self.train_x
        train_y = self.train_y
        gp = self.gp
        constraints_excluding = {
            k: v for k, v in self.constraints.items() if k != "is_feasible"
        }
        for i in range(batch_size):
            # 1) Optimize acquisition function (single point, q=1)
            acq_func = qLogExpectedImprovement(
                self.gp,
                best_f=train_y.max().item(),
                sampler=self.sampler,
            )

            candidate, acq_value = optimize_acqf(
                acq_function=acq_func,
                bounds=self.bounds,
                q=1,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
                return_best_only=True,
                **constraints_excluding,
            )
            acq_values.append(acq_value)

            # 2) Discretize / round if needed
            candidate = self.discretize_if_necessary(candidate)
            candidate = candidate.squeeze(0)  # Shape: (d,)

            # 3) Check feasibility; if infeasible, try to generate a feasible candidate.
            if not self.constraints["is_feasible"](candidate):
                max_attempts = 10
                for attempt in range(max_attempts):
                    candidate = self.constraints["ic_generator"](0, self.bounds, 1)[0]
                    candidate = self.discretize_if_necessary(candidate)
                    if self.constraints["is_feasible"](candidate):
                        break
                else:
                    raise RuntimeError(
                        f"Candidate remains infeasible after {max_attempts} attempts."
                    )

            candidates.append(candidate)

            # 4) Fantasize the outcome for the candidate and update the training data.
            with torch.no_grad():
                posterior = gp.posterior(candidate.unsqueeze(0))
                fantasy_y = posterior.mean.squeeze(-1)
            train_x = torch.cat([train_x, candidate.unsqueeze(0)], dim=0)
            train_y = torch.cat([train_y, fantasy_y], dim=0)
            # Retrain GP with updated data
            gp = SingleTaskGP(
                train_x,
                train_y.view(-1, 1),
                outcome_transform=Standardize(m=1),
                input_transform=Normalize(d=train_x.shape[1]),
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

        batch = torch.stack(candidates, dim=0)
        acq_values = torch.stack(acq_values, dim=0).flatten()
        return batch, acq_values
