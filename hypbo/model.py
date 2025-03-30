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

tkwargs = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float32,
}


class Model:
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
        random_seed: int = 0,
    ):
        self.name = name
        self.id = uuid.uuid4()
        self.bounds = torch.tensor([v[:2] for k, v in pbounds.items()], **tkwargs).T
        self.discretization_steps = None
        if all(len(p) == 3 for p in pbounds.values()) and all(
            isinstance(p[2], (float, int)) for p in pbounds.values()
        ):
            self.discretization_steps = torch.tensor(
                [p[2] for p in pbounds.values()], **tkwargs
            )

        # Sampler
        self.num_restarts = num_restarts
        self.mc_samples = mc_samples
        self.raw_samples = raw_samples
        self.sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.mc_samples]),
        )

        # Constraints
        self.constraints = {}
        self.init_constraints(
            linear_eq_constraints,
            linear_ineq_constraints,
            nonlinear_constraints,
            ic_generator,
            is_feasible,
        )

        # Data
        self.train_x: torch.Tensor = None
        self.train_y: torch.Tensor = None

        # Model
        self.gp: SingleTaskGP = None

    @property
    def has_linear_eq_constraints(self):
        return self.constraints.get("linear_eq_constraints", None) is not None

    @property
    def has_linear_ineq_constraints(self):
        return self.constraints.get("linear_ineq_constraints", None) is not None

    @property
    def has_nonlinear_constraints(self):
        return self.constraints.get("nonlinear_constraints", None) is not None

    @property
    def has_ic_generator(self):
        return self.constraints.get("ic_generator", None) is not None

    @property
    def has_is_feasible(self):
        return self.constraints.get("is_feasible", None) is not None

    def init_constraints(
        self,
        linear_eq_constraints,
        linear_ineq_constraints,
        nonlinear_constraints,
        ic_generator,
        is_feasible,
    ):
        for c_name, c in [
            ("linear_eq_constraints", linear_eq_constraints),
            ("linear_ineq_constraints", linear_ineq_constraints),
            ("nonlinear_constraints", nonlinear_constraints),
            ("ic_generator", ic_generator),
            ("is_feasible", is_feasible),
        ]:
            if c:
                self.constraints[c_name] = c
        if self.constraints != {}:
            self.constraints["batch_initial_conditions"] = ic_generator(
                None,
                self.bounds,
                self.num_restarts,
                **tkwargs,
            )

    def discretize_if_necessary(self, points):
        if self.discretization_steps is not None:
            points = (
                torch.round(points / self.discretization_steps)
                * self.discretization_steps
            )
        return points

    def generate_random_candidates(self, n) -> torch.Tensor:
        if self.has_ic_generator:
            candidates = self.constraints["ic_generator"](
                None,
                self.bounds,
                n,
                **tkwargs,
            )
        elif self.has_is_feasible:
            candidates = torch.empty((0, self.bounds.shape[1]), **tkwargs)
            for _ in range(self.num_restarts):
                new_candidates = (
                    torch.rand(n, self.bounds.shape[1], **tkwargs)
                    * (self.bounds[1, :] - self.bounds[0, :])
                    + self.bounds[0, :]
                )
                mask = self.constraints["is_feasible"](new_candidates)
                candidates = torch.cat(
                    (candidates, new_candidates[mask]),
                    dim=0,
                )
                if candidates.shape[0] >= n:
                    break
            if candidates.shape[0] == 0:
                raise RuntimeError("Could not generate feasible initial conditions.")
        elif self.has_linear_eq_constraints:
            # TODO put this in a function
            candidates = torch.empty((0, self.bounds.shape[1]), **tkwargs)
            for A, b, _ in self.constraints["linear_eq_constraints"]:
                new_candidates = (
                    torch.rand(n, self.bounds.shape[1], **tkwargs)
                    * (self.bounds[1, :] - self.bounds[0, :])
                    + self.bounds[0, :]
                )
                mask = torch.abs(A @ new_candidates.T - b) < 1e-3
                candidates = torch.cat(
                    (candidates, new_candidates[mask]),
                    dim=0,
                )
                if candidates.shape[0] >= n:
                    break
            if candidates.shape[0] == 0:
                raise RuntimeError("Could not generate feasible initial conditions.")
        else:
            candidates = (
                torch.rand(n, self.bounds.shape[1], **tkwargs)
                * (self.bounds[1, :] - self.bounds[0, :])
                + self.bounds[0, :]
            )

        if self.discretization_steps is not None:
            candidates = self.discretize_if_necessary(candidates)
        return candidates

    def filter_data(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.all(
            (x >= self.bounds[0, :]) & (x <= self.bounds[1, :]),
            dim=1,
        )
        if self.has_is_feasible:
            mask_constraints = self.constraints["is_feasible"](x)
            mask = mask & mask_constraints
        elif self.has_linear_eq_constraints:
            for A, b, _ in self.constraints["linear_eq_constraints"]:
                mask = mask & (torch.abs(A @ x.T - b) < 1e-3)
        elif self.has_linear_ineq_constraints:
            for A, b, _ in self.constraints["linear_ineq_constraints"]:
                mask = mask & (A @ x.T <= b)
        masked_x = x[mask]
        masked_y = y[mask]
        return masked_x, masked_y

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.train_x = x
        self.train_y = y
        self.gp = SingleTaskGP(
            self.train_x,
            self.train_y.view(-1, 1),
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(d=self.train_x.shape[1]),
        )
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)

    def recommend(self, batch_size: int) -> Tuple[torch.Tensor, List[float]]:
        if self.train_x is None or self.train_y is None:
            raise ValueError("Model has not been updated with training data.")

        qei = qLogExpectedImprovement(
            self.gp,
            best_f=self.train_y.max(),
            sampler=self.sampler,
        )
        batch, acq_values = optimize_acqf(
            acq_function=qei,
            bounds=self.bounds,
            q=batch_size,
            num_restarts=10,
            raw_samples=512,
            return_best_only=True,
            sequential=True,
        )
        batch = self.discretize_if_necessary(batch)

        return batch, acq_values
