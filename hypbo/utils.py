import torch
from typing import Dict, Tuple
import logging

tkwargs = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float32,
}


class HypothesisHelper:
    def __init__(
        self,
        experiment_pbounds: Dict[str, Tuple[float, float, float]],
        optimum: torch.Tensor,  # shape: [d]
        size: torch.Tensor,  # shape: [d]
    ):
        self.pbounds = torch.tensor(
            [v[:2] for k, v in experiment_pbounds.items()], **tkwargs
        ).T
        self.experiment_pbounds = experiment_pbounds
        self.optimum_location = optimum
        self.size = size

        # Logs
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def _poor_hypothesis(self):
        left_diff = self.optimum_location - self.pbounds[0]
        right_diff = self.pbounds[1] - self.optimum_location
        size = self.size

        # Ensure that in every dimension, the hypothesis size is smaller than the maximum distance.
        max_diff = torch.max(left_diff, right_diff)
        if not torch.all(max_diff > size):
            size = max_diff / 3 + 1e-3 * torch.ones_like(size)
            self.logger.warning(
                "The poor hypothesis size is too large. Adjusting to fit within bounds."
            )

        # Choose side: if left_diff is greater than or equal to right_diff, pick lower side,
        # otherwise pick the upper side.
        left_side = left_diff >= right_diff
        hyp_lb = torch.where(left_side, self.pbounds[0], self.pbounds[1] - size)
        hyp_ub = torch.where(left_side, self.pbounds[0] + size, self.pbounds[1])

        result = {
            "name": "Poor",
            "pbounds": {
                name: (hyp_lb[i].item(), hyp_ub[i].item(), v[2])
                for i, (name, v) in enumerate(self.experiment_pbounds.items())
            },
        }
        return result

    def _weak_hypothesis(self):
        size = self.size
        left_diff = self.optimum_location - self.pbounds[0]
        right_diff = self.pbounds[1] - self.optimum_location
        max_diff = torch.max(left_diff, right_diff)
        if not torch.all(max_diff > self.size):
            size = max_diff / 3 + 1e-3 * torch.ones_like(size)
            self.logger.warning(
                "The weak hypothesis size is too large. Adjusting to fit within bounds."
            )

        left_side = left_diff >= right_diff

        # For left side: place hypothesis between lb and optimum.
        hyp_ub_left = self.optimum_location - 0.2 * left_diff
        hyp_lb_left = hyp_ub_left - size
        # For right side: place hypothesis between optimum and ub.
        hyp_lb_right = self.optimum_location + 0.2 * right_diff
        hyp_ub_right = hyp_lb_right + size

        hyp_lb = torch.where(left_side, hyp_lb_left, hyp_lb_right)
        hyp_ub = torch.where(left_side, hyp_ub_left, hyp_ub_right)

        result = {
            "name": "Weak",
            "pbounds": {
                name: (hyp_lb[i].item(), hyp_ub[i].item(), v[2])
                for i, (name, v) in enumerate(self.experiment_pbounds.items())
            },
        }
        return result

    def _good_hypothesis(self):
        # Center the hypothesis on the optimum.
        size = self.size
        hyp_lb = self.optimum_location - size / 2
        hyp_ub = hyp_lb + size

        if not torch.all(hyp_lb >= self.pbounds[0]) or not torch.all(
            hyp_ub <= self.pbounds[1]
        ):
            hyp_lb = (self.optimum_location - self.pbounds[0]) / 2 + self.pbounds[0]
            hyp_ub = self.pbounds[1] - (self.pbounds[1] - self.optimum_location) / 2
            self.logger.warning(
                "Adjusted the good hypothesis bounds to fit within the defined limits."
            )

        result = {
            "name": "Good",
            "pbounds": {
                name: (hyp_lb[i].item(), hyp_ub[i].item(), v[2])
                for i, (name, v) in enumerate(self.experiment_pbounds.items())
            },
        }
        return result

    def create_hypotheses(self):
        return (
            self._poor_hypothesis(),
            self._weak_hypothesis(),
            self._good_hypothesis(),
        )


if __name__ == "__main__":

    h = HypothesisHelper(
        experiment_pbounds={"x1": (-32, 32, 0.5), "x2": (-32, 32, 1)},
        optimum=torch.tensor([0.0, 0.0], **tkwargs),
        size=torch.tensor([3.0, 3.0], **tkwargs),
    )
    print(h.create_hypotheses())
