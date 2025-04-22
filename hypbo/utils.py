"""
This module provides the HypothesisHelper class which generates three types of hypotheses
("Poor", "Weak", "Good") based on given parameter bounds, an optimum location, and a size.
It adjusts the hypothesis bounds if they exceed the given parameter limits.
"""

import torch
from typing import Dict, Tuple
import logging

# Set tensor related keyword arguments based on the available device.
tkwargs = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float32,
}


class HypothesisHelper:
    """
    A helper class to generate hypothesis parameter bounds for an experiment.

    Attributes:
        experiment_pbounds (Dict[str, Tuple[float, float, float]]): The experiment parameter
            bounds in the format {parameter: (lb, ub, granularity)}.
        optimum_location (torch.Tensor): The optimal location tensor for the experiment.
        size (torch.Tensor): The size of the hypothesis in each dimension.
        pbounds (torch.Tensor): A tensor of the lower and upper bounds for each parameter.
        logger (logging.Logger): Logger for information and warning messages.
    """

    def __init__(
        self,
        experiment_pbounds: Dict[str, Tuple[float, float, float]],
        optimum: torch.Tensor,  # shape: [d]
        size: torch.Tensor,  # shape: [d]
    ):
        """
        Initializes the HypothesisHelper with experiment bounds, optimum location, and hypothesis size.

        Args:
            experiment_pbounds (Dict[str, Tuple[float, float, float]]): Dictionary of experiment bounds.
            optimum (torch.Tensor): Optimum location tensor.
            size (torch.Tensor): Size of the hypothesis in each dimension.
        """
        # Convert the pbounds to a tensor for computation.
        self.pbounds = torch.tensor(
            [v[:2] for k, v in experiment_pbounds.items()], **tkwargs
        ).T
        self.experiment_pbounds = experiment_pbounds
        self.optimum_location = optimum
        self.size = size

        # Setup logger.
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def _poor_hypothesis(self) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
        """
        Create a 'Poor' hypothesis bound based on the optimum location and parameter bounds.

        Returns:
            dict: A dictionary containing the hypothesis name and its parameter bounds.
        """
        left_diff = self.optimum_location - self.pbounds[0]
        right_diff = self.pbounds[1] - self.optimum_location
        size = self.size

        # Ensure the hypothesis size does not exceed the maximum allowable difference.
        max_diff = torch.max(left_diff, right_diff)
        if not torch.all(max_diff > size):
            size = max_diff / 3 + 1e-3 * torch.ones_like(size)
            self.logger.warning(
                "The poor hypothesis size is too large. Adjusting to fit within bounds."
            )

        # Determine the side (left or right) with the larger distance.
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

    def _weak_hypothesis(self) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
        """
        Create a 'Weak' hypothesis bound based on the optimum location and parameter bounds.

        Returns:
            dict: A dictionary containing the hypothesis name and its parameter bounds.
        """
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

        # For the left side: place the hypothesis between the lower bound and the optimum.
        hyp_ub_left = self.optimum_location - 0.2 * left_diff
        hyp_lb_left = hyp_ub_left - size
        # For the right side: place the hypothesis between the optimum and the upper bound.
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

    def _good_hypothesis(self) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
        """
        Create a 'Good' hypothesis bound by centering the hypothesis around the optimum location.

        Returns:
            dict: A dictionary containing the hypothesis name and its parameter bounds.
        """
        size = self.size
        hyp_lb = self.optimum_location - size / 2
        hyp_ub = hyp_lb + size

        # Adjust hypothesis if it does not fully lie within parameter bounds.
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

    def create_hypotheses(
        self,
    ) -> Tuple[Dict[str, Dict[str, Tuple[float, float, float]]]]:
        """
        Creates all three types of hypotheses: Poor, Weak, and Good.

        Returns:
            tuple: A tuple containing three dictionaries corresponding to the generated hypotheses.
        """
        return (
            self._poor_hypothesis(),
            self._weak_hypothesis(),
            self._good_hypothesis(),
        )
