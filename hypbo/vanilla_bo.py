"""
Module for Bayesian Optimization using a vanilla BO approach.

This module implements the BO (Bayesian Optimization) class which performs
the optimization process by iteratively probing candidate solutions, updating
the surrogate model, and logging the results.

Dependencies:
    torch, numpy, pandas, logging, warnings
"""

import torch
import warnings
from typing import Callable, Tuple, List, Dict
from collections import deque
import numpy as np
from .model import Model
import logging
import pandas as pd


warnings.filterwarnings("ignore")

tkwargs = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float32,
}


class BO:
    """
    Bayesian Optimization (BO) class for optimizing a black-box function.

    This class manages the optimization process, including initialization of candidate
    solutions, probing the objective function, updating the surrogate model, and tracking
    the best observed sample.
    """

    def __init__(
        self,
        experiment: Callable,
        pbounds: Dict[str, Tuple[float, float, float]],
        target_feature: str = "target",
        random_seed: int = 0,
        verbose: bool = True,
        decimals: int = 3,
    ):
        """
        Initialize the BO class.

        Args:
            experiment (Callable): The objective function to optimize, which must provide
                additional constraints via `get_all_constraints()`.
            pbounds (Dict[str, Tuple[float, float, float]]): Parameter bounds and additional settings.
            target_feature (str, optional): The key corresponding to the target metrics. Defaults to "target".
            random_seed (int, optional): Random seed for reproducibility. Defaults to 0.
            verbose (bool, optional): Enables verbose logging if set to True. Defaults to True.
            decimals (int, optional): Number of decimal places to format logged outputs. Defaults to 3.
        """
        self.experiment = experiment
        self.constraints = experiment.get_all_constraints()
        self.pbounds = pbounds
        self.target_feature = target_feature
        self.seed = random_seed

        # Data buffers
        self.queue = deque()
        self.train_x = torch.tensor([], **tkwargs)
        self.train_y = torch.tensor([], **tkwargs)
        self.train_iteration = torch.tensor(
            [], dtype=torch.long, device=tkwargs["device"]
        )
        self.best_sample = {p: None for p in pbounds.keys()}
        self.best_sample[self.target_feature] = -np.inf

        # Surrogate Model initialization
        self.model = Model(
            "Global",
            pbounds,
            random_seed=random_seed,
            **self.constraints,
        )

        # Logging configuration
        self.decimals = decimals
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def initialize_queue(self):
        """
        Initialize the candidate solution queue with random candidates.

        This method generates an initial batch of candidates using the
        random generator of surrogate model and extends the internal
        processing queue.
        """
        batch = self.model.generate_random_candidates(self.n_init * self.batch_size)
        self.queue.extend([candidate for candidate in batch])

    def recommend(self) -> torch.Tensor:
        """
        Recommend a new batch of candidate solutions for evaluation.

        It first attempts to retrieve candidates from the internal queue. If
        the queue is empty, it asks the surrogate model for recommendations.

        Returns:
            torch.Tensor: A tensor containing the recommended candidates.
        """
        batch = None
        try:
            batch = [self.queue.popleft() for _ in range(self.batch_size)]
            batch = torch.stack(batch, dim=0).to(**tkwargs)
        except IndexError:
            batch, _ = self.model.recommend(
                self.batch_size,
                best_f=self.train_y.max().item(),
            )

        return batch

    def update_model(self):
        """
        Update the surrogate model with the current training data.

        This method updates the model only if both training inputs and outputs
        are available.
        """
        if self.train_x.numel() == 0 or self.train_y.numel() == 0:
            return
        self.model.update(self.train_x, self.train_y)

    def format_sample(self, sample: Dict[str, float]) -> Dict[str, str]:
        """
        Format a sample for logging purposes.

        Args:
            sample (Dict[str, float]): The sample to format.

        Returns:
            Dict[str, str]: A dictionary where numerical values are formatted
            as strings with the specified number of decimals.
        """
        return {
            k: (f"{v:.{self.decimals}f}" if isinstance(v, (int, float)) else v)
            for k, v in sample.items()
        }

    def update_best_sample(self, x_batch: torch.Tensor, y_batch: torch.Tensor):
        """
        Update the best sample based on the current batch evaluation.

        If the maximum target value in the batch exceeds the current best, it
        updates the best sample and logs the new best sample.

        Args:
            x_batch (torch.Tensor): The batch of candidate parameters.
            y_batch (torch.Tensor): The corresponding objective function outputs.
        """
        if self.best_sample[self.target_feature] < y_batch.max().item():
            self.best_sample[self.target_feature] = y_batch.max().item()
            point = x_batch[y_batch.argmax()]
            self.best_sample.update(
                {k: v for k, v in zip(self.pbounds.keys(), point.tolist())}
            )
            logging.info(f"Best sample: {self.format_sample(self.best_sample)}")

    def probe(
        self, x_batch: List[Tuple[torch.Tensor, str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the objective function on a batch of candidate solutions and
        update training data.

        This method calls the experiment with the candidate batch, appends the
        results to the training data, updates iteration counts, and logs the
        evaluation results.

        Args:
            x_batch (List[Tuple[torch.Tensor, str]]): A list of candidate
                samples to probe.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The batch of input samples and
                their evaluated outputs.
        """
        y_batch = self.experiment(x_batch)

        # Append inputs and outputs to training data
        self.train_x = torch.cat([self.train_x, x_batch], dim=0)
        self.train_y = torch.cat([self.train_y, y_batch], dim=0)

        # Update iteration count for each sample
        if self.train_iteration.numel() > 0:
            new_val = self.train_iteration[-1] + 1
        else:
            new_val = 1

        new_iterations = torch.full(
            (x_batch.shape[0],),
            new_val,
            dtype=torch.long,
            device=tkwargs["device"],
        )
        self.train_iteration = torch.cat([self.train_iteration, new_iterations])

        # Log the batch data
        features = dict(
            zip(self.pbounds.keys(), np.round(x_batch.cpu().numpy().T, self.decimals))
        )
        targets = np.round(y_batch.cpu().numpy(), self.decimals)
        data = features
        data.update({self.target_feature: targets})
        df = pd.DataFrame(data=data)
        logging.info(f"Data:\n{df}")

        return x_batch, y_batch

    def maximize(self, n_init: int, budget: int, batch_size: int = 1):
        """
        Execute the Bayesian Optimization process to maximize the objective
        function.

        This method orchestrates the optimization process through
        initialization, candidate recommendation, objective evaluation,
        model updating, and logging.

        Args:
            n_init (int): The number of initial random candidate evaluations.
            budget (int): The total number of evaluations to perform
                (including initial evaluations).
            batch_size (int, optional): The number of samples to evaluate per
                iteration. Defaults to 1.

        Raises:
            ValueError: If n_init is less than 1, if batch_size is less than 1,
                        or if budget is insufficient.
        """
        if n_init < 1:
            raise ValueError("n_init must be greater than 0.")
        if budget < n_init * batch_size:
            raise ValueError("budget must be greater than n_init * batch_size.")
        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0.")
        logging.info("Starting BO optimization...")
        logging.info(f"Budget: {budget}")
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"Number of initializations: {n_init}")

        self.n_init = n_init
        self.budget = budget
        self.batch_size = batch_size
        self.initialize_queue()

        iteration = 1
        while len(self.queue) > 0 or iteration * self.batch_size <= budget:
            logging.info(f"Iteration {iteration}")
            next_batch = self.recommend()
            x_batch, y_batch = self.probe(next_batch)
            self.update_best_sample(x_batch, y_batch)
            self.update_model()
            iteration += 1

        logging.info("Maximization completed.")
        logging.info(f"Best sample: {self.format_sample(self.best_sample)}")

    def save_data(self, filepath: str):
        """
        Save the training data to a CSV file.

        The saved data includes the input features, target values, and
        iteration numbers.

        Args:
            filepath (str): The destination file path for saving the CSV data.

        Raises:
            ValueError: If no training data is available.
        """
        if self.train_x is None or self.train_y is None:
            raise ValueError("No training data available to save.")

        features = dict(zip(self.pbounds.keys(), self.train_x.cpu().numpy().T))
        targets = self.train_y.cpu().numpy()
        iterations = self.train_iteration.cpu().numpy()
        data = features
        data.update({self.target_feature: targets})
        data.update({"iteration": iterations})
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
