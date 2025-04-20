import torch
import warnings
from typing import Callable, Tuple, List, Optional
from collections import deque
import numpy as np
from .model import Model
from typing import Dict
import logging

import pandas as pd


warnings.filterwarnings("ignore")

tkwargs = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float32,
}


class BO:
    def __init__(
        self,
        experiment: Callable,
        pbounds: Dict[str, Tuple[float, float, float]],
        target_feature: str = "target",
        random_seed: int = 0,
        verbose: bool = True,
        decimals: int = 3,
    ):
        self.experiment = experiment
        self.constraints = experiment.get_all_constraints()
        self.pbounds = pbounds
        self.target_feature = target_feature
        self.seed = random_seed

        # Data
        self.queue = deque()
        self.train_x: Optional[torch.Tensor] = None
        self.train_y: Optional[torch.Tensor] = None
        self.train_iteration: Optional[torch.Tensor] = None
        self.best_sample = {p: None for p in pbounds.keys()}
        self.best_sample[self.target_feature] = -np.inf

        # Models
        self.model = Model(
            "Global",
            pbounds,
            random_seed=random_seed,
            **self.constraints,
        )

        # Logging
        self.decimals = decimals
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def initialize_queue(self):
        mini_batch = self.model.generate_random_candidates(
            self.n_init * self.batch_size
        )
        self.queue.extend([candidate for candidate in mini_batch])

    def recommend(self):
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
        if self.train_x is None or self.train_y is None:
            return
        self.model.update(self.train_x, self.train_y)

    def format_sample(self, sample: Dict[str, float]) -> Dict[str, str]:
        """
        Format the sample for logging.

        Args:
            sample (Dict[str, float]): The sample to format.

        Returns:
            Dict[str, str]: The formatted sample.
        """
        return {
            k: (f"{v:.{self.decimals}f}" if isinstance(v, (int, float)) else v)
            for k, v in sample.items()
        }

    def update_best_sample(self, x_batch: torch.Tensor, y_batch: torch.Tensor):
        """
        Update the best sample based on the current batch.
        If the maximum target value in the batch is greater than the current
        best target value, update the best sample.
        """
        # Update the best value and sample if necessary
        if self.best_sample[self.target_feature] < y_batch.max().item():
            self.best_sample[self.target_feature] = y_batch.max().item()
            point = x_batch[y_batch.argmax()]
            self.best_sample.update(
                {k: v for k, v in zip(self.pbounds.keys(), point.tolist())}
            )
            logging.info(f"Best sample: {self.format_sample(self.best_sample)}")

    def probe(self, x_batch: List[Tuple[torch.Tensor, str]]):
        y_batch = self.experiment(x_batch)
        self.train_x = (
            torch.cat([self.train_x, x_batch], dim=0)
            if self.train_x is not None
            else x_batch
        )
        self.train_y = (
            torch.cat([self.train_y, y_batch], dim=0)
            if self.train_y is not None
            else y_batch
        )
        self.train_iteration = (
            torch.cat(
                [
                    self.train_iteration,
                    torch.full((x_batch.shape[0],), self.train_iteration[-1] + 1),
                ]
            )
            if self.train_iteration is not None
            else torch.full((x_batch.shape[0],), 1)
        )

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
        Maximize the function.

        Args:
            n_init (int, optional): The number of initializations.
            budget (int): The number of iterations for the optimization
                process including the number of initial batches.
            batch (int, optional): The number of samples to evaluate at each
                iteration. Defaults to 1.
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
        Save the data to a CSV file.

        Args:
            filepath (str): The path of the file to save the data to.
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
