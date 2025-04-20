"""
This script implements the HypBO (Hypothesis-based Bayesian Optimization)
algorithm. HypBO is a Bayesian optimization algorithm that incorporates
user-defined hypotheses to guide the optimization process.

The HypBO class provides an interface to initialize and run the optimization
process. It takes a function to be optimized, a list of feature names, an
optimization model, and other optional parameters. The optimization process
can be parallelized using multiple processes.

The Hypothesis class represents a user-defined hypothesis. It provides
methods to apply the hypothesis to input samples and convert it to a
string representation.
"""

import concurrent.futures
import uuid
import torch
import warnings
from typing import Callable, Tuple, List, Optional
from collections import deque
import numpy as np
from .model import Model
from typing import Dict
import logging
from enum import Enum
import pandas as pd


warnings.filterwarnings("ignore")

tkwargs = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float32,
}


class Level(Enum):
    LOCAL = 0
    GLOBAL = 1

    def __str__(self):
        return self.name.capitalize()


class HypBO:
    def __init__(
        self,
        experiment: Callable,
        pbounds: Dict[str, Tuple[float, float, float]],
        target_feature: str = "target",
        random_seed: int = 0,
        global_failure_limit: int = 5,
        local_failure_limit: int = 2,
        gamma: float = 0.0,
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
        self.train_model_ids: List[uuid.UUID] = []
        self.best_sample = {p: None for p in pbounds.keys()}
        self.best_sample[self.target_feature] = -np.inf

        # Models
        self.models = [
            Model(
                "Global",
                pbounds,
                random_seed=random_seed,
                **self.constraints,
            ),
        ]
        self.global_model_id = self.models[0].id

        # Optimizer parameters
        self.gamma = gamma
        self.failures = 0
        self.GLOBAL_LIMIT = global_failure_limit
        self.LOCAL_LIMIT = local_failure_limit

        # Logging
        self.decimals = decimals
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    @property
    def has_hypotheses(self) -> bool:
        return len(self.models) > 1

    def add_hypothesis(self, name: str, pbounds: Dict[str, Tuple[float, float, float]]):
        hypothesis = Model(name, pbounds, **self.constraints)
        self.models = [hypothesis] + self.models

    def split_n_init(self):
        n_candidates = self.n_init * self.batch_size
        total_models = len(self.models)
        allocation = {}
        if total_models > n_candidates:
            total_models = n_candidates
            local_models = [m for m in self.models if m.id != self.global_model_id][
                : (total_models - 1)
            ]
            allocation = {model.id: 1 for model in local_models}
            allocation[self.global_model_id] = 1
        else:
            base = n_candidates // total_models
            allocation = {model.id: base for model in self.models}
            remaining = n_candidates - sum(allocation.values())
            # assign any extra candidates to the last model
            allocation[self.global_model_id] += remaining
        return allocation

    def _generate_mini_batch(self, model, n_candidates):
        return model.id, model.generate_random_candidates(n_candidates)

    def initialize_queue(self):
        allocation = self.split_n_init()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(allocation)
        ) as executor:
            futures = [
                executor.submit(
                    self._generate_mini_batch,
                    model,
                    allocation[model.id],
                )
                for model in self.models
                if model.id in allocation
            ]
        for future in futures:
            model_id, mini_batch = future.result()
            self.queue.extend([(candidate, model_id) for candidate in mini_batch])

    def recommend(self):
        batch = None
        try:
            batch = [self.queue.popleft() for _ in range(self.batch_size)]
        except IndexError:
            batch = self._hypbo_recommend()

        return batch

    def _hypbo_recommend(self):
        next_level = self.get_next_level()
        logging.info(f"Next level: {next_level}")
        if next_level == Level.LOCAL:
            batch = self.get_local_recommendation()
        elif next_level == Level.GLOBAL:
            batch = self.get_global_recommendation()
        else:
            raise ValueError("Invalid optimization level.")

        return batch

    def get_local_recommendation(self):
        candidates = []

        def get_model_recommendation(model):
            if model.id == self.global_model_id:
                return []

            model_batch, acq_values = model.recommend(
                self.batch_size,
                best_f=self.train_y.max().item(),
            )
            acq_values = acq_values.view(-1, 1)  # Ensure acq values is 2D
            return [
                (candidate, model.id, acq_val.item())
                for candidate, acq_val in zip(model_batch, acq_values)
            ]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.models)
        ) as executor:
            futures = [
                executor.submit(get_model_recommendation, model)
                for model in self.models
            ]
            for future in concurrent.futures.as_completed(futures):
                candidates.extend(future.result())

        candidates.sort(key=lambda x: x[2], reverse=True)
        top_candidates = candidates[: self.batch_size]
        top_candidates = [(c, model_id) for c, model_id, _ in top_candidates]
        return top_candidates

    def get_global_recommendation(self):
        global_model = self.get_model(self.global_model_id)
        candidates, _ = global_model.recommend(
            self.batch_size,
            best_f=self.train_y.max().item(),
        )
        return [(candidate, self.global_model_id) for candidate in candidates]

    def get_model(self, id):
        model = [model for model in self.models if model.id == id][0]
        return model

    def get_threshold(self):
        """
        Compute a dynamic threshold based on the current best target and gamma.

        If the current best target is non-negative, the threshold is scaled up;
        if it's negative, the threshold is scaled down.
        """
        target = self.best_sample[self.target_feature]
        scale = 1 + self.gamma if target >= 0 else 1 - self.gamma
        return target * scale

    def get_next_level(self):
        """
        Determine the next optimization level based on consecutive failures.

        Returns:
            Level.LOCAL if we've just finished initialization or if the current global
            optimization level has reached the failure limit and needs to switch to local,
            or vice versa.
        Raises:
            ValueError: If train_model_ids is empty or current_level is invalid.
        """
        if not self.train_model_ids:
            raise ValueError(
                "No training levels set; cannot determine next optimization level."
            )

        # Start with local optimization right after initialization.
        if len(self.train_model_ids) == self.n_init * self.batch_size:
            return Level.LOCAL

        # Determine failure limit based on the current optimization level.
        current_level = self.models_to_levels([self.train_model_ids[-1]])[0]
        limit = self.GLOBAL_LIMIT if current_level == Level.GLOBAL else self.LOCAL_LIMIT

        # If failure count is below the limit, keep the current level.
        if self.failures < limit:
            return current_level

        # Reset failures and toggle the optimization level.
        self.failures = 0
        return Level.LOCAL if current_level == Level.GLOBAL else Level.GLOBAL

    def get_model_training_data(
        self, model_id: uuid.UUID
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model = self.get_model(model_id)
        train_x, train_y = model.filter_data(self.train_x, self.train_y)
        return train_x, train_y

    def models_to_levels(self, model_ids: List[uuid.UUID]) -> List[Level]:
        levels = [
            (
                Level.GLOBAL
                if self.get_model(model_id).id == self.global_model_id
                else Level.LOCAL
            )
            for model_id in model_ids
        ]
        return levels

    def update_models(self):
        if self.train_x is None or self.train_y is None:
            return
        # Update the global model first
        global_model = self.get_model(self.global_model_id)
        global_model.update(self.train_x, self.train_y)

        # Update the local models
        def update_model(model):
            train_x, train_y = self.get_model_training_data(model.id)
            if train_x.shape[0] > 0 and train_y.shape[0] > 0:
                model.update(train_x, train_y)
            else:
                model.gp = global_model.gp
                model.train_x = global_model.train_x
                model.train_y = global_model.train_y

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.models)
        ) as executor:
            futures = [
                executor.submit(update_model, model)
                for model in self.models
                if model.id != self.global_model_id
            ]
            concurrent.futures.wait(futures)

    def update_failure_count(self, y_batch: torch.Tensor):
        threshold = self.get_threshold()
        if y_batch.max().item() <= threshold:
            self.failures += 1
        else:
            self.failures = 0

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

    def probe(self, batch: List[Tuple[torch.Tensor, str]]):
        x_batch = torch.stack(
            [candidate for candidate, _ in batch],
            dim=0,
        ).to(**tkwargs)
        y_batch = self.experiment(x_batch)
        model_ids = [model_id for _, model_id in batch]
        self.train_model_ids.extend(model_ids)
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
        data.update({"model_name": [self.get_model(m_id).name for m_id in model_ids]})
        data.update({"level": self.models_to_levels(model_ids)})
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
        if not self.has_hypotheses:
            raise ValueError("No hypotheses provided for optimization!")

        logging.info("Starting HypBO optimization...")
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
            if iteration > self.n_init:
                self.update_failure_count(y_batch)
            self.update_best_sample(x_batch, y_batch)
            self.update_models()
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
        data.update(
            {"model_name": [self.get_model(m_id).name for m_id in self.train_model_ids]}
        )
        data.update({"level": self.models_to_levels(self.train_model_ids)})
        data.update({"iteration": iterations})
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
