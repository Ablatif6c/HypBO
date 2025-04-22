"""
This script implements the HypBO (Hypothesis-based Bayesian Optimization)
algorithm. HypBO is a Bayesian optimization algorithm that incorporates
user-defined hypotheses to guide the optimization process.

The HypBO class provides an interface to initialize and run the optimization
process. It takes a function to be optimized, a set of parameter bounds, and
other optional parameters. The optimization process can be parallelized using
multiple processes.

The Hypothesis class (encapsulated inside the Model instances) represents a
user-defined hypothesis. It provides methods to apply the hypothesis to input
samples and convert it to a string representation.
"""

import concurrent.futures
import uuid
import torch
import warnings
from typing import Callable, Tuple, List, Optional, Dict
from collections import deque
import numpy as np
from .model import Model
import logging
from enum import Enum
import pandas as pd

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Define Torch device and data type settings based on availability of cuda
tkwargs = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float32,
}


class Level(Enum):
    """
    Enum for representing optimization levels.
    """

    LOCAL = 0
    GLOBAL = 1

    def __str__(self):
        return self.name.capitalize()


class HypBO:
    """
    HypBO implements the Hypothesis-based Bayesian Optimization algorithm.

    Attributes:
        experiment (Callable): The function to be optimized.
        pbounds (Dict[str, Tuple[float, float, float]]): Parameter bounds.
        target_feature (str): Name of the target feature.
        seed (int): Random seed.
        constraints (dict): Constraints retrieved from the experiment.
        queue (deque): Queue holding candidate samples.
        train_x (torch.Tensor): Training inputs.
        train_y (torch.Tensor): Training outputs.
        train_iteration (torch.Tensor): Training iteration numbers.
        train_model_ids (List[uuid.UUID]): List of model IDs used in training.
        best_sample (dict): Best sample details.
        models (List[Model]): List of hypothesis models.
        global_model_id (uuid.UUID): ID of the global model.
        gamma (float): Scaling factor for dynamic threshold.
        failures (int): Consecutive failure count.
        GLOBAL_LIMIT (int): Failure limit for global model optimization.
        LOCAL_LIMIT (int): Failure limit for local model optimization.
        decimals (int): Number of decimals for logging numbers.
    """

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
        self.constraints = experiment.get_all_constraints()  # Retrieve constraints
        self.pbounds = pbounds
        self.target_feature = target_feature
        self.seed = random_seed
        torch.manual_seed(self.seed)
        if torch.device(tkwargs["device"]).type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        # Data storage initialization
        self.queue = deque()
        self.train_x = torch.tensor([], **tkwargs)
        self.train_y = torch.tensor([], **tkwargs)
        self.train_iteration = torch.tensor(
            [], dtype=torch.long, device=tkwargs["device"]
        )
        self.train_model_ids: List[uuid.UUID] = []
        self.best_sample = {p: None for p in pbounds.keys()}
        self.best_sample[self.target_feature] = -np.inf

        # Initialize models list with a default global model
        self.models = [
            Model(
                "Global",
                pbounds,
                **self.constraints,
            ),
        ]
        self.global_model_id = self.models[0].id

        # Optimizer parameters
        self.gamma = gamma
        self.failures = 0
        self.GLOBAL_LIMIT = global_failure_limit
        self.LOCAL_LIMIT = local_failure_limit

        # Logging configuration
        self.decimals = decimals
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    @property
    def has_hypotheses(self) -> bool:
        """
        Checks if there are hypotheses provided.

        Returns:
            bool: True if at least one hypothesis is present, else False.
        """
        return len(self.models) > 1

    def add_hypothesis(self, name: str, pbounds: Dict[str, Tuple[float, float, float]]):
        """
        Adds a new hypothesis as a local model.

        Args:
            name (str): The name of the hypothesis.
            pbounds (Dict[str, Tuple[float, float, float]]): Parameter bounds for the hypothesis.
        """
        hypothesis = Model(name, pbounds, **self.constraints)
        self.models = [hypothesis] + self.models  # Prepend hypothesis to models list

    def split_n_init(self):
        """
        Splits the initialization budget across models.

        Returns:
            Dict[uuid.UUID, int]: Mapping from model ID to number of candidates.
        """
        n_candidates = self.n_init * self.batch_size
        total_models = len(self.models)
        allocation = {}
        if total_models > n_candidates:
            total_models = n_candidates
            # Allocate one candidate for each local model while ensuring global model is included
            local_models = [m for m in self.models if m.id != self.global_model_id][
                : (total_models - 1)
            ]
            allocation = {model.id: 1 for model in local_models}
            allocation[self.global_model_id] = 1
        else:
            base = n_candidates // total_models
            allocation = {model.id: base for model in self.models}
            remaining = n_candidates - sum(allocation.values())
            # Assign any extra candidates to the global model
            allocation[self.global_model_id] += remaining
        return allocation

    def _generate_mini_batch(self, model, n_candidates):
        """
        Generates a mini batch of random candidate samples from a model.

        Args:
            model (Model): The model to generate candidates from.
            n_candidates (int): Number of candidate samples to generate.

        Returns:
            Tuple[uuid.UUID, List]: Model id and its list of candidates.
        """
        return model.id, model.generate_random_candidates(n_candidates)

    def initialize_queue(self):
        """
        Initializes the candidate queue by generating mini batches
        from each model based on the allocated number of candidates.
        """
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
        # Extend queue with generated candidates
        for future in futures:
            model_id, mini_batch = future.result()
            self.queue.extend([(candidate, model_id) for candidate in mini_batch])

    def recommend(self):
        """
        Recommends a batch of candidate samples for evaluation.

        Returns:
            List[Tuple[torch.Tensor, str]]: Candidate samples with model IDs.
        """
        batch = None
        try:
            # Try getting a batch from the precomputed queue
            batch = [self.queue.popleft() for _ in range(self.batch_size)]
        except IndexError:
            # If queue is empty, generate recommendations using HypBO logic
            batch = self._hypbo_recommend()
        return batch

    def _hypbo_recommend(self):
        """
        Generates a recommendation according to the current optimization level.

        Returns:
            List[Tuple[torch.Tensor, str]]: Candidate samples with model IDs.
        """
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
        """
        Obtains recommendations from all local models (non-global).

        Returns:
            List[Tuple[torch.Tensor, str]]: Top candidate samples from local models.
        """
        candidates = []

        def get_model_recommendation(model):
            # Skip global model when gathering local recommendations
            if model.id == self.global_model_id:
                return []
            model_batch, acq_values = model.recommend(
                self.batch_size,
                best_f=self.train_y.max().item(),
            )
            acq_values = acq_values.view(-1, 1)  # Ensure acquisition values is 2D
            return [
                (candidate, model.id, acq_val.item())
                for candidate, acq_val in zip(model_batch, acq_values)
            ]

        # Run recommendations concurrently for all models
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.models)
        ) as executor:
            futures = [
                executor.submit(get_model_recommendation, model)
                for model in self.models
            ]
            for future in concurrent.futures.as_completed(futures):
                candidates.extend(future.result())

        # Sort candidates by acquisition values in descending order
        candidates.sort(key=lambda x: x[2], reverse=True)
        top_candidates = candidates[: self.batch_size]
        top_candidates = [(c, model_id) for c, model_id, _ in top_candidates]
        return top_candidates

    def get_global_recommendation(self):
        """
        Obtains recommendations from the global model.

        Returns:
            List[Tuple[torch.Tensor, str]]: Candidate samples from the global model.
        """
        global_model = self.get_model(self.global_model_id)
        candidates, _ = global_model.recommend(
            self.batch_size,
            best_f=self.train_y.max().item(),
        )
        return [(candidate, self.global_model_id) for candidate in candidates]

    def get_model(self, id):
        """
        Retrieves a model given its unique identifier.

        Args:
            id (uuid.UUID): The model's identifier.

        Returns:
            Model: The matching model.
        """
        model = [model for model in self.models if model.id == id][0]
        return model

    def get_threshold(self):
        """
        Compute a dynamic threshold based on the current best target and gamma.

        If the current best target is non-negative, the threshold is scaled up;
        if it's negative, the threshold is scaled down.

        Returns:
            float: The computed threshold.
        """
        target = self.best_sample[self.target_feature]
        scale = 1 + self.gamma if target >= 0 else 1 - self.gamma
        return target * scale

    def get_next_level(self):
        """
        Determine the next optimization level based on consecutive failures.

        Returns:
            Level: The selected optimization level (LOCAL or GLOBAL).

        Raises:
            ValueError: If training information is insufficient.
        """
        if not self.train_model_ids:
            raise ValueError(
                "No training levels set; cannot determine next optimization level."
            )

        # Start with local optimization after initialization.
        if len(self.train_model_ids) == self.n_init * self.batch_size:
            return Level.LOCAL

        # Determine failure limit based on the current optimization level.
        current_level = self.models_to_levels([self.train_model_ids[-1]])[0]
        limit = self.GLOBAL_LIMIT if current_level == Level.GLOBAL else self.LOCAL_LIMIT

        # Continue with current level if failure count is below limit.
        if self.failures < limit:
            return current_level

        # Reset failures and toggle the optimization level.
        self.failures = 0
        return Level.LOCAL if current_level == Level.GLOBAL else Level.GLOBAL

    def get_model_training_data(
        self, model_id: uuid.UUID
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filters training data corresponding to a specific model.

        Args:
            model_id (uuid.UUID): The identifier of the model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The filtered inputs and outputs.
        """
        model = self.get_model(model_id)
        train_x, train_y = model.filter_data(self.train_x, self.train_y)
        return train_x, train_y

    def models_to_levels(self, model_ids: List[uuid.UUID]) -> List[Level]:
        """
        Converts model IDs to their corresponding optimization levels.

        Args:
            model_ids (List[uuid.UUID]): List of model identifiers.

        Returns:
            List[Level]: List of Levels (GLOBAL or LOCAL) for each model.
        """
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
        """
        Updates all models using the accumulated training data.
        Global model is updated first; local models then are updated using
        filtered data. In case local data is insufficient, the global model's
        data is used.
        """
        if self.train_x.numel() == 0 or self.train_y.numel() == 0:
            return

        # Update global model
        global_model = self.get_model(self.global_model_id)
        global_model.update(self.train_x, self.train_y)

        # Define update procedure for local models
        def update_model(model):
            train_x, train_y = self.get_model_training_data(model.id)
            if train_x.shape[0] > 0 and train_y.shape[0] > 0:
                model.update(train_x, train_y)
            else:
                # Fallback to global model data if local data is empty
                model.gp = global_model.gp
                model.train_x = global_model.train_x
                model.train_y = global_model.train_y

        # Update local models concurrently
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
        """
        Updates the consecutive failure count based on evaluated batch results.

        Args:
            y_batch (torch.Tensor): Batch of target outputs.
        """
        threshold = self.get_threshold()
        if y_batch.max().item() <= threshold:
            self.failures += 1
        else:
            self.failures = 0

    def format_sample(self, sample: Dict[str, float]) -> Dict[str, str]:
        """
        Format the sample for logging purposes.

        Args:
            sample (Dict[str, float]): The sample to format.

        Returns:
            Dict[str, str]: The formatted sample with rounded values.
        """
        return {
            k: (f"{v:.{self.decimals}f}" if isinstance(v, (int, float)) else v)
            for k, v in sample.items()
        }

    def update_best_sample(self, x_batch: torch.Tensor, y_batch: torch.Tensor):
        """
        Updates the recorded best sample based on the current batch.

        If the maximum target value in the batch is greater than the current
        best target value, update the best sample and log the result.

        Args:
            x_batch (torch.Tensor): Batch of candidate inputs.
            y_batch (torch.Tensor): Batch of evaluated outputs.
        """
        if self.best_sample[self.target_feature] < y_batch.max().item():
            self.best_sample[self.target_feature] = y_batch.max().item()
            point = x_batch[y_batch.argmax()]
            # Update best sample with parameter values
            self.best_sample.update(
                {k: v for k, v in zip(self.pbounds.keys(), point.tolist())}
            )
            logging.info(f"Best sample: {self.format_sample(self.best_sample)}")

    def probe(self, batch: List[Tuple[torch.Tensor, str]]):
        """
        Evaluates a batch of candidate inputs and updates training data.

        Args:
            batch (List[Tuple[torch.Tensor, str]]): Candidate inputs paired with model IDs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The evaluated inputs and outputs.
        """
        # Stack candidate samples into a tensor and move to appropriate device and dtype
        x_batch = torch.stack(
            [candidate for candidate, _ in batch],
            dim=0,
        ).to(**tkwargs)
        y_batch = self.experiment(x_batch)  # Evaluate experiment function

        # Record which models produced the samples
        model_ids = [model_id for _, model_id in batch]
        self.train_model_ids.extend(model_ids)

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

        # Log batch data using pandas DataFrame for readability
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
        Maximize the target function via HypBO optimization.

        Args:
            n_init (int): Number of initialization batches.
            budget (int): Total number of evaluations (iterations) allowed.
            batch_size (int): Number of candidate evaluations per iteration.

        Raises:
            ValueError: If initialization parameters are invalid.
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
        # Continue until budget is exhausted or queue is empty
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
        Save training data and metadata to a CSV file.

        Args:
            filepath (str): Target file path for saving CSV data.

        Raises:
            ValueError: If no training data is available.
        """
        if self.train_x.numel() == 0 or self.train_y.numel() == 0:
            raise ValueError("No training data available to save.")

        # Format training data into a dictionary format
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
