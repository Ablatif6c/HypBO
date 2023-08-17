import os
import warnings
from copy import deepcopy
from multiprocessing import Pool
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from Hypothesis import Hypothesis

warnings.filterwarnings("ignore")


def expected_improvement(
    data,
    X: np.ndarray,
    xi: float = 0.0001,
    use_ei: bool = True,
    mu_sample_opt=None,
    seed: int = 1,
):
    """Computes the EI at points X based on existing samples X_sample and
    Y_sample using a Gaussian process surrogate model.

    Args: X: Points at which EI shall be computed (m x d). X_sample:
    Sample locations (n x d).
    Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor
    fitted to samples.
    xi: Exploitation-exploration trade-off parameter.

    Returns: Expected improvements at points X."""
    X_sample, y_sample = data

    # Create GPR
    dim = X_sample.shape[1]
    length_scale = [1] * dim
    kernel = Matern(length_scale=length_scale,
                    length_scale_bounds=(1e-01, 1e4),
                    nu=2.5) * ConstantKernel(1.0, (0.5, 5))
    gpr = GaussianProcessRegressor(
                                kernel=kernel,
                                alpha=1e-6,
                                normalize_y=False,
                                n_restarts_optimizer=10 * dim,
                                random_state=seed,
                                )
    X_data, y_data = data
    gpr.fit(X=X_data, y=y_data)
    mu, sigma = gpr.predict(
        X,
        return_std=True,
        )

    if not use_ei:
        return mu
    else:
        # calculate EI
        if mu_sample_opt is None:
            mu_sample = gpr.predict(X_sample)
            mu_sample_opt = np.max(mu_sample)
        sigma = sigma.reshape(-1, 1)
        with np.errstate(divide="warn"):
            imp = mu - mu_sample_opt - xi
            imp = imp.reshape((-1, 1))
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei


class HBO:
    def __init__(
        self,
        func,
        feature_names: List,
        model,
        model_kwargs: dict,
        conjs: List[Hypothesis] = [],
        seed: int = 0,
        n_processes: int = 1,
        discretization=True,
            ):
        # Model parameters
        self.model = model
        self.model_kwargs = model_kwargs

        # HBO
        self.process_count = n_processes
        self.func = func
        self.discretization = discretization
        self.feature_names = feature_names
        self.seed = seed
        self.conjs = conjs
        self.X = []
        self.y = []

        # Tracking
        self.curt_best_value = -np.inf
        self.curt_best_sample = None
        self.optimisation_levels = []

        # Optimiser parameters
        self.failures = 0
        self.GLOBAL_LIMIT = 5
        self.LOCAL_LIMIT = 2
        self.GLOBAL_OPTIMISATION = 0
        self.LOCAL_OPTIMISATION = 1
        self.curt_optimisation_level = self.GLOBAL_OPTIMISATION

    def save_data(
            self,
            func_name: str,
            scenario_name: str,
            seed: int = 0,
            folder: str = os.path.join("data"),
    ):
        # Create the data folder if it does not exist
        if not os.path.exists(folder):
            os.makedirs(folder)
        # Create the function subfolder if it does not exist
        path = os.path.join(
            folder,
            func_name,
        )
        if not os.path.exists(path):
            os.makedirs(path)
        # Create the scenario subfolder if it does not exist
        path = os.path.join(
            folder,
            func_name,
            scenario_name,
        )
        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(
            folder,
            func_name,
            scenario_name,
            f"dataset_{func_name}_{scenario_name}_{seed}.csv",
        )
        df = pd.concat(
            [
                pd.DataFrame(self.X, columns=self.feature_names),
                pd.DataFrame(self.y, columns=['Target']),
                pd.DataFrame(self.optimisation_levels, columns=['Hypothesis'])
                ],
            axis=1,)
        df.to_csv(path)

    def _check_samples_types(self, samples):
        # Turn into arrays
        if isinstance(samples[0], dict):
            # TODO order samples
            samples = [
                np.array([v for v in sample.values()]) for sample in samples
                ]
            samples = np.array(samples)
        return samples

    def _get_local_model(self, conj):
        # Get the bounds
        lb, ub = conj.sol_bounds
        kwargs = None
        model_conj = None

        kwargs = deepcopy(self.model_kwargs)
        constraints = kwargs.pop("constraints", [])
        constraints.extend(conj.stringify(ge=True))
        pbounds = kwargs.pop("pbounds")
        if self.discretization is True:
            for j, feature in enumerate(self.feature_names):
                step = pbounds[feature][-1]
                pbounds[feature] = (lb[j], ub[j], step)
        else:
            for j, feature in enumerate(self.feature_names):
                pbounds[feature] = (lb[j], ub[j])

        # Create the optimizer for the hypothesis
        model_conj = self.model(
            pbounds=pbounds,
            constraints=constraints,
            **kwargs,
                )
        return model_conj

    def _initialize_conjs(
            self,
            batch: int,
            conjs: List):
        all_X = []
        for conj in conjs:
            model_conj = self._get_local_model(conj=conj)

            X_init_conj = model_conj.initialize(
                            n_init=20000,
                            batch=batch,
                            )
            # Only keep the samples that satisfy the conj
            X = []
            for i in range(len(X_init_conj)):
                get_out_first = False
                for j in range(batch):
                    sample = X_init_conj[i][j]
                    if conj.apply(input=sample):
                        X.append((sample, conj.name))
                        if len(X) == batch:
                            get_out_first = True
                            break
                if get_out_first:
                    all_X.extend(X)
                    break
            if len(X) == 0:
                raise ValueError("No data for conj")
        return [all_X]

    def initialize_data(
            self,
            n_init: int,
            batch: int,
            ):
        # Hypothesiss
        if self.has_hypotheses():
            X_init_conj = []
            with Pool(processes=self.process_count) as pool:
                conjs_chunks = np.array_split(self.conjs, self.process_count)
                # Make sure all conjs_chunks are not empty
                conjs_chunks = [
                    chunk for chunk in conjs_chunks if len(chunk) > 0]
                args = [(
                        batch,              # batch
                        conjs_chunks[i],    # conjs
                        )
                        for i in range(len(conjs_chunks))]
                results = pool.starmap(
                            self._initialize_conjs,
                            args,)
                results = [r for r in results if r is not None]
                for r in results:
                    for _r in r:
                        X_init_conj.extend(_r)
                X_init_conj = np.array(X_init_conj)
            # Evaluate init samples
            for sample, conj_name in X_init_conj:
                self.collect_samples(
                    sample=sample,
                    optimisation_level=conj_name,
                )

        # Global
        n_init_conj = len(self.X)
        n_init = n_init - n_init_conj
        if n_init <= 0:
            n_init = 1
        model_global = self.model(**self.model_kwargs)
        X_init_global = model_global.initialize(
                        n_init,
                        batch,
                        )

        # Evaluate init samples
        for i in range(len(X_init_global)):
            for j in range(batch):
                self.collect_samples(
                    sample=X_init_global[i][j],
                    optimisation_level="Global",
                )

    def has_hypotheses(self):
        return len(self.conjs) > 0

    def get_optimisation_level(self, iteration: int) -> int:
        """Computes the level of optimisation (global or local) to perform\
            next.

        Returns:
            int: 0 for global optimisation, 1 for a local one.
        """

        # Global optimisation if there are no hypothesiss.
        if not self.has_hypotheses():
            self.curt_optimisation_level = self.GLOBAL_OPTIMISATION
            return self.curt_optimisation_level

        # The optimisation starts with a local optimisation.
        if iteration == 0:
            self.curt_optimisation_level = self.LOCAL_OPTIMISATION
            return self.curt_optimisation_level

        LIMIT = (
            self.GLOBAL_LIMIT
            if self.curt_optimisation_level == self.GLOBAL_OPTIMISATION
            else self.LOCAL_LIMIT
        )
        # Keep performing current level while we don't have {limit}
        # consecutive failures.
        if 0 <= self.failures < LIMIT:
            return self.curt_optimisation_level

        # Switch optimisation if we have {limit} consecutive failures.
        else:
            # Re-initialise failure count.
            self.failures = 0

            # Switch optimisation level.
            if self.curt_optimisation_level == self.GLOBAL_OPTIMISATION:
                self.curt_optimisation_level = self.LOCAL_OPTIMISATION
            else:
                self.curt_optimisation_level = self.GLOBAL_OPTIMISATION

            return self.curt_optimisation_level

    def get_conj_training_data(self, conj):
        X_conj = []
        y_conj = []
        for x, y in zip(self.X, self.y):
            if conj.apply(x) is True:
                X_conj.append(x)
                y_conj.append(y)
        if len(X_conj) == 0:
            raise ValueError(f"Hypothesis {conj.name} has 0 data!")
        X_conj = np.array(X_conj)
        y_conj = np.array(y_conj)
        return (X_conj, y_conj)

    def collect_samples(
        self,
        sample: np.ndarray,
        optimisation_level: str = "Global",
    ) -> float:
        """Evaluate the sample, add it to the samples and return its value.

        Args:
            sample (list): the sample to evaluate.
            is_conj_path (bool, optional): is True if the sample comes from a
            hypothesis. Defaults to False.

        Returns:
            float: evaluation of the sample
        """
        value = self.func(sample)

        if value > self.curt_best_value:
            self.curt_best_value = value
            self.curt_best_sample = sample

        if len(self.X) == 0:
            self.X = np.array([sample])
        else:
            self.X = np.append(self.X, [sample], axis=0)
        self.y = np.append(self.y, value)
        self.optimisation_levels = np.append(
            self.optimisation_levels,
            optimisation_level,
                                            )
        return value

    def sample_hypothesiss(
            self,
            batch: int,
            conjs: List,
            seed: int,
            ):
        samples = []
        # Compute the number of samples we ask from each hypothesis
        conj_num_samples = int(5 * np.ceil(batch / len(conjs)))

        for i, conj in enumerate(conjs):
            # Get the local model
            model_conj = self._get_local_model(conj=conj)

            # Populate its dataset
            X_conj, y_conj = self.get_conj_training_data(conj=conj)
            for x, y in zip(X_conj, y_conj):
                model_conj.register(
                    x,
                    y,
                    )

            # Sampling
            proposed_conj_samples = model_conj.suggest(
                conj_num_samples,
                multiprocessing=self.process_count,
                seed=seed,
                )
            proposed_conj_samples = self._check_samples_types(
                proposed_conj_samples)

            _samples = []
            for conj_sample in proposed_conj_samples:
                if conj.apply(conj_sample):
                    _samples.append(conj_sample)
            proposed_conj_samples = np.array(_samples)

            # Compute the samples EI and add it as a column
            eis = expected_improvement(
                data=(X_conj, y_conj),
                X=proposed_conj_samples,
                use_ei=True,
                mu_sample_opt=self.curt_best_value,
                seed=self.seed,
            )
            eis = eis.flatten()
            proposed_conj_samples = [
                [proposed_conj_samples[j], eis[j], i]
                for j in range(
                    len(proposed_conj_samples),
                )
            ]
            samples.extend(proposed_conj_samples)

        return samples

    def search(
        self,
        budget: int,
        n_init: int = 5,
        batch: int = 1,
        target: Optional[float] = None,
            ):
        self.initialize_data(n_init, batch)

        scenario_name = ""
        if self.has_hypotheses():
            scenario_name = '_'.join([c.name for c in self.conjs])
        else:
            scenario_name = "No conj"
        print(f"'{scenario_name}' optimisation has started...")
        # For each iteration
        for idx in range(budget):
            print(f"{scenario_name} Iter {idx+1}/{budget}")
            # Choose between global and local optimisations.
            optimisation_level = self.get_optimisation_level(iteration=idx)
            level_log_message = "Global"
            if optimisation_level == self.LOCAL_OPTIMISATION:
                level_log_message = "Local"
            print(f"\tOptimisation level: {level_log_message}")
            optimisation_levels = []
            samples = []

            # Local optimisation
            if optimisation_level == self.LOCAL_OPTIMISATION:
                # Compute the number of samples we ask from each hypothesis
                conj_num_samples = int(5 * np.ceil(batch / len(self.conjs)))

                for i, conj in enumerate(self.conjs):
                    # Get the local model
                    print(
                        f"\t\tSampling hypothesis {conj.name} - {i+1}/{len(self.conjs)}")
                    model_conj = self._get_local_model(conj=conj)

                    # Populate its dataset
                    X_conj, y_conj = self.get_conj_training_data(conj=conj)
                    for x, y in zip(X_conj, y_conj):
                        model_conj.register(
                            x,
                            y,
                            )

                    # Sampling
                    proposed_conj_samples = model_conj.suggest(
                        conj_num_samples,
                        multiprocessing=self.process_count,
                        seed=idx,
                        )
                    proposed_conj_samples = self._check_samples_types(
                        proposed_conj_samples)

                    _samples = []
                    for conj_sample in proposed_conj_samples:
                        if conj.apply(conj_sample):
                            _samples.append(conj_sample)
                    proposed_conj_samples = np.array(_samples)

                    # Compute the samples EI and add it as a column
                    eis = expected_improvement(
                        data=(X_conj, y_conj),
                        X=proposed_conj_samples,
                        use_ei=True,
                        mu_sample_opt=self.curt_best_value,
                        seed=self.seed,
                    )
                    eis = eis.flatten()
                    proposed_conj_samples = [
                        [proposed_conj_samples[j], eis[j], i]
                        for j in range(
                            len(proposed_conj_samples),
                        )
                    ]
                    samples.extend(proposed_conj_samples)

                # Sort the union of the conj samples by the descending EI
                samples = np.array(samples)
                samples = samples[samples[:, 1].argsort()][::-1]
                # keep the best ones
                samples = samples[:batch]
                # only keep the sample not the ei
                optimisation_levels = [
                    self.conjs[i].name for i in samples[:, 2]
                    ]
                print(f"\t\t\tLocal(s): {optimisation_levels}")
                samples = samples[:, 0].flatten()

            # Global optimisation
            else:
                # Create the optimizer for the hypothesis
                model_global = self.model(**self.model_kwargs)

                # Populate its dataset
                for x, y in zip(self.X, self.y):
                    model_global.register(
                        x,
                        y,
                        )

                # Sampling
                samples = model_global.suggest(
                    batch,
                    multiprocessing=self.process_count,
                    seed=idx,
                    )
                samples = self._check_samples_types(samples)
                optimisation_levels = ["Global"] * batch

            # Evaluation
            value = -np.inf
            for i in range(len(samples)):
                value = self.collect_samples(
                    sample=samples[i],
                    optimisation_level=optimisation_levels[i],
                )
                _sample_to_print = dict(zip(
                    self.feature_names,
                    np.round(samples[i], 2),
                    ))
                print(f"\tSample {_sample_to_print} \t Target: {value:.3f}")

                # Update the optimisation level success/failure
                # local
                curt_value = self.y[-1]
                if curt_value >= self.curt_best_value:
                    self.failures = 0
                else:
                    self.failures += 1

            print(f"\t Best target: {self.curt_best_value:.3f}")
            # Early stopping
            if target is not None and value >= target:
                return

        print(f"{scenario_name} optimisation has finished.")
