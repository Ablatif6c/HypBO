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


class HypBO:
    def __init__(
            self,
            func,
            feature_names: List,
            model,
            model_kwargs: dict,
            hypotheses: List[Hypothesis] = [],
            seed: int = 0,
            n_processes: int = 1,
            discretization: bool = True,
            global_failire_limit: int = 5,
            local_failure_limit: int = 2):
        # Model parameters
        self.model = model
        self.model_kwargs = model_kwargs

        # HBO
        self.process_count = n_processes
        self.func = func
        self.discretization = discretization
        self.feature_names = feature_names
        self.seed = seed
        self.hypotheses = hypotheses
        self.X = []
        self.y = []

        # Tracking
        self.curt_best_value = -np.inf
        self.curt_best_sample = None
        self.optimisation_levels = []

        # Optimiser parameters
        self.failures = 0
        self.GLOBAL_LIMIT = global_failire_limit
        self.LOCAL_LIMIT = local_failure_limit
        self.GLOBAL_OPTIMISATION = 0
        self.LOCAL_OPTIMISATION = 1
        self.curt_optimisation_level = self.GLOBAL_OPTIMISATION

    def save_data(
            self,
            func_name: str,
            scenario_name: str,
            seed: int = 0,
            folder_path: str = os.path.join("data"),
    ):
        # Create the folders that constitute the path.
        path = os.path.join(
            folder_path,
            func_name,
            scenario_name
        )
        folders = path.split(os.sep)
        for i in range(len(folders)):
            _path = os.path.join(*folders[:i+1])
            # Create the folder if it does not exist
            if not os.path.exists(_path):
                os.makedirs(_path)

        file_path = os.path.join(
            path,
            f"dataset_{func_name}_{scenario_name}_{seed}.csv",
        )
        df = pd.concat(
            [
                pd.DataFrame(self.X, columns=self.feature_names),
                pd.DataFrame(self.y, columns=['Target']),
                pd.DataFrame(self.optimisation_levels, columns=['Hypothesis'])
            ],
            axis=1,)
        df.to_csv(file_path)

    def _check_samples_types(self, samples):
        # Turn into arrays
        if isinstance(samples[0], dict):
            # TODO order samples
            samples = [
                np.array([v for v in sample.values()]) for sample in samples
            ]
            samples = np.array(samples)
        return samples

    def _get_local_model(self, hypothesis):
        # Get the bounds
        lb, ub = hypothesis.sol_bounds
        kwargs = None
        hypothesis_model = None

        kwargs = deepcopy(self.model_kwargs)
        constraints = kwargs.pop("constraints", [])
        constraints.extend(hypothesis.stringify(ge=True))
        pbounds = kwargs.pop("pbounds")
        if self.discretization is True:
            for j, feature in enumerate(self.feature_names):
                step = pbounds[feature][-1]
                pbounds[feature] = (lb[j], ub[j], step)
        else:
            for j, feature in enumerate(self.feature_names):
                pbounds[feature] = (lb[j], ub[j])

        # Create the optimizer for the hypothesis
        hypothesis_model = self.model(
            pbounds=pbounds,
            constraints=constraints,
            **kwargs,
        )
        return hypothesis_model

    def _initialize_hypotheses(
            self,
            batch: int,
            hypotheses: List):
        all_X = []
        for hypothesis in hypotheses:
            hypothesis_model = self._get_local_model(hypothesis=hypothesis)

            X_init_hyp = hypothesis_model.initialize(
                n_init=20000,
                batch=batch,
            )
            # Only keep the samples that satisfy the hypothesis
            X = []
            for i in range(len(X_init_hyp)):
                get_out_first = False
                for j in range(batch):
                    sample = X_init_hyp[i][j]
                    if hypothesis.apply(input=sample):
                        X.append((sample, hypothesis.name))
                        if len(X) == batch:
                            get_out_first = True
                            break
                if get_out_first:
                    all_X.extend(X)
                    break
            if len(X) == 0:
                raise ValueError("No data for hypothesis")
        return [all_X]

    def initialize_data(
            self,
            n_init: int,
            batch: int,
    ):
        # Hypothesiss
        if self.has_hypotheses():
            X_init_hyp = []
            with Pool(processes=self.process_count) as pool:
                hyp_chunks = np.array_split(
                    self.hypotheses, self.process_count)
                # Make sure all hyp_chunks are not empty
                hyp_chunks = [
                    chunk for chunk in hyp_chunks if len(chunk) > 0]
                args = [(
                        batch,              # batch
                        hyp_chunks[i],    # conjs
                        )
                        for i in range(len(hyp_chunks))]
                results = pool.starmap(
                    self._initialize_hypotheses,
                    args,)
                results = [r for r in results if r is not None]
                for r in results:
                    for _r in r:
                        X_init_hyp.extend(_r)
                X_init_hyp = np.array(X_init_hyp)
            # Evaluate init samples
            for sample, conj_name in X_init_hyp:
                self.collect_samples(
                    sample=sample,
                    optimisation_level=conj_name,
                )

        # Global
        n_init_hyp = len(self.X)
        n_init = n_init - n_init_hyp
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
        return len(self.hypotheses) > 0

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

    def get_hypothesis_training_data(self, hypothesis):
        hypothesis_X = []
        hypothesis_y = []
        for x, y in zip(self.X, self.y):
            if hypothesis.apply(x) is True:
                hypothesis_X.append(x)
                hypothesis_y.append(y)
        if len(hypothesis_X) == 0:
            raise ValueError(f"Hypothesis {hypothesis.name} has 0 data!")
        hypothesis_X = np.array(hypothesis_X)
        hypothesis_y = np.array(hypothesis_y)
        return (hypothesis_X, hypothesis_y)

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

        for i, hyp in enumerate(conjs):
            # Get the local model
            model_hyp = self._get_local_model(hypothesis=hyp)

            # Populate its dataset
            hypo_x, hypo_y = self.get_hypothesis_training_data(hypothesis=hyp)
            for x, y in zip(hypo_x, hypo_y):
                model_hyp.register(
                    x,
                    y,
                )

            # Sampling
            proposed_conj_samples = model_hyp.suggest(
                conj_num_samples,
                multiprocessing=self.process_count,
                seed=seed,
            )
            proposed_conj_samples = self._check_samples_types(
                proposed_conj_samples)

            _samples = []
            for conj_sample in proposed_conj_samples:
                if hyp.apply(conj_sample):
                    _samples.append(conj_sample)
            proposed_conj_samples = np.array(_samples)

            # Compute the samples EI and add it as a column
            eis = expected_improvement(
                data=(hypo_x, hypo_y),
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
    ):
        self.initialize_data(n_init, batch)

        scenario_name = ""
        if self.has_hypotheses():
            scenario_name = '_'.join([c.name for c in self.hypotheses])
        else:
            scenario_name = "No hypothesis"
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
                hyp_num_samples = int(
                    5 * np.ceil(batch / len(self.hypotheses)))

                for i, hypothesis in enumerate(self.hypotheses):
                    # Get the local model
                    print(
                        f"\t\tSampling hypothesis {hypothesis.name} - {i+1}/{len(self.hypotheses)}")
                    hypothesis_model = self._get_local_model(
                        hypothesis=hypothesis)

                    # Populate its dataset
                    hyp_X, hyp_y = self.get_hypothesis_training_data(
                        hypothesis=hypothesis)
                    for x, y in zip(hyp_X, hyp_y):
                        hypothesis_model.register(
                            x,
                            y,
                        )

                    # Sampling
                    proposed_hyp_samples = hypothesis_model.suggest(
                        hyp_num_samples,
                        multiprocessing=self.process_count,
                        seed=idx,
                    )
                    proposed_hyp_samples = self._check_samples_types(
                        proposed_hyp_samples)

                    _samples = []
                    for hyp_sample in proposed_hyp_samples:
                        if hypothesis.apply(hyp_sample):
                            _samples.append(hyp_sample)
                    proposed_hyp_samples = np.array(_samples)

                    # Compute the samples EI and add it as a column
                    eis = expected_improvement(
                        data=(hyp_X, hyp_y),
                        X=proposed_hyp_samples,
                        use_ei=True,
                        mu_sample_opt=self.curt_best_value,
                        seed=self.seed,
                    )
                    eis = eis.flatten()
                    proposed_hyp_samples = [
                        [proposed_hyp_samples[j], eis[j], i]
                        for j in range(
                            len(proposed_hyp_samples),
                        )
                    ]
                    samples.extend(proposed_hyp_samples)

                # Sort the union of the conj samples by the descending EI
                samples = np.array(samples)
                samples = samples[samples[:, 1].argsort()][::-1]
                # keep the best ones
                samples = samples[:batch]
                # only keep the sample not the ei
                optimisation_levels = [
                    self.hypotheses[i].name for i in samples[:, 2]
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

        print(f"{scenario_name} optimisation has finished.")
