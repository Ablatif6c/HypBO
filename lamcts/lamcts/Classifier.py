# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy as cp
import logging
import os
from multiprocessing import Pool
from typing import List, Literal, Tuple

# import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.svm import SVC
from torch.quasirandom import SobolEngine

from .turbo.turbo.turbo_1 import Turbo1


class Classifier:
    def __init__(
        self,
        samples: List[np.ndarray],
        dim: int,
        kernel_type: Literal[
            "linear",
            "poly",
            "rbf",
            "sigmoid",
            "precomputed",
        ],
        gamma_type: Literal[
            "scale",
            "auto",
        ] = "auto",
    ):
        self.training_counter = 0
        assert dim >= 1
        assert isinstance(samples, list)
        self.dim = dim

        # create a gaussian process regressor
        kernel = None
        kernel_length_scale = np.array([1 for _ in range(self.dim)])
        noise = 0.1
        if noise:
            kernel = (
                Matern(
                    length_scale=kernel_length_scale,
                    nu=2.5,
                )
                + WhiteKernel()
            )
        else:
            kernel = Matern(
                length_scale=kernel_length_scale,
                nu=2.5,
            )
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
        )  # default to CPU
        self.kmean = KMeans(n_clusters=2)
        # learned boundary
        self.svm = SVC(
            kernel=kernel_type,
            gamma=gamma_type,
        )
        # data structures to store
        self.samples = []
        self.X = np.array([])
        self.fX = np.array([])

        # good region is labeled as zero
        # bad  region is labeled as one
        self.good_label_mean = -1
        self.bad_label_mean = -1

        self.update_samples(samples)

    def is_svm_splittable(self):
        plabel = self.learn_clusters()
        self.learn_boundary(plabel)
        svm_label = self.svm.predict(self.X)
        if len(np.unique(svm_label)) == 1:
            return False
        else:
            return True

    def get_max(self):
        return np.max(self.fX)

    # def plot_samples_and_boundary(self, func, name):
    #     assert func.dims == 2

    #     plabels = self.svm.predict(self.X)
    #     good_counts = len(self.X[np.where(plabels == 0)])
    #     bad_counts = len(self.X[np.where(plabels == 1)])
    #     good_mean = np.mean(self.fX[np.where(plabels == 0)])
    #     bad_mean = np.mean(self.fX[np.where(plabels == 1)])

    #     if np.isnan(good_mean) is False and np.isnan(bad_mean) is False:
    #         assert good_mean > bad_mean

    #     lb = func.lb
    #     ub = func.ub
    #     x = np.linspace(lb[0], ub[0], 100)
    #     y = np.linspace(lb[1], ub[1], 100)
    #     xv, yv = np.meshgrid(x, y)
    #     true_y = []
    #     for row in range(0, xv.shape[0]):
    #         for col in range(0, xv.shape[1]):
    #             x = xv[row][col]
    #             y = yv[row][col]
    #             true_y.append(func(np.array([x, y])))
    #     true_y = np.array(true_y)
    #     pred_labels = self.svm.predict(np.c_[xv.ravel(), yv.ravel()])
    #     pred_labels = pred_labels.reshape(xv.shape)

    #     fig, ax = plt.subplots()
    #     ax.contour(
    #         xv,
    #         yv,
    #         true_y.reshape(xv.shape),
    #         cmap=cm.coolwarm,
    #     )
    #     ax.contourf(
    #         xv,
    #         yv,
    #         pred_labels,
    #         alpha=0.4,
    #     )

    #     ax.scatter(
    #         self.X[np.where(plabels == 0), 0],
    #         self.X[np.where(plabels == 0), 1],
    #         marker="x",
    #         label=f"good-{str(np.round(good_mean, 2))}-{str(good_counts)}",
    #     )
    #     ax.scatter(
    #         self.X[np.where(plabels == 1), 0],
    #         self.X[np.where(plabels == 1), 1],
    #         marker="x",
    #         label="bad-" + str(np.round(bad_mean, 2)) + "-" + str(bad_counts),
    #     )
    #     ax.legend(loc="best")
    #     ax.set_xlabel("x1")
    #     ax.set_ylabel("x2")
    #     ax.set_xlim(left=-10, right=10)
    #     ax.set_ylim(bottom=-10, top=10)
    #     plt.savefig(name)
    #     plt.close()

    def get_mean(self):
        return np.mean(self.fX)

    def update_samples(self, latest_samples):
        assert isinstance(latest_samples, list)
        X = []
        fX = []
        for sample in latest_samples:
            X.append(sample[0])
            fX.append(sample[1])

        self.X = np.asarray(X, dtype=np.float32).reshape(-1, self.dim)
        self.fX = np.asarray(fX, dtype=np.float32).reshape(-1)
        self.samples = latest_samples

    def train_gpr(self, samples):
        X = []
        fX = []
        for sample in samples:
            X.append(sample[0])
            fX.append(sample[1])
        X = np.asarray(X).reshape(-1, self.dim)
        fX = np.asarray(fX).reshape(-1)

        # print("training GPR with ", len(X), " data X")
        self.gpr.fit(X, fX)

    ###########################
    # BO sampling with EI
    ###########################

    def expected_improvement(
        self,
        X,
        xi=0.0001,
        use_ei=True,
        mu_sample_opt=None,
    ):
        """Computes the EI at points X based on existing samples X_sample and
        Y_sample using a Gaussian process surrogate model.

        Args: X: Points at which EI shall be computed (m x d). X_sample:
        Sample locations (n x d).
        Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor
        fitted to samples.
        xi: Exploitation-exploration trade-off parameter.

        Returns: Expected improvements at points X."""
        X_sample = self.X
        # Y_sample = self.fX.reshape((-1, 1))

        gpr = self.gpr

        mu, sigma = gpr.predict(X, return_std=True)

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

    # def plot_boundary(self, X):
    #     if X.shape[1] > 2:
    #         return
    #     fig, ax = plt.subplots()
    #     ax.scatter(X[:, 0], X[:, 1], marker=".")
    #     ax.scatter(self.X[:, 0], self.X[:, 1], marker="x")
    #     ax.set_xlim(left=-10, right=10)
    #     ax.set_ylim(bottom=-10, top=10)
    #     plt.savefig("boundary.pdf")
    #     plt.close()

    def get_sample_ratio_in_region(
        self,
        cands,
        path,
    ):
        total = len(cands)
        for node, idx in path:
            boundary = node.classifier.svm
            if len(cands) == 0:
                return 0, np.array([])
            assert len(cands) > 0
            cands = cands[boundary.predict(cands) == idx]
            # node[1] store the direction to go
        ratio = len(cands) / total
        assert len(cands) <= total
        return ratio, cands

    def propose_rand_samples_probe(
        self,
        nums_samples,
        path,
        lb,
        ub,
    ):
        seed = 0  # np.random.randint(int(1e6))
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=seed)

        center = np.mean(self.X, axis=0)
        # check if the center located in the region
        ratio, tmp = self.get_sample_ratio_in_region(
            np.reshape(center, (1, len(center))),
            path,
        )
        if ratio == 0:
            print("==>center not in the region, using random samples")
            return self.propose_rand_samples(nums_samples, lb, ub)
        # it is possible that the selected region has no points,
        # so we need check here

        axes = len(center)

        final_L = []
        for axis in range(0, axes):
            L = np.zeros(center.shape)
            L[axis] = 0.01
            ratio = 1

            while ratio >= 0.9:
                L[axis] = L[axis] * 2
                if L[axis] >= (ub[axis] - lb[axis]):
                    break
                lb_ = np.clip(center - L / 2, lb, ub)
                ub_ = np.clip(center + L / 2, lb, ub)
                cands_ = (
                    sobol.draw(10000)
                    .to(
                        dtype=torch.float64,
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
                cands_ = (ub_ - lb_) * cands_ + lb_
                ratio, tmp = self.get_sample_ratio_in_region(
                    cands_,
                    path,
                )
            final_L.append(L[axis])

        final_L = np.array(final_L)
        lb_ = np.clip(center - final_L / 2, lb, ub)
        ub_ = np.clip(center + final_L / 2, lb, ub)
        print("center:", center)
        print("final lb:", lb_)
        print("final ub:", ub_)

        count = 0
        cands = np.array([])
        while len(cands) < 10000:
            count += 10000
            cands = (
                sobol.draw(count)
                .to(
                    dtype=torch.float64,
                )
                .cpu()
                .detach()
                .numpy()
            )

            cands = (ub_ - lb_) * cands + lb_
            ratio, cands = self.get_sample_ratio_in_region(cands, path)
            # samples_count = len(cands)

        return cands

    def generate_candidates_around_points(
            self,
            centers,
            path,
            lb,
            ub,
            previous_samples,
            seed,
            ):
        # Initialise sampler
        np.random.seed(seed)
        sobol = SobolEngine(
            dimension=self.dim,
            scramble=True,
            seed=seed,
        )

        lb_ = None
        ub_ = None

        final_cands = []
        # Generate candidates around each existing point (center)
        for center in centers:
            cands = (
                sobol.draw(2000)
                .to(
                    dtype=torch.float64,
                )
                .cpu()
                .detach()
                .numpy()
            )
            ratio = 1

            L = 0.0001
            Blimit = np.max(ub - lb)

            # Extend the sampling hypersquare region until it contains
            # the given region.
            while ratio == 1 and (L < Blimit):
                lb_ = np.clip(center - L/2, lb, ub)
                ub_ = np.clip(center + L/2, lb, ub)
                cands_ = cp.deepcopy(cands)
                cands_ = (ub_ - lb_) * cands_ + lb_
                ratio, _ = self.get_sample_ratio_in_region(
                    cands_,
                    path,
                )
                if ratio < 1:
                    # Make sure all the proposed candidates were not
                    # previously sampled
                    cands_ = self.filter_out_new_cands(
                        cands=cands_,
                        previous_samples=previous_samples,
                    )
                    ratio, cands_ = self.get_sample_ratio_in_region(
                        cands_,
                        path,
                    )

                    # Append to final list if ratio still < 1
                    if 0 < ratio < 1:
                        final_cands.extend(cands_.tolist())
                L *= 2

        return final_cands

    def propose_rand_samples_sobol(
        self,
        num_samples,
        path,
        lb,
        ub,
        previous_samples,
        seed=0,  # np.random.randint(int(1e6))
        multiprocessing: int = 1,
    ):
        # centers = cp.deepcopy(self.X)
        ratio_check, centers = self.get_sample_ratio_in_region(self.X, path)
        if ratio_check == 0 or len(centers) == 0:
            return []

        # Generate candidates around each existing point (center)
        with Pool(processes=multiprocessing) as pool:
            # Give to each process a chunk of mask
            centers_chunk = np.array_split(centers, multiprocessing)
            args = [(
                    centers_chunk[i],
                    path,
                    lb,
                    ub,
                    previous_samples,
                    seed + i,
                    ) for i in range(multiprocessing)]
            results = pool.starmap(
                self.generate_candidates_around_points,
                args,
                )
            results = [r for r in results if len(r) > 0]
            final_cands = np.vstack(results) if len(results) > 0 else []

        if len(final_cands) > num_samples:
            final_cands_idx = np.random.choice(len(final_cands), num_samples)
            final_cands = final_cands[final_cands_idx]
        else:
            if len(final_cands) == 0:
                final_cands = self.propose_rand_samples(
                    num_samples=num_samples,
                    previous_samples=previous_samples,
                    lb=lb,
                    ub=ub,
                    seed=seed,
                )
        final_cands = np.unique(final_cands, axis=0)
        return final_cands

    def filter_out_new_cands(self, cands, previous_samples):
        new_cands = []
        # Making sure the proposed samples have not
        # been proposed already.
        for c in cands:
            in_previous_samples = any(
                np.array_equal(
                    c,
                    p[0],
                )
                for p in previous_samples
            )
            if not in_previous_samples:
                new_cands.append(c)
        new_cands = np.array(new_cands)
        return new_cands

    # TODO sampling when the current node (conj node) has no samples
    def propose_rand_conj_samples_sobol(
        self,
        num_samples,
        lb,
        ub,
        hypothesis,
        seed=0,  # np.random.randint(int(1e6))
    ):
        # Initialise sampler
        np.random.seed(seed)
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=seed)

        centers = cp.deepcopy(self.X)
        lb_ = None
        ub_ = None
        final_cands = []
        for center in centers:
            cands = (
                sobol.draw(2000)
                .to(
                    dtype=torch.float64,
                )
                .cpu()
                .detach()
                .numpy()
            )

            L = 0.0001
            Blimit = np.max(ub - lb)

            while L < Blimit:
                lb_ = np.clip(center - L / 2, lb, ub)
                ub_ = np.clip(center + L / 2, lb, ub)
                cands_ = cp.deepcopy(cands)
                cands_ = (ub_ - lb_) * cands_ + lb_
                cands_ = [
                    cand
                    for cand in cands_
                    if hypothesis.apply(
                        cand,
                    )
                    is True
                ]
                final_cands.extend(cands_)
                L = L * 2

        final_cands = np.array(final_cands)
        if len(final_cands) > num_samples:
            final_cands_idx = np.random.choice(len(final_cands), num_samples)
            return final_cands[final_cands_idx]
        else:
            if len(final_cands) == 0:
                return self.propose_rand_samples(num_samples, lb, ub)
            else:
                return final_cands

    def propose_rand_samples_in_region(
        self,
        num_samples,
        lb,
        ub,
        seed=0,
    ):
        # Initialise sampler
        np.random.seed(seed)
        sobol = SobolEngine(
            dimension=self.dim,
            scramble=True,
            seed=seed,
        )

        # Get the centers
        centers = []
        dim = len(lb)
        for x in cp.deepcopy(self.X):
            x_is_center = True
            for i in range(dim):
                if not (lb[i] <= x[i] <= ub[i]):
                    x_is_center = False
            if x_is_center:
                centers.append(x)

        lb_ = None
        ub_ = None

        final_cands = []
        for center in centers:
            cands = (
                sobol.draw(2000)
                .to(
                    dtype=torch.float64,
                )
                .cpu()
                .detach()
                .numpy()
            )
            outside = False
            L = 0.0001
            Blimit = np.max(ub - lb)

            # Extend the sampling hypersquare region until it contains
            # the given region.
            while outside is False:
                lb_ = np.clip(center - L / 2, lb, ub)
                ub_ = np.clip(center + L / 2, lb, ub)
                cands_ = cp.deepcopy(cands)
                cands_ = (ub_ - lb_) * cands_ + lb_
                # Check if we have extended outside the bounds
                if L > Blimit:
                    outside = True
                if outside:
                    final_cands.extend(cands_.tolist())
                L = L * 2
        final_cands = np.array(final_cands)
        if len(final_cands) == 0:
            return self.propose_rand_samples(num_samples, lb, ub)
        else:
            return final_cands

    def propose_samples_bo(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        samples: List[Tuple[np.ndarray, float]],
        path: list,
        num_samples: int = 10,
        seed: int = 0,
        multiprocessing: int = 1,
    ):
        """Proposes the next sampling points by optimizing the acquisition
        function.

        Args:
            acquisition: Acquisition function.
            X_sample: Sample locations (n x d).
        Returns:
            np.ndarray: Location of the acquisition function maximum."""
        assert path is not None and len(path) >= 0
        assert lb is not None and ub is not None
        assert samples is not None and len(samples) > 0

        X = []
        num_rand_samples = 10000

        # Sampling for the root node:
        if len(path) == 0:
            # Return random sampling if the node is too small to train a GP.
            if len(self.samples) < self.dim:
                return self.propose_rand_samples(
                    num_samples=num_samples,
                    previous_samples=samples,
                    lb=lb,
                    ub=ub,
                    seed=seed,
                    multiprocessing=multiprocessing,
                )

            X = self.propose_rand_samples(
                num_samples=num_rand_samples,
                previous_samples=samples,
                lb=lb,
                ub=ub,
                seed=seed,
                multiprocessing=multiprocessing,
            )
            self.train_gpr(self.samples)

        # Sampling for the other nodes:
        else:
            # Get root kid parent node
            root_node = path[0][0]
            root_kid_idx = path[0][1]
            parent = root_node.kids[root_kid_idx]
            self.train_gpr(parent.bag)
            # self.train_gpr(self.samples)
            # self.train_gpr(samples)  # learn in unit cube

            X = self.propose_rand_samples_sobol(
                num_rand_samples,
                path,
                lb,
                ub,
                previous_samples=samples,
                seed=seed,
                multiprocessing=multiprocessing,
            )

            # self.plot_boundary(X)
            if len(X) == 0:
                # TODO Remove already sampled points.
                return self.propose_rand_samples(
                    num_samples=num_samples,
                    previous_samples=samples,
                    lb=lb,
                    ub=ub,
                    seed=seed,
                )

        # Propose a diverse batch of candidates.
        proposed_X = []
        xi = 0.0001
        for _ in range(num_samples):
            X_ei = self.expected_improvement(
                X,
                xi=xi,
                use_ei=True,
            )
            X_ei = X_ei.reshape(len(X))

            # Get best x
            index = np.argsort(X_ei)[-1]
            x = X[index]

            # # Making sure the prposed sample is not in the samples or has not
            # # been proposed already.
            # in_samples = any(np.array_equal(x, s[0]) for s in samples)
            # in_proposed = any(np.array_equal(x, s) for s in proposed_X)
            # while in_samples or in_proposed:
            #     # Remove item
            #     X_ei = np.delete(X_ei, index)
            #     if len(X_ei) == 0:
            #         raise ValueError("No possible sample!s")

            #     # Get next best point
            #     index = np.argsort(X_ei)[-1]
            #     x = X[index]

            #     # Check if the newly proposed sample is new.
            #     in_samples = any(np.array_equal(x, s[0]) for s in samples)
            #     in_proposed = any(np.array_equal(x, s) for s in proposed_X)

            # Append proposition
            proposed_X.append(x)

            # Increment xi
            xi *= 5

        proposed_X = np.array(proposed_X)
        return proposed_X

    ###########################
    # sampling with turbo
    ###########################
    # version 1: select a partition, perform one-time turbo search

    def propose_samples_turbo(
        self,
        num_samples,
        path,
        func,
    ):
        # throw a uniform sampling in the selected partition
        X_init = self.propose_rand_samples_sobol(
            num_samples=30,
            path=path,
            lb=func.lb,
            ub=func.ub,
        )

        # get samples around the selected partition
        logging.info(f"sampled {len(X_init)} for the initialization")
        turbo1 = Turbo1(
            f=func,  # Handle to objective function
            lb=func.lb,  # Numpy array specifying lower bounds
            ub=func.ub,  # Numpy array specifying upper bounds
            n_init=30,  # Number of initial bounds from an Latin
            # hypercube design
            max_evals=num_samples,  # Maximum number of evaluations
            batch_size=1,  # How large batch size TuRBO uses
            verbose=True,  # Print information from each batch
            use_ard=True,  # Set to true if you want to use ARD for
            # the GP kernel
            max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
            n_training_steps=50,  # Number of steps of ADAM to learn the hypers
            min_cuda=1024,  # Run on the CPU for small datasets
            device="cpu",  # "cpu" or "cuda"
            dtype="float32",  # float64 or float32
            X_init=X_init,
        )

        proposed_X, fX = turbo1.optimize()
        fX = fX * -1

        return proposed_X, fX

    ###########################
    # random sampling
    ###########################

    # def check_x_in_samples(self, x, samples):
    #     x_in_samples = False
    #     for sample in samples:
    #         if np.array_equal(x, sample):
    #             x_in_samples = True
    #             break
    #     return x_in_samples

    def propose_rand_samples(
        self,
        num_samples: int,
        previous_samples: List[Tuple[np.ndarray, float]],
        lb,
        ub,
        seed=0,
        multiprocessing: int = 1,
    ):
        np.random.seed(seed)
        samples = []
        while len(samples) < num_samples:
            for sample in samples:
                previous_samples.append((sample, None))
            x = np.random.uniform(
                lb,
                ub,
                size=(num_samples - len(samples), self.dim),
            )
            with Pool(processes=multiprocessing) as pool:
                x_chunks = np.array_split(x, multiprocessing)
                args = [(
                    x_chunks[i],    # cands
                    previous_samples,           # previous_samples
                    )
                    for i in range(multiprocessing)]
                results = pool.starmap(
                    self.filter_out_new_cands,
                    args)
                results = [r for r in results if len(r) > 0]
                if len(results) > 0:
                    x = np.vstack(results)
                    samples.extend([(x_, None) for x_ in x])
        samples = [s for s, _ in samples]
        samples = samples[:num_samples]
        return samples

    ###########################
    # learning boundary
    ###########################

    def get_cluster_mean(self, plabel):
        assert plabel.shape[0] == self.fX.shape[0]

        zero_label_fX = []
        one_label_fX = []

        for idx in range(0, len(plabel)):
            if plabel[idx] == 0:
                zero_label_fX.append(self.fX[idx])
            elif plabel[idx] == 1:
                one_label_fX.append(self.fX[idx])
            else:
                message = "kmean should only predict two clusters"
                message += ", Classifiers.py:line73"
                print(message)
                os._exit(1)

        good_label_mean = np.mean(np.array(zero_label_fX))
        bad_label_mean = np.mean(np.array(one_label_fX))
        return good_label_mean, bad_label_mean

    def learn_boundary(self, plabel):
        assert len(plabel) == len(self.X)
        self.svm.fit(self.X, plabel)

    def learn_clusters(self):
        assert len(self.samples) >= 2, "samples must > 0"
        assert self.X.shape[0], "points must > 0"
        assert self.fX.shape[0], "fX must > 0"
        assert self.X.shape[0] == self.fX.shape[0]

        tmp = np.concatenate((self.X, self.fX.reshape([-1, 1])), axis=1)
        assert tmp.shape[0] == self.fX.shape[0]

        self.kmean = self.kmean.fit(tmp)
        plabel = self.kmean.predict(tmp)

        # the 0-1 labels in kmean can be different from the actual
        # flip the label is not consistent
        # 0: good cluster, 1: bad cluster

        self.good_label_mean, self.bad_label_mean = self.get_cluster_mean(
            plabel,
        )

        if self.bad_label_mean > self.good_label_mean:
            for idx in range(0, len(plabel)):
                if plabel[idx] == 0:
                    plabel[idx] = 1
                else:
                    plabel[idx] = 0

        self.good_label_mean, self.bad_label_mean = self.get_cluster_mean(
            plabel,
        )

        return plabel

    def split_data(self):
        good_samples = []
        bad_samples = []
        train_good_samples = []
        train_bad_samples = []
        if len(self.samples) == 0:
            return good_samples, bad_samples

        plabel = self.learn_clusters()
        self.learn_boundary(plabel)

        for idx in range(0, len(plabel)):
            if plabel[idx] == 0:
                # ensure the consistency
                assert self.samples[idx][-1] - self.fX[idx] <= 1
                good_samples.append(self.samples[idx])
                train_good_samples.append(self.X[idx])
            else:
                bad_samples.append(self.samples[idx])
                train_bad_samples.append(self.X[idx])

        train_good_samples = np.array(train_good_samples)
        train_bad_samples = np.array(train_bad_samples)

        assert len(good_samples) + len(bad_samples) == len(
            self.samples
        ), "The sum of the good and bad child nodes' samples must be equal to \
        the parent node's samples!"

        return good_samples, bad_samples
