# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import List, Literal, Tuple

import numpy as np

from .Classifier import Classifier


class Node:
    obj_counter = 0

    def __init__(
        self,
        dim: int = 0,
        parent=None,
        reset_id=False,
        kernel_type: Literal[
            "linear",
            "poly",
            "rbf",
            "sigmoid",
            "precomputed",
        ] = "rbf",
        gamma_type: Literal[
            "scale",
            "auto",
        ] = "auto",
    ):
        # Note: every node is initialized as a leaf,
        # only internal nodes equip with classifiers to make decisions
        self.parent = parent
        self.x_bar = float("inf")
        self.n = 0
        self.kids = []  # 0:good, 1:bad

        self.bag = []
        self.is_svm_splittable = False
        self.dim = dim
        self.classifier = Classifier(
            samples=[],
            dim=self.dim,
            kernel_type=kernel_type,
            gamma_type=gamma_type,
        )

        if reset_id:
            Node.obj_counter = 0

        self.id = Node.obj_counter

        # data for good and bad kids, respectively
        Node.obj_counter += 1
        self.score = self.get_uct()
        # self.scores = {}

    def __str__(self):
        name = self.get_name()
        name = self.pad_str_to_8chars(name, 7)
        name += self.pad_str_to_8chars(
            "is good:" + str(self.is_good_kid()),
            15,
        )
        name += self.pad_str_to_8chars(
            "is leaf:" + str(self.is_leaf()),
            15,
        )

        # val = 0
        name += self.pad_str_to_8chars(
            " mean:{0:.4f}   ".format(round(self.get_xbar(), 3)), 20
        )
        name += self.pad_str_to_8chars(
            " score:{0:.4f}   ".format(round(self.score, 3)), 20
        )

        name += self.pad_str_to_8chars(
            "sp/n:" + str(len(self.bag)) + "/" + str(self.n), 15
        )
        # upper_bound = np.around(
        #                       np.max(self.classifier.X, axis=0),
        #                       decimals=2,
        #                       )
        # lower_bound = np.around(
        #                       np.min(self.classifier.X, axis=0),
        #                       decimals=2,
        #                       )
        # boundary = ""
        # for idx in range(0, self.dims):
        #     boundary += str(lower_bound[idx]) + f">{str(upper_bound[idx])} "

        # name  += ( self.pad_str_to_8chars( 'bound:' + boundary, 60 ) )

        parent = "----"
        if self.parent is not None:
            parent = self.parent.get_name()
        parent = self.pad_str_to_8chars(parent, 10)

        name += " parent:" + parent

        kids = ""
        kid = ""
        for k in self.kids:
            kid = self.pad_str_to_8chars(k.get_name(), 10)
            kids += kid
        name += " kids:" + kids

        return name

    def update_kids(self, good_kid, bad_kid):
        """Add the good and bad kids to the node's kids.

        Args:
            good_kid (Node): the child node with the greater mean.
            bad_kid (Node): the child node with the lower mean.
        """
        assert len(self.kids) == 0

        self.kids.append(good_kid)
        self.kids.append(bad_kid)
        mean_good_kid = self.kids[0].classifier.get_mean()
        mean_bad_kid = self.kids[1].classifier.get_mean()

        assert (
            mean_good_kid >= mean_bad_kid
        ), f"The good child node mean ({mean_good_kid}) must be greater than \
        the bad child node mean ({mean_bad_kid})!"

    def is_good_kid(self) -> bool:
        """Checks if the current node is its parent's good node.

        Returns:
            bool: True if it is, False if not.
        """
        if self.parent is not None:
            if self.parent.kids[0] == self:
                return True
            else:
                return False
        else:
            return False

    def is_leaf(self) -> bool:
        """Checks if the current node is a leaf node.

        Returns:
            bool: True if it is, False if not.
        """
        if len(self.kids) == 0:
            return True
        else:
            return False

    def visit(self):
        self.n += 1

    def update_bag(
        self,
        samples,
    ):
        assert len(samples) > 0

        self.bag.clear()
        self.bag.extend(samples)
        self.classifier.update_samples(self.bag)
        if len(self.bag) <= 2:
            self.is_svm_splittable = False
        else:
            self.is_svm_splittable = self.classifier.is_svm_splittable()
        self.x_bar = self.classifier.get_mean()
        self.n = len(self.bag)
        # self.scores[self.n] = self.x_bar

    def clear_data(self):
        self.bag.clear()

    def get_name(self):
        # state is a list of jsons
        return "node" + str(self.id)

    def pad_str_to_8chars(
        self,
        ins,
        total,
    ):
        if len(ins) <= total:
            ins += " " * (total - len(ins))
            return ins
        else:
            return ins[0:total]

    def get_rand_sample_from_bag(self):
        # np.random.seed(0)
        if len(self.bag) > 0:
            upeer_boundary = len(list(self.bag))
            rand_idx = np.random.randint(0, upeer_boundary)
            return self.bag[rand_idx][0]
        else:
            return None

    def get_parent_str(self) -> str:
        """Returns the parent's name.

        Raises:
            ValueError: if the node is root and hence has no parent.

        Returns:
            str: the parent node name.
        """
        if self.parent is not None:
            return self.parent.get_name()
        else:
            raise ValueError(
                "This node is the root node and therefore has no\
                 parents."
            )

    def propose_samples_bo(
        self,
        num_samples: int,
        path: list,
        samples: List[Tuple[np.ndarray, float]],
        lb,
        ub,
        seed: int = 0,
        multiprocessing: int = 1,
    ) -> np.ndarray:
        """Proposes the next sampling points in the node using BO.

        Args:
            num_samples (int): number of samples to return.
            path (list): path from the root node to the current node.
            samples (list): list of samples to train the Gaussian process\
                against

        Returns:
            np.ndarray: array of proposed samples.
        """
        # Get proposed samples from the classifier.
        proposed_X = self.classifier.propose_samples_bo(
            num_samples=num_samples,
            path=path,
            lb=lb,
            ub=ub,
            samples=samples,
            seed=seed,
            multiprocessing=multiprocessing,
        )

        return proposed_X

    def propose_samples_turbo(
        self,
        num_samples: int,
        path: list,
        func,
    ):
        proposed_X, fX = self.classifier.propose_samples_turbo(
            num_samples=num_samples,
            path=path,
            func=func,
        )
        return proposed_X, fX

    def propose_samples_rand(
        self,
        num_samples,
        previous_samples,
        seed=0,
    ):
        assert num_samples > 0
        samples = self.classifier.propose_rand_samples(
            num_samples=num_samples,
            previous_samples=previous_samples,
            lb=self.lb,
            ub=self.ub,
            seed=seed,
        )
        return samples

    def get_uct(self, Cp: float = 1):
        # Max uct if it is a root conj and
        # does not have enough samples so that the
        # conj will be chosen and sampled
        if self.parent is None:
            return float("inf")
        if self.n == 0:
            return float("inf")

        uct = self.x_bar + 2 * Cp * math.sqrt(
            2 * np.power(self.parent.n, 0.5) / self.n,
        )
        return uct

    def get_score(self, Cp: float = 1):
        self.score = self.get_uct(Cp=Cp)
        return self.score
        _, y = zip(*self.bag)
        self.score = self.get_uct(Cp=Cp) + max(y)
        return self.score

    def get_xbar(self):
        return self.x_bar

    def get_n(self):
        return self.n

    def train_and_split(self):
        assert len(self.bag) >= 2
        self.classifier.update_samples(self.bag)
        good_kid_data, bad_kid_data = self.classifier.split_data()
        assert len(good_kid_data) + len(bad_kid_data) == len(self.bag)
        return (
            good_kid_data,
            bad_kid_data,
        )

    def plot_samples_and_boundary(self, func):
        name = self.get_name() + ".pdf"
        self.classifier.plot_samples_and_boundary(func, name)
