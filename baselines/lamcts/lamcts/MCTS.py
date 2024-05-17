"""This module contains the Monte Carlo Tree Searh with hypothesis algorithm.

The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import pickle
from typing import List, Literal, Optional, Tuple

import numpy as np

from hypothesis import Hypothesis

from .Node import Node
from .utils import sample_region

# Set the logging level
logging.basicConfig(level=logging.ERROR)


class MCTS:
    """This class is the meta-algorithm partitioning the search space and\
    applying BO where necessary."""

    def __init__(
        self,
        pbounds: dict,
        func,
        constraints: Optional[List] = None,
        Cp: float = 1.0,
        leaf_size: int = 10,
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
        solver_type: Literal[
            "bo",
            "turbo",
        ] = "bo",
        random_seed: int = 0,
    ):
        """Creates an instance of the meta-algorithm paritioning the search\
            space and applying BO where necessary.

        Args:
            lb (np.ndarray): lower bounds of the black box function's features.
            ub (np.ndarray): upper bounds of the black box function's features.
            dim (int): dimension of the black box function.
            func (_type_): black box function to optimise.
            Cp (float, optional): exploration-exploitation parameter tradeoff.\
                Defaults to 1.0.
            leaf_size (int, optional): minimum sample size of a node.\
                Defaults to 20.
            kernel_type (str, optional): kernel type of the SVM boundary.\
                Defaults to "rbf".
            gamma_type (str, optional): Kernel coefficient for 'rbf', 'poly'\
                and 'sigmoid' kernels. Defaults to "auto".
            solver_type (str, optional): optimiser to apply in selected node.\
                Defaults to "bo" Bayesian optimisation.
        """
        # Black box function parameters
        self.dim = len(pbounds)
        self.lb = np.array([_lb for _lb, _ in pbounds.values()])
        self.ub = np.array([_ub for _, _ub in pbounds.values()])

        # Tree parameters
        self.Cp = Cp
        self.leaf_sample_size = leaf_size
        self.kernel_type = kernel_type
        self.gamma_type = gamma_type
        self.func = func
        self.seed = random_seed

        self.nodes = []
        self.constraints = constraints
        self.samples = []
        self.curt_best_value = float("-inf")
        self.curt_best_sample = None
        self.best_value_trace = []  # For logging

        # Sampling solver parameters
        self.solver_type = solver_type.lower()  # solver can be 'bo' or 'turbo'
        assert self.solver_type in [
            "bo",
            "turbo",
        ], "Solver must be BO or TURBO!"

        # We start the most basic form of the tree: 1 root node and height = 1
        root = Node(
            parent=None,
            dim=self.dim,
            reset_id=True,
            kernel_type=self.kernel_type,
            gamma_type=self.gamma_type,
        )
        self.nodes.append(root)
        self.ROOT = root
        self.CURT = self.ROOT

        # Other parameters
        self.visualization = False

    def initialize(
            self,
            n_init: int = 5,
            batch: int = 1,):
        """Generates initialisation samples in the search space using latin\
        hyper space.
        Args:
            seed (int): seed for the random number generator."""
        init_samples = []
        for i in range(n_init):
            batch_samples = sample_region(
                count=batch,
                lb=self.lb,
                ub=self.ub,
                seed=self.seed + i,
            )
            init_samples.append(batch_samples)
        return init_samples

    def register(
            self,
            x,
            y,
            ):
        self.samples.append((x, y))

    def suggest(
        self,
        n: int = 1,
        multiprocessing: int = 1,
        seed: int = 0,
    ):
        # Treefication: create the tree.
        lb, ub = self.dynamic_treeify()

        # Selection: select the best path.
        leaf, path = self.select()

        # Sampling
        samples = []
        if self.solver_type == "bo":
            samples = leaf.propose_samples_bo(
                num_samples=n,
                path=path,
                samples=self.samples,
                lb=lb,
                ub=ub,
                seed=seed,
                multiprocessing=multiprocessing
            )

        elif self.solver_type == "turbo":
            samples, values = leaf.propose_samples_turbo(
                num_samples=10000,
                path=path,
                func=self.func,
            )

        else:
            raise ValueError("solver not implemented")
        return samples

    def populate_training_data(self, conj=None):
        """Add all samples to the root bag."""
        # Clear all nodes.
        self.ROOT.obj_counter = 0
        for node in self.nodes:
            node.clear_data()
        self.nodes.clear()

        # Only keep the root node.
        new_root = Node(
            parent=None,
            dim=self.dim,
            reset_id=True,
            kernel_type=self.kernel_type,
            gamma_type=self.gamma_type,
        )
        self.nodes.append(new_root)
        self.ROOT = new_root
        self.CURT = self.ROOT

        samples = self.samples
        if conj is not None:
            samples = []
            for s in self.samples:
                if conj.apply(s[0]) is True:
                    samples.append(s)

        if len(samples) > 0:
            self.ROOT.update_bag(samples)

    def get_leaf_statuses(self):
        """Returns the leave's status.

        Returns:
            np.array: the list of the statuses.
        """
        statuses = []
        for node in self.nodes:
            if (
                node.is_leaf() is True
                and len(node.bag) > self.leaf_sample_size
                and node.is_svm_splittable is True
            ):
                statuses.append(True)
            else:
                statuses.append(False)
        return np.array(statuses)

    def get_leaves_to_split_idx(self) -> np.ndarray:
        """Returns the indices of the splittable leaves.

        Returns:
            np.ndarray: array of the splittable leaf indices.
        """
        statuses = self.get_leaf_statuses()
        split_by_samples = np.argwhere(statuses == True).reshape(-1)
        return split_by_samples

    def is_splitable(self) -> bool:
        """Checks if a node is splittable.

        Returns:
            bool: True if the node is splittable else False
        """
        statuses = self.get_leaf_statuses()
        if True in statuses:
            return True
        else:
            return False

    def dynamic_treeify(
        self, hyp: Optional[Hypothesis] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create the tree. A node is split once it contains more than the\
        [leaf_sample_size]. The node will bifurcate into a good and a bad\
        kid.

        Args:
            hyp [Hypothesis]: hypothesis

        Returns:
            Tuple (np.ndarray, np.ndarray): lower and upper bounds of the\
            search space.
        """
        # Partition the space based on the level of optimisation.
        lb = self.lb
        ub = self.ub

        # If a hypothesis is given, only keep the samples within the
        # hypothesis.
        if hyp is not None:
            self.populate_training_data(conj=hyp)
            lb, ub = hyp.get_sol_bounds()

        # If No Hypothesis is given, keep all the samples.
        else:
            self.populate_training_data()
            assert len(self.ROOT.bag) == len(
                self.samples
            ), "The root must contain all the samples at the tree\
                  initialisation."

        assert (
            len(self.nodes) == 1
        ), "There must be only one node (root) at the tree initialisation."

        # While the tree can be further split...
        while self.is_splitable():
            # Get the splitable nodes
            to_split = self.get_leaves_to_split_idx()
            logging.info(f"==> to split:{to_split} total:{len(self.nodes)}")

            for nidx in to_split:
                parent = self.nodes[nidx]

                # parent check if the boundary is splittable by svm
                assert (
                    len(parent.bag) >= self.leaf_sample_size
                ), f"The number of samples in the node ({len(parent.bag)}) \
                    must be at least {self.leaf_sample_size}!"
                assert (
                    parent.is_svm_splittable is True
                ), "The node must be \
                    splittable!"
                logging.info(
                    f"spliting {parent.get_name()} \
                    {len(parent.bag)}"
                )

                # creat two kids, assign the data, and push into lists
                # children's lb and ub will be decided by its parent
                good_kid_data, bad_kid_data = parent.train_and_split()
                assert len(good_kid_data) + len(bad_kid_data) == len(
                    parent.bag,
                )
                assert len(good_kid_data) > 0
                assert len(bad_kid_data) > 0
                good_kid = Node(
                    parent=parent,
                    dim=self.dim,
                    reset_id=False,
                    kernel_type=self.kernel_type,
                    gamma_type=self.gamma_type,
                )
                bad_kid = Node(
                    parent=parent,
                    dim=self.dim,
                    reset_id=False,
                    kernel_type=self.kernel_type,
                    gamma_type=self.gamma_type,
                )
                good_kid.update_bag(good_kid_data)
                bad_kid.update_bag(bad_kid_data)

                parent.update_kids(
                    good_kid=good_kid,
                    bad_kid=bad_kid,
                )
                self.nodes.append(good_kid)
                self.nodes.append(bad_kid)

            logging.info(f"continue split:{self.is_splitable()}")

        self.print_tree()
        return lb, ub

    def collect_samples(
        self,
        sample: list,
        value=None,
    ) -> float:
        """Evaluate the sample, add it to the samples and return its value.

        Args:
            sample (list): the sample to evaluate.
            value (float, optional): value of the sample. Defaults to None.

        Returns:
            float: evaluation of the sample
        """
        if value is None:
            value = self.func(sample)

        if value > self.curt_best_value:
            self.curt_best_value = value
            self.curt_best_sample = sample
            self.best_value_trace.append(
                (
                    value,
                    len(self.samples),
                )
            )

        self.samples.append((sample, value))
        return value

    def print_tree(self):
        """Prints the tree structure."""
        logging.info("-" * 100)
        for node in self.nodes:
            node.get_score(
                Cp=self.Cp,
            )  # Compute the node score
            logging.info(node)
        logging.info("-" * 100)

    def reset_to_root(self):
        """Sets the current node to be the root."""
        self.CURT = self.ROOT

    def load_agent(self, agent_path: str = "mcts_agent"):
        """Loads the agent from a pickle file.

        Args:
            node_path (str, optional): path of the file to unpickle. Defaults \
            to "mcts_agent".
        """
        if os.path.isfile(agent_path) is True:
            with open(agent_path, "rb") as json_data:
                self = pickle.load(json_data)
                logging.info(f"=====>loads:{len(self.samples)} samples")

    def dump_agent(self, agent_path: str = "mcts_agent"):
        """Pickles the agent into a file.

        Args:
            agent_path (str, optional): path of the file to save the object in\
            . Defaults to "mcts_agent".
        """
        logging.info("dumping the agent.....")
        with open(agent_path, "wb") as outfile:
            pickle.dump(self, outfile)

    def dump_samples(self, folder: str):
        """Saves the samples into a file.

        Args:
            folder (str): folder to save the sample file in.
        """
        sample_counter = len(self.samples)
        sample_path = f"{folder}/samples_{str(sample_counter)}"
        with open(sample_path, "wb") as outfile:
            pickle.dump(self.samples, outfile)

    def dump_trace(self):
        """Saves the trace into a file."""
        trace_path = "best_values_trace"
        final_results_str = json.dumps(self.best_value_trace)
        with open(trace_path, "a") as f:
            f.write(final_results_str + "\n")

    def greedy_select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        path = []
        if self.visualization is True:
            curt_node.plot_samples_and_boundary(self.func)
        while curt_node.is_leaf() is False:
            UCT = []
            for i in curt_node.kids:
                UCT.append(i.get_xbar())
            choice = np.random.choice(
                np.argwhere(
                    UCT == np.amax(UCT),
                ).reshape(-1),
                1,
            )[0]
            path.append((curt_node, choice))
            curt_node = curt_node.kids[choice]
            if curt_node.is_leaf() is False and self.visualization is True:
                curt_node.plot_samples_and_boundary(self.func)
            print("=>", curt_node.get_name(), end=" ")
        print("")
        return curt_node, path

    def select(self) -> Tuple[Node, List[Tuple[Node, int]]]:
        """Selects the most promising path from the root to a leaf node.

        Returns:
            Tuple[Node, List[Tuple[Node, int]]]: the leaf node and the path.
        """
        self.reset_to_root()
        curt_node = self.ROOT
        path = []
        path_in_str = ""

        while curt_node.is_leaf() is False:
            scores = []
            # Compute the kids' scores.
            for kid in curt_node.kids:
                kid_score = kid.get_score(
                    Cp=self.Cp,
                )
                # kid.get_uct(Cp=self.Cp)
                scores.append(kid_score)

            # Choose the kid with the best score.
            choice = np.random.choice(
                np.argwhere(scores == np.amax(scores)).reshape(-1),
                1,
            )[0]
            path.append((curt_node, choice))
            curt_node = curt_node.kids[choice]

            path_in_str += f"=> {curt_node.get_name()} "
            logging.info(path_in_str)

        return curt_node, path

    def backpropogate(
        self,
        leaf,
        acc,
    ):
        curt_node = leaf
        while curt_node is not None:
            if curt_node.n == 0:
                curt_node.x_bar = acc
            else:
                assert curt_node.n > 0
                curt_node.x_bar = (curt_node.x_bar * curt_node.n + acc) / (
                    curt_node.n + 1
                )
            curt_node.n += 1
            curt_node = curt_node.parent

    def search(
        self,
        budget: int,
        batch: int = 1,
        target: Optional[float] = None,
    ):
        """Searches for the maximum of the black box function.

        Args:
            budget (int): number of iterations.
            batch (int, optional): number of samples per batch. Defaults to 1.
            target (float, optional): stop the optimisation when the optimiser\
                  finds this value. Defaults to None.
            optimisation (_type_, optional): optimisation object from \
                the CATAI platform. Defaults to None.

        Raises:
            ValueError: Solver must be either 'bo' or 'turbo'
            Exception: _description_
        """
        # For each iteration
        for idx in range(budget):
            # Pint logs
            logging.info("")
            logging.info("=" * 10)
            logging.info(f"iteration: {idx + 1}/{budget}")
            logging.info("=" * 10)

            samples = []

            # Treefication: create the tree.
            lb, ub = self.dynamic_treeify()

            # Selection: select the best path.
            leaf, path = self.select()

            # Sampling
            values = []
            if self.solver_type == "bo":
                samples = leaf.propose_samples_bo(
                    num_samples=batch,
                    path=path,
                    samples=self.samples,
                    lb=lb,
                    ub=ub,
                    seed=idx,
                )

            elif self.solver_type == "turbo":
                samples, values = leaf.propose_samples_turbo(
                    num_samples=10000,
                    path=path,
                    func=self.func,
                )

            else:
                raise ValueError("solver not implemented")

            # Evaluation and Backpropagation
            for i in range(len(samples)):
                if self.solver_type == "bo":
                    value = self.collect_samples(
                        sample=samples[i],
                    )

                elif self.solver_type == "turbo":
                    value = self.collect_samples(
                        sample=samples[i],
                        value=values[i],
                    )
                else:
                    raise Exception("solver not implemented")

                self.backpropogate(leaf, value)

                # Early stopping
                if target is not None and value >= target:
                    return

                # Logs
                logging.info(f"total samples: {len(self.samples)}")
                logging.info(f"current sample: {self.samples[-1][0]}")
                logging.info(f"current value: {self.samples[-1][1]}")
                logging.info(
                    f"current best f(x): {self.curt_best_value}",
                )
                logging.info(f"current best x: {self.curt_best_sample}")
