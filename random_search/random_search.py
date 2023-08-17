import os
import re

import numpy as np
import pandas as pd

from DBO.bayes_opt.util import (ensure_rng, get_rnd_quantities,
                               get_rng_complement)


class RandomOptimizer:

    def __init__(self, space, constraints=[], steps=None):
        self.space = space
        self.dim = len(space)
        self.constraints = self.array_like_constraints(constraints)
        self.results = []
        self.steps = steps
        self.bin = False
        if self.steps:
            self.bin = True

    @staticmethod
    def dict2array(d):
        if not isinstance(d, dict):
            return d
        return list(d.values())

    def _bin(self, x):
        binned = np.empty((self.dim, 1))
        for col, (lower, upper) in enumerate(self.space.values()):
            binned[col] = np.floor(
                (x[col]-lower)/self.steps[col])*self.steps[col]+lower
        return binned.ravel()

    def array_like_constraints(self, constraints):
        '''
        Takes list of logical constraints in terms of space keys,
        and replaces the constraints in terms of array indicies.
        This allows direct evaluation in the acquisition function.
        Parameters
        ----------
        constraints: list of string constraints
        '''
        keys = list(self.space.keys())
        array_like = []
        if keys[0].startswith('x0'):
            # Order the keys from last variable to first variable x0
            keys = [int(k[1:]) for k in keys]    # Remove the 'x'
            keys = sorted(keys, reverse=True)
            keys = [f"x{k}" for k in keys]

            for constraint in constraints:
                tmp = constraint
                for idx, key in enumerate(keys):
                    i = len(keys) - 1 - idx
                    tmp = tmp.replace(key, f'x[{i}]')
                array_like.append(tmp)
        else:
            for constraint in constraints:
                tmp = constraint
                for idx, key in enumerate(keys):
                    tmp = tmp.replace(key, f'x[{idx}]')
                array_like.append(tmp)

        return array_like

    def constrained_rng(
            self,
            n_points,
            ):
        '''
        Random number generator that deals more effectively with highly\
            constrained spaces.

        Works only off single constraint of form L - sum(x_i) >=0
        Where the lower bound of each x_i is 0.

        Generates a fraction of points from a nonuniform sampling that favors\
            limiting cases. (n_var/50)
        Parameters
        ----------
        n_points: integer number of points to generate
        '''
        bounds = [v for v in self.space.values()]

        # Get size and max amount from single constraint
        s = self.constraints[0]
        p = re.compile('(\d+)\]<0.5')
        ms = p.findall(s)
        # Count variables and constrained variables
        n_var = self.dim
        n_constrained_var = 0
        for i in range(n_var):
            if 'x[{}]'.format(i) in s:
                n_constrained_var += 1
        # Get extra points from nonuniformity
        n_nonuniform = int(n_points * n_var/50)
        n_points -= n_nonuniform
        # Initialize randoms
        x = np.zeros((n_points+n_nonuniform, n_var))
        # Get max value of liquid constraint
        try:
            max_val = float(s.split(' ')[0])
        except:
            raise SyntaxError("")

        # Generator for complements that are consistent with max scaled by 
        # simplex if relevant
        # followed by simplex sampling for constrained
        # followed by random sampling for unconstrained
        complements = []
        if ms:
            complements = [int(m) for m in ms]
            for i in range(n_points):
                rem_max_val = -1
                while rem_max_val <= 0:
                    rem_max_val = max_val
                    for complement in complements:
                        if 'x[{}]'.format(complement) in s:
                            x[i, complement] = get_rng_complement(
                                n_constrained_var, self.random_state)
                            # Extract regex, includes two options for
                            # ordering issues
                            reductions = []
                            p = re.compile(
                                '- \(\(x\[{:d}\]<0.5\) \* \(\(\(0.5 - x\[{:d}\]\)/0.5\) \* \(\d+.\d+-\d+.\d+\) \+ \d+.\d+\) \) '.format(
                                    complement, complement))
                            reductions.append(p.findall(s)[0])
                            p = re.compile(
                                '- \(\(x\[{:d}+\]>=0.5\) \* \(\(\(x\[{:d}\] - 0.5\)/0.5\) \* \(\d+.\d+-\d+.\d+\) \+ \d+.\d+\) \) '.format(
                                    complement, complement))
                            reductions.append(p.findall(s)[0])
                            for reduction in reductions:
                                rem_max_val += pd.eval(
                                    reduction, local_dict={'x': x[i, :]})
                        else:
                            x[i, complement] = self.random_state.uniform(
                                bounds[complement, 0], bounds[complement, 1])
                    # Removing lower bound greater than 0 from rem_max_val
                    # and adding in later
                    for j in range(n_var):
                        if j in complements:
                            continue
                        elif 'x[{}]'.format(j) in self.constraints[0]:
                            rem_max_val -= bounds[j][0]
                rnd = get_rnd_quantities(
                    rem_max_val, n_constrained_var, self.random_state)
                cnt = 0
                for j in range(n_var):
                    if j in complements:
                        continue
                    elif 'x[{}]'.format(j) in self.constraints[0]:
                        x[i, j] = min(rnd[cnt]+bounds[j][0], bounds[j][1])
                        cnt += 1
                    else:
                        x[i, j] = self.random_state.uniform(
                            bounds[j][0], bounds[j][1])
        else:
            # Removing lower bound greater than 0 from rem_max_val
            # and adding in later
            rem_max_val = max_val
            for j in range(n_var):
                if 'x[{}]'.format(j) in self.constraints[0]:
                    rem_max_val -= bounds[j][0]
            for i in range(n_points):
                cnt = 0
                rnd = get_rnd_quantities(
                    rem_max_val, n_constrained_var, self.random_state)
                for j in range(n_var):
                    if 'x[{}]'.format(j) in self.constraints[0]:
                        x[i, j] = min(rnd[cnt]+bounds[j][0], bounds[j][1])
                        cnt += 1
                    else:
                        x[i, j] = self.random_state.uniform(
                            bounds[j][0], bounds[j][1])
        # Add nonuniform sampling
        for i in range(n_points, n_nonuniform+n_points):
            rem_max_val = max_val
            var_list = list(range(n_var))
            # Removing lower bound greater than 0 from rem_max_val
            # and adding in later
            rem_max_val = max_val
            for j in range(n_var):
                if 'x[{}]'.format(j) in self.constraints[0]:
                    rem_max_val -= bounds[j][0]
            while var_list:
                j = var_list.pop(np.random.choice(range(len(var_list))))
                if 'x[{}]'.format(j) in self.constraints[0]:
                    x[i, j] = np.random.uniform(
                        0, min(bounds[j][1]-bounds[j][0], rem_max_val)
                        ) + bounds[j][0]
                    if j in complements:
                        reductions = []
                        p = re.compile(
                            '- \(\(x\[{:d}\]<0.5\) \* \(\(\(0.5 - x\[{:d}\]\)/0.5\) \* \(\d+.\d+-\d+.\d+\) \+ \d+.\d+\) \) '.format(
                                complement, complement))
                        reductions.append(p.findall(s)[0])
                        p = re.compile(
                            '- \(\(x\[{:d}+\]>=0.5\) \* \(\(\(x\[{:d}\] - 0.5\)/0.5\) \* \(\d+.\d+-\d+.\d+\) \+ \d+.\d+\) \) '.format(
                                complement, complement))
                        reductions.append(p.findall(s)[0])
                        for reduction in reductions:
                            rem_max_val += pd.eval(
                                reduction, local_dict={'x': x[i, :]})
                        # Contingency for complement generation irreverent
                        # to remaining value
                        # Uses ratio with respect to max_value to pull
                        # fraction of remaining
                        while rem_max_val < 0:
                            for reduction in reductions:
                                rem_max_val -= pd.eval(
                                    reduction, local_dict={'x': x[i, :]})
                            x[i, j] = (rem_max_val / max_val * (0.5 - x[i, j])) + 0.5
                            for reduction in reductions:
                                rem_max_val += pd.eval(
                                    reduction, local_dict={'x': x[i, :]})
                    else:
                        rem_max_val -= (x[i, j] - bounds[j][0])
                else:
                    x[i, j] = self.random_state.uniform(
                        bounds[j][0], bounds[j][1])
        if self.bin:
            x = [self._bin(_x) for _x in x]
        return x

    def random_sampling(self, bin=False):
        guess = []
        for key in self.space:
            guess.append(np.random.uniform(
                low=self.space[key][0],
                high=self.space[key][1]))
        if bin:
            guess = self._bin(guess)
        return guess

    def suggest(self, n_suggestions=1):
        guesses = self.constrained_rng(n_suggestions)
        # x_guess = []
        # for _ in range(n_suggestions):
        #     x_guess.append(self.random_sampling(bin=self.bin))
        # return x_guess
        return guesses

    def maximize(self, obj, init_points=5, n_iter=100, batch=1, seed=0):
        self.seed = seed
        # np.random.seed(self.seed)
        self.random_state = ensure_rng(seed)

        best_value = -np.inf

        for _ in range(init_points):
            x_guesses = self.constrained_rng(batch)
            for x_guess in x_guesses:
                y = obj(self.dict2array(x_guess))
                if isinstance(y, list) or isinstance(y, np.ndarray):
                    y = y[0]
                x_guess = self.dict2array(list(x_guess))
                row = x_guess + [y]
                self.results.append(row)

        iteration = 0
        while iteration < n_iter:
            x_guess = self.suggest(n_suggestions=1)[0]
            y = obj(self.dict2array(x_guess))
            if isinstance(y, list) or isinstance(y, np.ndarray):
                y = y[0]
            x_guess = self.dict2array(list(x_guess))
            row = self.dict2array(x_guess) + [y]
            self.results.append(row)
            if y > best_value:
                best_value = y
            iteration += 1

    def save_data(
            self,
            func_name: str,
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
        scenario_name = "No Hypothesis"
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
            f"dataset_{func_name}_{scenario_name}_{self.seed}.csv",
        )
        columns = list(self.space.keys()) + ["Target"]
        df = pd.DataFrame(
            data=self.results,
            columns=columns,)
        df.to_csv(path)
