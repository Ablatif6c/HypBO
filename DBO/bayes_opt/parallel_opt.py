import itertools
import re
import time
import warnings
from collections import OrderedDict
from multiprocessing import Pool, Process, Queue

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.cluster import KMeans

from .target_space import _hashable
from .util import UtilityFunction, ensure_rng

TIMEOUT_TIME = 4 * 60 * 60  # Hours to timeout


class LocalOptimizer():
    ''' Class of helper functions for minimization (Class needs to be picklable)'''

    def __init__(self, ac, gp, y_max, bounds, method="L-BFGS-B"):
        self.ac = ac
        self.gp = gp
        self.y_max = y_max
        self.bounds = bounds
        self.method = method

    def func_max(self, x):
        return -self.ac(x.reshape(1, -1), gp=self.gp, y_max=self.y_max)

    def func_min(self, x):
        return self.ac(x.reshape(1, -1), gp=self.gp, y_max=self.y_max)

    def minimizer(self, x_try):
        return minimize(self.func_min,
                        x_try.reshape(1, -1),
                        bounds=self.bounds,
                        method="L-BFGS-B")

    def maximizer(self, x_try):
        res = minimize(self.func_max,
                       x_try.reshape(1, -1),
                       bounds=self.bounds,
                       method="L-BFGS-B")
        res.fun[0] = -1 * res.fun[0]
        return res


class LocalConstrainedOptimizer():
    ''' Class of helper functions for minimization \
    (Class needs to be picklable)'''

    def __init__(self, ac, gp, y_max, bounds, method="SLSQP", constraints=()):
        self.ac = ac
        self.gp = gp
        self.y_max = y_max
        self.bounds = bounds
        self.method = method
        self.constraints = constraints

    def func_max(self, x):
        return -self.ac(x.reshape(1, -1), gp=self.gp, y_max=self.y_max)

    def func_min(self, x):
        return self.ac(x.reshape(1, -1), gp=self.gp, y_max=self.y_max)

    def minimizer(self, x_try):
        return minimize(self.func_min,
                        x_try.reshape(1, -1),
                        bounds=self.bounds,
                        method="L-BFGS-B")

    def maximizer(self, x_try):
        res = minimize(self.func_max,
                       x_try.reshape(1, -1),
                       bounds=self.bounds,
                       method=self.method,
                       constraints=self.constraints)
        res.fun = [-1 * res.fun]
        return res


class LocalComplementOptimizer(LocalConstrainedOptimizer):
    ''' Class of helper functions for optimization including complement \
        variables. TAKES STRING CONSTRAINTS NOT FUNCTIONS'''

    def __init__(self, ac, gp, y_max, bounds, method="SLSQP", constraints=[],
                 text_constraints=[],
                 ):
        super().__init__(ac, gp, y_max, bounds, method, constraints)
        self.text_constraints = text_constraints  # Array like constraints
        self.constraint_sets = []

        # Set up complemets
        ms = []
        p = re.compile('(\d+)\]<0.5')
        for s in self.text_constraints:
            ms.extend(p.findall(s))
        # Shifted to avoid sign issue with 0
        complements = [int(m) + 1 for m in ms]
        complement_assignments = list(itertools.product(*((x, -x) for x in complements)))
        for assignment in complement_assignments:
            dicts = []
            for constraint in self.text_constraints:
                dicts.append(self.relax_complement_constraint(constraint, assignment))
            self.constraint_sets.append(dicts)

    def relax_complement_constraint(self, constraint, assignment):
        '''
        Takes in string constraint containing complement, and removes one 
        term and all logicals to create continuous function. 
        Term removed depends on sign in assignment
        Arguments
        ==========
        constraint: string, array style constraint
        assignment: tuple, postive or negative integers to dictate removal of complement constraints
            These should be 1+ index in the array to avoid issue with 0=-0
            Negative will remove the condition where x[i] >=0.5
            Positive will remove the condition where x[i] < 0.5
        '''
        new_constraint = constraint
        for i in assignment:
            if i < 0:
                p = re.compile(
                    '- \(\(x\[{:d}+\]>=0.5\) \* \(\(\(x\[{:d}\] - 0.5\)/0.5\) \* \(\d+.\d+-\d+.\d+\) \+ \d+.\d+\) \)'.format(
                        abs(i + 1), abs(i + 1)))
                new_constraint = p.sub('', new_constraint)
                p = re.compile('\(x\[{:d}\]<0.5\) \* '.format(abs(i + 1)))
                new_constraint = p.sub('', new_constraint)
            else:
                p = re.compile(
                    '- \(\(x\[{:d}\]<0.5\) \* \(\(\(0.5 - x\[{:d}\]\)/0.5\) \* \(\d+.\d+-\d+.\d+\) \+ \d+.\d+\) \)'.format(
                        abs(i - 1), abs(i - 1)))
                new_constraint = p.sub('', new_constraint)
                p = re.compile('\(x\[{:d}+\]>=0.5\) \* '.format(abs(i - 1)))
                new_constraint = p.sub('', new_constraint)
        funcs = []
        st = "def f_{}(x): return pd.eval({})\nfuncs.append(f_{})".format(1, new_constraint, 1)
        exec(st)
        dict = {'type': 'ineq', 'fun': funcs[0]}
        return dict

    def maximizer(self, x_try):
        '''Overide maximizer to generate multiple options for each \
            complement'''
        results = []
        for constraint_set in self.constraint_sets:
            res = minimize(self.func_max,
                           x_try.reshape(1, -1),
                           bounds=self.bounds,
                           method=self.method,
                           constraints=constraint_set,
                           )
            res.fun = [-1 * res.fun]
            tmp = False
            for dict in self.constraints:
                if dict['fun'](res.x) < 0:
                    tmp = True
            if tmp:
                res.success = False
            results.append(res)
        results.sort(key=lambda x: x.fun[0], reverse=True)
        return results[0]


def satisfy_initial_point(
        instance,
        mask,
        x,
        bin=True,
        seed: int = 0,
        ):
    for idx in range(len(mask)):
        while ~mask[idx]:
            seed = seed + idx + 1
            mask[idx] = True
            proposal = instance.constrained_rng(
                n_points=1,
                bin=bin,
                process_seed=seed,
                )
            proposal = proposal.reshape(-1,)
            x[idx] = proposal
            for dict in instance.get_constraint_dict():
                if dict['fun'](proposal) < 0:
                    mask[idx] = False
    return mask, x


def maximize_acq(
        instance,
        ac,
        x,
        complements: bool,
):
    proposals = []
    for x_try in x:
        try:
            gp = instance._gp
            y_max = instance._space.target.max()
            bounds = instance._space.bounds

            lo = None
            # Class of helper functions for minimization
            # (Class needs to be picklable)
            if complements:
                lo = LocalComplementOptimizer(
                    ac,
                    gp,
                    y_max,
                    bounds,
                    constraints=instance.get_constraint_dict(),
                    text_constraints=instance.constraints,
                    )
            else:
                lo = LocalConstrainedOptimizer(
                    ac,
                    gp,
                    y_max,
                    bounds,
                    constraints=instance.get_constraint_dict(),
                    )
    
            res = lo.maximizer(x_try)
        # SLSQP can diverge if it starts near or outside a boundary on a
        # flat surface
        except ValueError:
            print("Note for Phil's benefit, \
                    ValueError in sklearn based maximizer.")
        # See if success
        if not res.success:
            continue
        point = _hashable(instance.space._bin(res.x))
        # Double check on constraints
        tmp = False
        for dict in instance.get_constraint_dict():
            if dict['fun'](point) < 0:
                tmp = True
        if tmp:
            continue
        if point in instance.space:
            continue
        if point in instance.partner_space:
            continue
        proposals.append((res.x, float(res.fun[0])))
    return proposals


def disc_acq_max(ac, instance, n_acqs=1, n_warmup=100000, n_iter=250,
                 multiprocessing=1,):
    """
    A function to find the maximum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (250) random starting points.

    Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    instance: DiscreteBayesianOptimization object instance.
    n_acqs: Integer number of acquisitions to take from acquisition function ac.
    n_warmup: number of times to randomly sample the aquisition function
    n_iter: number of times to run scipy.minimize
    multiprocessing: number of cores for multiprocessing of scipy.minimize

    Returns
    -------
    List of the arg maxs of the acquisition function.
    """

    # Inialization from instance
    gp = instance._gp
    y_max = instance._space.target.max()
    bounds = instance._space.bounds
    steps = instance._space.steps
    random_state = instance._random_state

    # Class of helper functions for minimization (Class needs to be picklable)
    lo = LocalOptimizer(ac, gp, y_max, bounds)

    # Warm up with random points
    x_tries = np.floor((random_state.uniform(bounds[:, 0], bounds[:, 1],
                                             size=(n_warmup, bounds.shape[0])) - bounds[:, 0]) /
                       steps) * steps + bounds[:, 0]
    ys = ac(x_tries, gp=gp, y_max=y_max)

    # Using a dictionary to update top n_acqs,and retains the threshold for the bottom
    x_tries = x_tries[ys.argsort()[::-1]]
    ys = ys[ys.argsort()[::-1]]
    acqs = {}
    for idx in range(x_tries.shape[0]):
        if _hashable(x_tries[idx, :]) in instance.space:
            continue
        else:
            acqs[_hashable(x_tries[idx, :])] = ys[idx]
        if len(acqs) > n_acqs: break
    acq_threshold = sorted(acqs.items(), key=lambda t: (t[1], t[0]))[0]

    # Explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))

    if multiprocessing > 1:
        # Memory unconscious multiprocessing makes list of n_iter results...
        pool = Pool(multiprocessing)
        results = list(pool.imap_unordered(lo.maximizer, x_seeds))
        pool.close()
        pool.join()
        for res in results:
            if not acqs or res.fun[0] >= acq_threshold[1]:
                if _hashable(instance.space._bin(res.x)) in instance.space:
                    continue
                if _hashable(instance.space._bin(res.x)) in instance.partner_space:
                    continue
                acqs[_hashable(instance.space._bin(res.x))] = res.fun[0]
                if len(acqs) > n_acqs:
                    del acqs[acq_threshold[0]]
                    acq_threshold = sorted(acqs.items(), key=lambda t: (t[1], t[0]))[0]
    else:
        for x_try in x_seeds:
            # Maximize the acquisition function
            res = lo.maximizer(x_try)
            # See if success
            if not res.success:
                continue

            # Attempt to store it if better than previous maximum.
            # If it is new point, delete and replace threshold value
            if not acqs or res.fun[0] >= acq_threshold[1]:
                if _hashable(instance.space._bin(res.x)) in instance.space:
                    continue
                if _hashable(instance.space._bin(res.x)) in instance.partner_space:
                    continue
                acqs[_hashable(instance.space._bin(res.x))] = res.fun[0]
                if len(acqs) > n_acqs:
                    del acqs[acq_threshold[0]]
                    acq_threshold = sorted(acqs.items(), key=lambda t: (t[1], t[0]))[0]

    return [key for key in acqs.keys()]


def disc_acq_KMBBO(ac, instance, n_acqs=1, n_slice=200, n_warmup=100000,
                   n_iter=250, multiprocessing=1,):
    """
    A function to find the batch sampled acquisition function. Uses slice \
        sampling of continuous space,
    followed by k-means.The k- centroids are then binned and checked for \
        redundancy.
    slice: J. Artif. Intell. Res. 1, 1-24 (2017)
    slice+k-maens: arXiv:1806.01159v2
    
    Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    instance: DiscreteBayesianOptimization object instance.
    n_acqs: Integer number of acquisitions to take from acquisition function \
        ac (the k in k-means).
    n_slice: integer number of slice samples (the data fed to k-means)
    n_warmup: number of times to randomly sample the aquisition function for \
        a_min
    n_iter: number of times to run scipy.minimize for a_min
    multiprocessing: number of cores for multiprocessing of scipy.minimize

    Returns
    -------
    List of the sampled means of the acquisition function.
    """
    assert n_slice >= n_acqs, "number of points in slice (n_slice) must be greater \
                             than number of centroids in k-means (n_acqs)"

    # Inialization from instance
    gp = instance._gp
    y_max = instance._space.target.max()
    bounds = instance._space.bounds
    steps = instance._space.steps
    random_state = instance._random_state
    slice = np.zeros((n_slice, bounds.shape[0]))

    # Class of helper functions for optimization (Class needs to be picklable)
    lo = LocalOptimizer(ac, gp, y_max, bounds)

    # First finding minimum of acquisition function
    # Warm up with random points
    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_warmup, bounds.shape[0]))
    ys = ac(x_tries, gp=gp, y_max=y_max)
    a_min = ys.min()
    # Explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))
    if multiprocessing > 1:
        # Memory unconscious multiprocessing makes list of n_iter results...
        pool = Pool(multiprocessing)
        results = list(pool.imap_unordered(lo.minimizer, x_seeds))
        pool.close()
        pool.join()
        a_min = sorted(results, key=lambda x: x.fun[0])[0].fun[0]
    else:
        for x_try in x_seeds:
            res = lo.minimizer(x_try)
            if not res.success:
                continue
            if a_min is None or res.fun[0] <= a_min:
                a_min = res.fun[0]
    if a_min > 0: a_min = 0  # The algorithm will fail the minimum found is greater than 0

    # Initial sample over space
    s = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(1, bounds.shape[0]))
    # Slice aggregation
    for i in range(n_slice):
        u = random_state.uniform(a_min, ac(s, gp=gp, y_max=y_max))
        while True:
            s = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(1, bounds.shape[0]))
            if ac(s, gp=gp, y_max=y_max) > u:
                slice[i] = s
                break

    unique = False
    i = 0
    while not unique:
        i += 1
        if i > 50: raise RuntimeError("KMBBO sampling cannot find unique new values after 50 attempts.")
        # Find centroids
        kmeans = KMeans(n_clusters=n_acqs,
                        random_state=random_state,
                        n_jobs=multiprocessing).fit(slice)

        # Make hashable, check for uniqueness, and assert length
        acqs = {}
        unique = True
        for i in range(n_acqs):
            if _hashable(instance.space._bin(kmeans.cluster_centers_[i, :])) in instance.space:
                unique = False
                break
            if _hashable(instance.space._bin(kmeans.cluster_centers_[i, :])) in instance.partner_space:
                unique = False
                break
            acqs[_hashable(instance.space._bin(kmeans.cluster_centers_[i, :]))] = i
        if len(acqs) != n_acqs:
            unique = False
        random_state = None
    assert len(acqs) == n_acqs, "k-means clustering is not distinct in discretized space!"
    return [key for key in acqs.keys()]


def disc_constrained_acq_max(
        ac,
        instance,
        n_acqs: int = 1,
        n_warmup: int = 100,
        n_iter: int = 300,
        multiprocessing: int = 1,
        complements: bool = False,
        seed: int = 0,
        ):
    """
    A function to find the maximum of the acquisition function subject to \
        inequality constraints

    It uses a combination of random sampling (cheap) and the 'SLSQP'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running SLSQP from `n_iter` (250) random starting points.
    # TODO: parallelization. Issues present in pickling constraint functions

    Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    instance: DiscreteBayesianOptimization object instance.
    n_acqs: Integer number of acquisitions to take from acquisition function\
          ac.
    n_warmup: number of times to randomly sample the aquisition function
    n_iter: number of times to run scipy.minimize
    multiprocessing: integer, number of processes to use
    complements: logical, whether or not to consider complements

    Returns
    -------
    List of the arg maxs of the acquisition function.
    """
    import random
    random.seed(seed)
    # Inialization from instance
    gp = instance._gp
    y_max = instance._space.target.max()

    # Warm up with random points
    x_tries = []
    with Pool(processes=multiprocessing) as pool:
        args = [(
            n_warmup // multiprocessing,    # n_points
            True,                           # bin
            random.randint(0, 10000),       # process_seed
            )
            for i in range(multiprocessing,)]
        results = pool.starmap(
            instance.constrained_rng,
            args)
        x_tries = np.vstack(results)  
    # Apply constraints to initial tries
    mask = np.ones((x_tries.shape[0],), dtype=bool)
    for dict in instance.get_constraint_dict():
        for i, x in enumerate(x_tries[:]):
            if dict['fun'](x) < 0:
                mask[i] = False

    # Satisfy each initial point to ensure n_warmup
    # This should not be needed given the nature of the constrained_rng
    with Pool(processes=multiprocessing) as pool:
        # Give to each process a chunk of mask
        masks = np.array_split(mask, multiprocessing)
        x = np.array_split(x_tries, multiprocessing)
        args = [(
                instance,
                masks[i],
                x[i],
                True,       # bin
                random.randint(0, 10000),
                ) for i in range(multiprocessing)]
        results = pool.starmap(
            satisfy_initial_point,
            args,
            )
        masks, x = list(zip(*results))
        mask = np.concatenate(masks)
        x_tries = np.vstack(x)

        # Remove duplicates
        x_tries = np.array(
            list(OrderedDict((tuple(x), x) for x in x_tries).values()))

    ys = ac(
            x_tries,
            gp=gp,
            y_max=y_max,
            )

    # Using a dictionary to update top n_acqs,and retains the threshold for
    # the bottom
    x_tries = x_tries[ys.argsort()[::-1]]
    ys = ys[ys.argsort()[::-1]]
    acqs = {}
    for idx in range(x_tries.shape[0]):
        point = _hashable(x_tries[idx, :])
        breakout = False
        for dict in instance.get_constraint_dict():
            if dict['fun'](point) < 0:
                breakout = True
        if breakout:
            continue
        if point in instance.space:
            continue
        elif point in instance.partner_space:
            continue
        else:
            acqs[point] = ys[idx]
        if len(acqs) >= n_acqs:
            break
    acq_threshold = None
    if len(acqs) > 0:
        acq_threshold = sorted(acqs.items(), key=lambda t: (t[1], t[0]))[0]

    # Explore the parameter space more throughly
    x_seeds = []
    with Pool(processes=multiprocessing) as pool:
        args = [(
                n_iter // multiprocessing,  # n_points
                True,                      # bin
                random.randint(0, 10000),       # process_seed
                )
                for i in range(multiprocessing,)]
        results = pool.starmap(
            instance.constrained_rng,
            args)
        x_seeds = np.vstack(results)

    # Ensure seeds satisfy initial constraints
    mask = np.ones(
        (x_seeds.shape[0],),
        dtype=bool,
        )
    for dict in instance.get_constraint_dict():
        for i, x in enumerate(x_seeds[:]):
            if dict['fun'](x) < 0:
                mask[i] = False

    # If not replace seeds with satisfactory points
    with Pool(processes=multiprocessing) as pool:
        # Give to each process a chunk of mask and x_seeds
        masks = np.array_split(mask, multiprocessing)
        x = np.array_split(x_seeds, multiprocessing)
        args = [(
                instance,
                masks[i],
                x[i],
                True,      # bin
                random.randint(0, 10000),       # process_seed
                ) for i in range(multiprocessing)]
        results = pool.starmap(
            satisfy_initial_point,
            args,
            )
        masks, x = list(zip(*results))
        mask = np.concatenate(masks)
        x_seeds = np.vstack(x)

        # Remove duplicates
        x_seeds = np.array(
            list(OrderedDict((tuple(x), x) for x in x_tries).values()))

    # Maximize the acquisition function
    with Pool(processes=multiprocessing) as pool:
        # Give to each process a chunk of x_seeds
        x = np.array_split(x_seeds, multiprocessing)
        args = [(
                instance,
                ac,
                x[i],
                complements,
                ) for i in range(multiprocessing)]
        results = pool.starmap(
            maximize_acq,
            args,
            )
        results = [r for r in results if len(r) > 0]

        if len(results) > 0:
            proposals = np.concatenate(results)
            for k, v in proposals:
                if not acqs or v >= acq_threshold[1]:
                    acqs[_hashable(instance.space._bin(k))] = v
                    if len(acqs) > n_acqs:
                        del acqs[acq_threshold[0]]
                        acq_threshold = sorted(acqs.items(), key=lambda t: (t[1], t[0]))[0]

    if instance.verbose == 3:
        print("Sorted acquisition function values: ", sorted(acqs.values()))
    return [key for key in acqs.keys()]


def disc_constrained_acq_KMBBO(
                                ac,
                                instance,
                                n_acqs=1,
                                n_slice=200,
                                n_warmup=100000,
                                n_iter=250,
                                multiprocessing=1):
    """
    A function to find the batch sampled acquisition function. \
        Uses slice sampling of continuous space,
    followed by k-means.The k- centroids are then binned and \
        checked for redundancy.
    slice: J. Artif. Intell. Res. 1, 1-24 (2017)
    slice+k-maens: arXiv:1806.01159v2

    Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    instance: DiscreteBayesianOptimization object instance.
    n_acqs: Integer number of acquisitions to take from acquisition \
        function ac (the k in k-means).
    n_slice: integer number of slice samples (the data fed to k-means)
    n_warmup: number of times to randomly sample the aquisition function \
        for a_min
    n_iter: number of times to run scipy.minimize for a_min
    multiprocessing: number of cores for multiprocessing of scipy.minimize

    Returns
    -------
    List of the sampled means of the acquisition function.
    """
    assert n_slice >= n_acqs, "number of points in slice (n_slice) must be greater \
                             than number of centroids in k-means (n_acqs)"

    # Inialization from instance
    gp = instance._gp
    y_max = instance._space.target.max()
    bounds = instance._space.bounds
    steps = instance._space.steps
    random_state = instance._random_state
    slice = np.zeros((n_slice, bounds.shape[0]))
    constraint_dict = instance.get_constraint_dict()
    # Uses LBGFS for minding min (could be outside of constraints)
    lo = LocalOptimizer(ac, gp, y_max, bounds)

    # First finding minimum of acquisition function
    # Warm up with random points
    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_warmup, bounds.shape[0]))
    ys = ac(x_tries, gp=gp, y_max=y_max)
    a_min = ys.min()
    # Explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))
    if multiprocessing > 1:
        # Memory unconscious multiprocessing makes list of n_iter results...
        pool = Pool(multiprocessing)
        results = list(pool.imap_unordered(lo.minimizer, x_seeds))
        pool.close()
        pool.join()
        a_min = min(0, sorted(results, key=lambda x: x.fun[0])[0].fun[0])
        # Note: The algorithm needs a minimum l.e.q. 0. 
    else:
        for x_try in x_seeds:
            res = lo.minimizer(x_try)
            if not res.success:
                continue
            if a_min is None or res.fun[0] <= a_min:
                a_min = res.fun[0]

    # Initial sample over space
    invalid = True
    while invalid:
        s = instance.constrained_rng(1, bin=False)
        invalid = False
        for dict in constraint_dict:
            if dict['fun'](s.squeeze()) < 0: invalid = True
            # Slice aggregation
    start_time = time.time()
    for i in range(n_slice):
        u = random_state.uniform(a_min, ac(s, gp=gp, y_max=y_max))
        while True:
            invalid = True
            while invalid:
                s = instance.constrained_rng(1, bin=False)
                invalid = False
                for dict in constraint_dict:
                    if dict['fun'](s.squeeze()) < 0: invalid = True
            if ac(s, gp=gp, y_max=y_max) > u:
                slice[i] = s
                break
        if time.time() - start_time > 0.5 * TIMEOUT_TIME:
            raise TimeoutError("Failure in KMMBO optimizer. Slice aggregation is failing..."
                               " Check number of desired slices (n_slice)")

    # k-means
    start_time = time.time()
    unique = False
    i = 0
    while not unique:
        i += 1
        if i > 50: raise RuntimeError("KMBBO sampling cannot find unique new values after 50 attempts.")
        # Find centroids
        kmeans = KMeans(n_clusters=n_acqs,
                        random_state=random_state,
                        n_jobs=multiprocessing).fit(slice)

        # Make hashable, check for uniqueness, and assert length
        acqs = {}
        unique = True
        for i in range(n_acqs):
            if _hashable(instance.space._bin(kmeans.cluster_centers_[i, :])) in instance.space:
                unique = False
                break
            if _hashable(instance.space._bin(kmeans.cluster_centers_[i, :])) in instance.partner_space:
                unique = False
                break
            acqs[_hashable(instance.space._bin(kmeans.cluster_centers_[i, :]))] = i
        if len(acqs) != n_acqs:
            unique = False
        random_state = None
        if time.time() - start_time > 0.5 * TIMEOUT_TIME:
            raise TimeoutError("Failure in KMMBO optimizer. k-means clustering is failing..."
                               " Check number of desired slices (n_slice) and batch size")

    assert len(acqs) == n_acqs, "k-means clustering is not distinct in discretized space!"
    return [key for key in acqs.keys()]


def capitalist_worker(ucb_max, instance, n_warmup, n_iter, complements, procnums, utilities, market_sizes, out_q):
    """Worker function for multiprocessing"""
    outdict = {}
    for k in range(len(utilities)):
        outdict[procnums[k]] = ucb_max(ac=utilities[k].utility,
                                       instance=instance,
                                       n_acqs=market_sizes[k],
                                       n_warmup=n_warmup,
                                       n_iter=n_iter,
                                       multiprocessing=1,
                                       complements=complements
                                       )
    out_q.put(outdict)


def disc_capitalist_max(instance, exp_mean=1, n_splits=4, n_acqs=4, n_warmup=10000, n_iter=250, multiprocessing=1,
                        complements=False):
    """
    The capitalist acquisition function creates an unequal distribution of greed/wealth in sampling in parallel.
    A suite of Upper Confidence Bound (UCB) acquisition functions are created with hyperparameter lambda drawn
    from an exponential distribution. Multiple local maxima are then take from these acquisition functions.

    If the number of acquisitions do not divide evenly into the number of markets, the more greedy markets get used first

    Parallel Algorithm Configuration, F. Hutter and H. Hoos and K. Leyton-Brown, 55--70  (2012)
    Parameters
    ----------
    instance: DiscreteBayesianOptimization object instance.
    exp_mean: float, mean of exponential distribution funciton to draw from. A lower mean will create a more greedy market
    n_splits: int, number of
    n_acqs: int, number of acquisitions to take from acquisition function ac.
    n_warmup: int, number of times to randomly sample the aquisition function
    n_iter: int, number of times to run scipy.minimize
    multiprocessing: int, number of processes to use
    complements: bool, whether or not to consider complements

    Returns
    -------
    suggestions from set of acquisition functions: list of tuples
    """
    if instance.constraints:
        ucb_max = disc_constrained_acq_max
    else:
        ucb_max = disc_acq_max

    assert n_acqs >= n_splits, "Number of desired acquisitions from capitalist sampling must be larger than the" \
                               " number of market segments"

    ucb_params = np.sort(np.random.exponential(exp_mean, n_splits))
    utilities = []
    for param in ucb_params:
        utilities.append(UtilityFunction(kind='ucb', kappa=param, xi=0.0))

    market_sizes = [0 for _ in range(n_splits)]
    while sum(market_sizes) < n_acqs:
        for i in range(n_splits):
            if sum(market_sizes) < n_acqs:
                market_sizes[i] += 1
            else:
                break

    results = []
    start_time = time.time()
    while time.time() - start_time < 0.5 * TIMEOUT_TIME:
        if multiprocessing > 1:
            out_q = Queue()
            procs = []
            n_processes = min(multiprocessing, len(utilities))
            chunksize = int(np.ceil(len(utilities) / float(n_processes)))
            n_processes = int(np.ceil(len(utilities) / chunksize))  # For uneven splits
            for i in range(n_processes):
                p = Process(target=capitalist_worker,
                            args=(ucb_max,
                                  instance,
                                  n_warmup,
                                  n_iter,
                                  complements,
                                  range(chunksize * i, chunksize * (i + 1)),
                                  utilities[chunksize * i:chunksize * (i + 1)],
                                  market_sizes[chunksize * i:chunksize * (i + 1)],
                                  out_q))
                procs.append(p)
                p.start()
            resultsdict = {}
            for i in range(n_processes):
                resultsdict.update(out_q.get())
            for p in procs:
                p.join()
            trial_results = [item for sublist in resultsdict.values() for item in sublist]
            np.random.shuffle(trial_results)
            for trial in trial_results:
                if _hashable(trial) not in results:
                    results.append(_hashable(trial))
                    instance.partner_register(trial)
        else:
            for i in range(n_splits):
                trial_results = ucb_max(ac=utilities[i].utility,
                                        instance=instance,
                                        n_acqs=market_sizes[i],
                                        n_warmup=n_warmup,
                                        n_iter=n_iter,
                                        multiprocessing=multiprocessing,
                                        complements=complements)
                for trial in trial_results:
                    if _hashable(trial) not in results:
                        results.append(_hashable(trial))
                        instance.partner_register(trial)

        if len(results) >= n_acqs:
            results = results[:n_acqs]
            break
        else:
            print("Redundancies detected across capitalist markets. ",
                  "Running another market level loop...",
                  "\nTime at {:5.2f} minutes. Maximum set to {:5.2f} minutes. ".format((time.time() - start_time) / 60,
                                                                                       TIMEOUT_TIME * 0.5 / 60),
                  "Completed {} of {} acquisitions found".format(len(results),n_acqs))

    if len(results) < n_acqs:
        utility = UtilityFunction(kind='ucb')
        results.extend(ucb_max(ac=utility[i].utility,
                               instance=instance,
                               n_acqs=n_acqs - len(results),
                               n_warmup=n_warmup,
                               n_iter=n_iter,
                               multiprocessing=multiprocessing,
                               complements=complements))
    return results
