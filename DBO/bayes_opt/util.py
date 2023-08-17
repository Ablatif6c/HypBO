"""
Inspired by and borrowed from:
    https://github.com/fmfn/BayesianOptimization
"""
import warnings
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def _hashable(x):
    """ ensure that an point is hashable by a python dict """
    res = tuple(x)
    return res


def maximize_acq(
        x_seeds,
        ac,
        gp,
        instance,
        y_max,
        max_acq,
        bounds,
        ):
    results = []
    for x_try in x_seeds:
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method="L-BFGS-B")

        # See if success
        if not res.success:
            continue

        out = None
        if type(res.fun) is float:
            out = res.fun
        else:
            out = res.fun[0]

        # Make sure the x tries are new points
        point = np.clip(res.x, bounds[:, 0], bounds[:, 1])
        point = _hashable(point)

        breakout = False
        for dict in instance.get_constraint_dict():
            if dict['fun'](point) < 0:
                breakout = True
        if breakout:
            continue
        if point in instance.space:
            continue
        else:
            if max_acq is None or -out >= max_acq:
                # x_max = np.clip(res.x, bounds[:, 0], bounds[:, 1])
                max_acq = -out
                results.append((np.array(point), max_acq))

    return results


def acq_max(
        instance,
        ac,
        gp,
        y_max,
        bounds,
        random_state,
        n_warmup=1000,
        n_iter=250,
        n=1,
        multiprocessing: int = 1,):
    """
    A function to find the maximum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (250) random starting points.

    Parameters
    ----------
    :param ac:
        The acquisition function object that return its point-wise value.

    :param gp:
        A gaussian process fitted to the relevant data.

    :param y_max:
        The current maximum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq max.

    :param random_state:
        instance of np.RandomState random number generator

    :param n_warmup:
        number of times to randomly sample the aquisition function

    :param n_iter:
        number of times to run scipy.minimize

    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """

    # Warm up with random points
    x_tries = random_state.uniform(
                                    bounds[:, 0],
                                    bounds[:, 1],
                                    size=(n_warmup, bounds.shape[0]))

    # Make sure the x tries are new points
    _x_tries = []
    breakout = True
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
        else:
            _x_tries.append(x_tries[idx, :])
    x_tries = np.array(_x_tries)

    ys = ac(x_tries, gp=gp, y_max=y_max)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    # Keep the n best samples
    x_tries = [x for _, x in sorted(
                                    zip(ys, x_tries),
                                    key=lambda pair: pair[0],
                                    reverse=True,)]
    ys = np.array(sorted(ys, reverse=True))
    x = x_tries[:n]
    y = ys[:n]

    # Explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))

    # Find the minimum of minus the acquisition function
    with Pool(processes=multiprocessing) as pool:
        # Give to each process a chunk of x_seeds
        x_seeds_chunks = np.array_split(x_seeds, multiprocessing)
        args = [(
                x_seeds_chunks[i],
                ac,
                gp,
                instance,
                y_max,
                max_acq,
                bounds,
                ) for i in range(multiprocessing)]
        results = pool.starmap(
            maximize_acq,
            args,
            )
        results = [r for r in results if len(r) > 0]

        if len(results) > 0:
            results = list(np.concatenate(results))
            xy = np.array(list(zip(x, y)))
            results.extend(xy)
            # remove duplicates
            results = np.array(
                list(OrderedDict((tuple(s[0]), s) for s in results).values()))
            x = [_x for _x, _ in sorted(
                                        results,
                                        key=lambda pair: pair[1],
                                        reverse=True,)]

    return x[:n]


class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, xi):
        """
        If UCB is to be used, a constant kappa is needed.
        """
        self.kappa = kappa

        self.xi = xi

        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x, gp, y_max):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)

    @staticmethod
    def _ucb(x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi) / std
        return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi) / std
        return norm.cdf(z)


def load_logs(optimizer, logs):
    """Load previous ...

    """
    import json

    if isinstance(logs, str):
        logs = [logs]

    for log in logs:
        with open(log, "r") as j:
            while True:
                try:
                    iteration = next(j)
                except StopIteration:
                    break

                iteration = json.loads(iteration)
                try:
                    optimizer.register(
                        params=iteration["params"],
                        target=iteration["target"],
                    )
                except KeyError:
                    pass

    return optimizer


def ensure_rng(random_state=None):
    """
    Creates a random number generator based on an optional seed.  This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state


class Colours:
    """Print in nice colours."""

    # BLUE = '\033[94m'
    # BOLD = '\033[1m'
    # CYAN = '\033[96m'
    # DARKCYAN = '\033[36m'
    # END = '\033[0m'
    # GREEN = '\033[92m'
    # PURPLE = '\033[95m'
    # RED = '\033[91m'
    # UNDERLINE = '\033[4m'
    # YELLOW = '\033[93m'
    BLUE = ''
    BOLD = ''
    CYAN = ''
    DARKCYAN = ''
    END = ''
    GREEN = ''
    PURPLE = ''
    RED = ''
    UNDERLINE = ''
    YELLOW = ''

    @classmethod
    def _wrap_colour(cls, s, colour):
        return colour + s + cls.END

    @classmethod
    def black(cls, s):
        """Wrap text in black."""
        return cls._wrap_colour(s, cls.END)

    @classmethod
    def blue(cls, s):
        """Wrap text in blue."""
        return cls._wrap_colour(s, cls.BLUE)

    @classmethod
    def bold(cls, s):
        """Wrap text in bold."""
        return cls._wrap_colour(s, cls.BOLD)

    @classmethod
    def cyan(cls, s):
        """Wrap text in cyan."""
        return cls._wrap_colour(s, cls.CYAN)

    @classmethod
    def darkcyan(cls, s):
        """Wrap text in darkcyan."""
        return cls._wrap_colour(s, cls.DARKCYAN)

    @classmethod
    def green(cls, s):
        """Wrap text in green."""
        return cls._wrap_colour(s, cls.GREEN)

    @classmethod
    def purple(cls, s):
        """Wrap text in purple."""
        return cls._wrap_colour(s, cls.PURPLE)

    @classmethod
    def red(cls, s):
        """Wrap text in red."""
        return cls._wrap_colour(s, cls.RED)

    @classmethod
    def underline(cls, s):
        """Wrap text in underline."""
        return cls._wrap_colour(s, cls.UNDERLINE)

    @classmethod
    def yellow(cls, s):
        """Wrap text in yellow."""
        return cls._wrap_colour(s, cls.YELLOW)


'''
Bits related to sampling under constraints.
'''


def get_rnd_simplex(dimension, random_state):
    '''
    uniform point on a simplex, i.e. x_i >= 0 and sum of the coordinates is 1.
    Donald B. Rubin, The Bayesian bootstrap Ann. Statist. 9, 1981, 130-134.
    https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    '''
    t = random_state.uniform(0, 1, dimension - 1)
    t = np.append(t, [0, 1])
    t.sort()

    return np.array([(t[i + 1] - t[i]) for i in range(len(t) - 1)])


def get_rng_complement(dimension, random_state):
    """
    This samples a unit simplex uniformly, takes the first value and scales it
    to center about the complement range.
    Parameters
    ----------
    dimension
    random_state

    Returns
    -------
    float, [0,1]
    """
    res = 0.5 + get_rnd_simplex(
                    dimension,
                    random_state)[0] * [-0.5, 0.5][np.random.randint(2)]
    return res


def get_rnd_quantities(max_amount, dimension, random_state):
    '''
    Get an array of quantities x_i>=0 which sum up to max_amount at most.

    This samples a unit simplex uniformly, then scales the sampling by a value
    m between  0 and max amount, with a probability proportionate to the volume
    of a regular simplex with vector length m.
    '''
    r = random_state.uniform(0, 1)
    m = (r * (max_amount ** (dimension + 1))) ** (1 / (dimension + 1))
    return get_rnd_simplex(dimension, random_state) * m


if __name__ == "__main__":
    for i in range(10):
        a = get_rnd_quantities(5.0, 9, np.random.RandomState())
        print(np.sum(a), np.min(a), np.max(a), sep='...')
