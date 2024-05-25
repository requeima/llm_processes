"""
Copied from:
https://github.com/mwhoffman/benchfunk/blob/master/benchfunk/functions/classics.py
https://github.com/mwhoffman/benchfunk/blob/master/benchfunk/utils.py

Copyright (c) 2014, the benchfunk authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Benchmark problems which correspond to so-called "classic" problems from the
global-optimization literature. All problems are transformed into maximization
problems and can be combined with additional additive noise.
"""

import numpy as np

def rstate(rng=None):
    """
    Return a RandomState object. This is just a simple wrapper such that if rng
    is already an instance of RandomState it will be passed through, otherwise
    it will create a RandomState object using rng as its seed.
    """
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)
    return rng


def repr_args(obj, *args, **kwargs):
    """
    Return a repr string for an object with args and kwargs.
    """
    typename = type(obj).__name__
    args = ['{:s}'.format(str(val)) for val in args]
    kwargs = ['{:s}={:s}'.format(name, str(val))
              for name, val in kwargs.items()]
    return '{:s}({:s})'.format(typename, ', '.join(args + kwargs))

__all__ = ['Sinusoidal', 'Gramacy', 'Branin', 'Bohachevsky', 'Goldstein',
           'Hartmann3', 'Hartmann6']


class Benchmark(object):
    """
    Base class for benchmark functions which should be MAXIMIZED. Each class
    which implements this interface should include a method self._f(x) which
    takes an (n,d)-array of input locations and returns the value of each
    input.
    """
    def __init__(self, sn2=0.0, rng=None):
        self._sn2 = float(sn2)
        self._rng = rstate(rng)

    def __repr__(self):
        args = []
        kwargs = {}
        if self._sn2 > 0:
            args.append(self._sn2)
        return repr_args(self, *args, **kwargs)

    def __call__(self, x):
        return self.get(x)[0]

    def _f(self, X):
        """Vectorized access to the noise-free benchmark function. Note: this
        function does no bounds checking."""
        raise NotImplementedError

    def get(self, X):
        """Vectorized access to the benchmark function."""
        y = self.get_f(X)
        if self._sn2 > 0:
            y += self._rng.normal(scale=np.sqrt(self._sn2), size=len(y))
        return y

    def get_f(self, X):
        """Vectorized access to the noise-free benchmark function."""
        X = np.array(X, ndmin=2, dtype=float, copy=False)
        if X.shape != (X.shape[0], self.bounds.shape[0]):
            raise ValueError('function inputs must be {:d}-dimensional'
                             .format(self.ndim))
        return self._f(X)


class Sinusoidal(Benchmark):
    """
    Simple sinusoidal function bounded in [0, 2pi] given by -cos(x)-sin(3x).
    """
    bounds = np.array([[0, 2*np.pi]])
    xopt = np.array([3.61439678])
    ndim = 1

    def _f(self, x):
        return -np.ravel(np.cos(x) + np.sin(3*x))


class Gramacy(Benchmark):
    """
    Sinusoidal function in 1d used by Gramacy and Lee in "Cases for the nugget
    in modeling computer experiments".
    """
    bounds = np.array([[0.5, 2.5]])
    xopt = np.array([0.54856343])
    ndim = 1

    def _f(self, x):
        return -np.ravel(np.sin(10*np.pi*x) / (2*x) + (x-1)**4)


class Branin(Benchmark):
    """
    The 2d Branin function bounded in [-5,10] to [0,15]. Global optimizers
    exist at [-pi, 12.275], [pi, 2.275], and [9.42478, 2.475] with no local
    optimizers.
    """
    bounds = np.array([[-5, 10.], [0, 15]])
    xopt = np.array([np.pi, 2.275])
    ndim = 2

    def _f(self, x):
        y = (x[:, 1]-(5.1/(4*np.pi**2))*x[:, 0]**2+5*x[:, 0]/np.pi-6)**2
        y += 10*(1-1/(8*np.pi))*np.cos(x[:, 0])+10
        # NOTE: this rescales branin by 10 to make it more manageable.
        y /= 10.
        return -y


class Bohachevsky(Benchmark):
    """
    The Bohachevsky function in 2d, bounded in [-100, 100] for both variables.
    There is only one global optimizer at [0, 0].
    """
    bounds = np.array([[-100, 100.], [-100, 100]])
    xopt = np.array([0, 0.])
    ndim = 2

    def _f(self, x):
        y = 0.7 + x[:, 0]**2 + 2.0*x[:, 1]**2
        y -= 0.3*np.cos(3*np.pi*x[:, 0])
        y -= 0.4*np.cos(4*np.pi*x[:, 1])
        return -y


class Goldstein(Benchmark):
    """
    The Goldstein & Price function in 2d, bounded in [-2,-2] to [2,2]. There
    are several local optimizers and a single global optimizer at [0,-1].
    """
    bounds = np.array([[-2, 2.], [-2, 2]])
    xopt = np.array([0, -1.])
    ndim = 2

    def _f(self, x):
        a = (1 +
             (x[:, 0] + x[:, 1]+1)**2 *
             (19-14*x[:, 0] +
              3*x[:, 0]**2 - 14*x[:, 1] + 6*x[:, 0]*x[:, 1] + 3*x[:, 1]**2))
        b = (30 +
             (2*x[:, 0] - 3*x[:, 1])**2 *
             (18 - 32*x[:, 0] + 12*x[:, 0]**2 + 48*x[:, 1] - 36*x[:, 0]*x[:, 1]
              + 27*x[:, 1]**2))
        return -a * b


class Hartmann3(Benchmark):
    """
    The 3-dimensional Hartmann function bounded in [0,1]^3.
    """
    ndim = 3
    bounds = np.array(3 * [[0., 1.]])
    xopt = np.array([0.114614, 0.555649, 0.852547])

    def _f(self, x):
        a = np.array([[3.0, 10., 30.],
                      [0.1, 10., 35.],
                      [3.0, 10., 30.],
                      [0.1, 10., 35.]])[None]
        c = np.array([1., 1.2, 3., 3.2])[None]
        p = np.array([[0.36890, 0.11700, 0.26730],
                      [0.46990, 0.43870, 0.74700],
                      [0.10910, 0.87320, 0.55470],
                      [0.03815, 0.57430, 0.88280]])[None]
        return np.sum(c * np.exp(-np.sum(a * (x[:, None] - p)**2, -1)), -1)


class Hartmann6(Benchmark):
    """
    The 6-dimensional Hartmann function bounded in [0,1]^6.
    """
    ndim = 6
    bounds = np.array(6 * [[0., 1.]])
    xopt = np.array([0.201667, 0.150011, 0.476874,
                     0.275332, 0.311652, 0.6573])

    def _f(self, x):
        a = np.array([[10., 3.0, 17., 3.5, 1.7, 8.0],
                      [.05, 10., 17., 0.1, 8.0, 14.],
                      [3.0, 3.5, 1.7, 10., 17., 8.0],
                      [17., 8.0, .05, 10., 0.1, 14.]])[None]
        c = np.array([1.0, 1.2, 3.0, 3.2])[None]
        p = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                      [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                      [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                      [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])[None]
        F = np.sum(c * np.exp(-np.sum(a * (x[:, None] - p)**2, -1)), -1)
        # NOTE: this rescales the problem to have mean zero and variance 1, as
        # used in Picheny et al., 2012.
        F = (F + 2.58) / 1.94
        return F
    

function_map = {
    'Sinusoidal': Sinusoidal(),
    'Gramacy': Gramacy(),
    'Branin': Branin(),
    'Bohachevsky': Bohachevsky(),
    'Goldstein': Goldstein(),
    'Hartmann3': Hartmann3(),
    'Hartmann6': Hartmann6()   
}
