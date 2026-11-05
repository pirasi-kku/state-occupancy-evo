# -*- coding: utf-8 -*-
"""
constraint_handling.py
======================
Constraint-handling wrappers that turn a constrained problem into a scalar
objective, so the unconstrained algorithms in evo_core.py can be run on it
unchanged.

make_static_penalty      f(x) + C * sum(max(0, g(x))), C = 1e6 in the
                         manuscript
make_feasibility_ranked  a stationary scalarization that reproduces Deb's
                         feasibility rules (Deb 2000): f(x) when feasible,
                         BIG + v(x) otherwise, with BIG above any attainable
                         feasible objective
"""

import numpy as np

# Objectives in this suite span roughly 1e-12 to 1.6e8 (RC28 rolling bearing),
# so 1e12 sits far above any attainable feasible value while leaving ample
# headroom for the violation term.
BIG_DEFAULT = 1e12


def total_violation(con_fn, x):
    """Total constraint violation, sum(max(0, g_i(x))). Zero means feasible."""
    g = np.atleast_1d(np.asarray(con_fn(x), dtype=float))
    return float(np.sum(np.maximum(0.0, g)))


def make_feasibility_ranked(obj_fn, con_fn, big=BIG_DEFAULT):
    """Deb's feasibility rules as a stationary scalar score (lower is better)."""
    def scored(x):
        v = total_violation(con_fn, x)
        if v <= 0.0:
            return float(obj_fn(x))
        return big + v
    return scored


def make_static_penalty(obj_fn, con_fn, coeff=1e6):
    """Static penalty: f(x) + coeff * total violation."""
    def scored(x):
        v = total_violation(con_fn, x)
        return float(obj_fn(x)) + coeff * v
    return scored


def is_feasible_score(score, big=BIG_DEFAULT):
    """True when a feasibility-ranked score corresponds to a feasible solution."""
    return float(score) < big


SCHEMES = {
    'penalty':     make_static_penalty,
    'feasibility': make_feasibility_ranked,
}
