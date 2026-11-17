# -*- coding: utf-8 -*-
"""
soevo.py
========
SO-EVO: State-Occupancy-Guided Energy Valley Optimizer.

  sampling = 'occupancy'  each coefficient is drawn from its occupancy vector,
                          in proportion to the occupancy values (SO-EVO)
           = 'epsilon'    epsilon-greedy: with probability eps a uniform draw
                          over the states, otherwise the occupancy draw; eps
                          decays linearly from eps_init to eps_end
           = 'uniform'    no occupancy vectors: the four coefficients are
                          drawn from U(0,1) as in EVO, with the SO-EVO
                          operators (EVO with the rewritten operators)

  credit   = 'selective'  an accepted candidate increments only the occupancy
                          vectors of the coefficients used to generate it
           = 'all'        an accepted candidate increments all four vectors

Coefficients used by each candidate:

  move          condition       X1             X2
  ------------  --------------  -------------  -------------
  alpha/gamma   SL_i >  SB      alpha          beta, gamma
  beta          SL_i <= SB      alpha, beta    gamma, delta
  positron      f(X_i) <= EB    delta          --
"""

import numpy as np

from evo_core import _nearest_neighbour, _bh_clip

# BS best solution, WS worst solution, EB enrichment bound, SL stability
# level, SB stability bound, XNg nearest neighbour, XCP population centroid,
# NEL objective values. Pa, Pb, Pg, Pd are the occupancy vectors of alpha,
# beta, gamma, delta.


# ==============================================================================
#  Core
# ==============================================================================

def soevo(func, lb, ub, D, N=50, max_FEs=150000, VTR=1e-12,
          ns=20, sampling='occupancy', credit='selective',
          eps_init=0.40, eps_end=0.01,
          seed=None, record_curve=False):
    """
    Returns: (nfe_to_vtr_or_None, best_fitness, curve)
    """
    if sampling not in ('occupancy', 'epsilon', 'uniform'):
        raise ValueError("sampling must be 'occupancy', 'epsilon' or 'uniform', "
                         f"got {sampling!r}")
    if credit not in ('all', 'selective'):
        raise ValueError(f"credit must be 'all' or 'selective', got {credit!r}")

    if seed is not None:
        np.random.seed(seed)

    # states c_k = (k+1)/(ns+1), k = 0..ns; occupancy vectors start at one
    parts = np.array([(k + 1) / (ns + 1) for k in range(ns + 1)])
    Pa = np.ones(ns + 1); Pb = np.ones(ns + 1)
    Pg = np.ones(ns + 1); Pd = np.ones(ns + 1)

    def _sample_plain(occ):
        prob = occ / occ.sum()
        return int(np.random.choice(ns + 1, p=prob))

    def _sample_eps(occ, eps):
        if np.random.rand() < eps:
            return int(np.random.randint(ns + 1))
        prob = occ / occ.sum()
        return int(np.random.choice(ns + 1, p=prob))

    pop     = lb + (ub - lb) * np.random.rand(N, D)
    fitness = np.array([func(pop[i]) for i in range(N)])
    FEs     = N
    curve   = [float(np.min(fitness))] if record_curve else None

    while FEs < max_FEs:
        if sampling == 'epsilon':
            eps = eps_init + (eps_end - eps_init) * (FEs / max_FEs)
        else:
            eps = None

        BS_idx = int(np.argmin(fitness)); BS = pop[BS_idx]
        WS_val = float(np.max(fitness));  BS_val = float(np.min(fitness))
        NEL    = fitness.copy()
        EB     = float(np.mean(NEL))
        XCP    = np.mean(pop, axis=0)

        for i in range(N):
            SLi = (NEL[i] - BS_val) / (WS_val - BS_val + 1e-300)
            XNg = pop[_nearest_neighbour(pop, i)]

            if sampling == 'uniform':
                ka = kb = kg = kd = None
                a, b, g, d = np.random.rand(4)
            elif sampling == 'occupancy':
                ka = _sample_plain(Pa); a = parts[ka]
                kb = _sample_plain(Pb); b = parts[kb]
                kg = _sample_plain(Pg); g = parts[kg]
                kd = _sample_plain(Pd); d = parts[kd]
            else:
                ka = _sample_eps(Pa, eps); a = parts[ka]
                kb = _sample_eps(Pb, eps); b = parts[kb]
                kg = _sample_eps(Pg, eps); g = parts[kg]
                kd = _sample_eps(Pd, eps); d = parts[kd]

            if NEL[i] > EB:
                SB = np.random.rand()
                if SLi > SB:
                    Xn1 = pop[i] + a * (BS - pop[i]) * SLi
                    Xn2 = pop[i] + b * (BS - pop[i]) + g * (XNg - pop[i])
                    used1 = ('a',)
                    used2 = ('b', 'g')
                else:
                    Xn1 = pop[i] + (a * BS - b * XCP) / (SLi + 1e-300)
                    Xn2 = pop[i] + g * (BS - pop[i]) - d * XNg
                    used1 = ('a', 'b')
                    used2 = ('g', 'd')
                Xn1 = _bh_clip(Xn1, lb, ub)
                Xn2 = _bh_clip(Xn2, lb, ub)
                f1 = func(Xn1); f2 = func(Xn2); FEs += 2
                if f1 <= f2:
                    Xbest, fbest, used = Xn1, f1, used1
                else:
                    Xbest, fbest, used = Xn2, f2, used2
            else:
                r = np.random.uniform(lb, ub, D)
                Xbest = _bh_clip(pop[i] + d * r, lb, ub)
                fbest = func(Xbest); FEs += 1
                used  = ('d',)

            if fbest < fitness[i]:
                pop[i] = Xbest; fitness[i] = fbest
                if sampling == 'uniform':
                    pass
                elif credit == 'all':
                    Pa[ka] += 1; Pb[kb] += 1; Pg[kg] += 1; Pd[kd] += 1
                else:
                    if 'a' in used: Pa[ka] += 1
                    if 'b' in used: Pb[kb] += 1
                    if 'g' in used: Pg[kg] += 1
                    if 'd' in used: Pd[kd] += 1

            if FEs >= max_FEs:
                break

        bv = float(np.min(fitness))
        if bv <= VTR:
            return FEs, bv, curve
        if record_curve:
            curve.append(bv)

    return None, float(np.min(fitness)), curve


# ==============================================================================
#  Arm registry
# ==============================================================================

ARMS = {
    'SOEVO_sel':     dict(sampling='occupancy', credit='selective'),
    'SOEVO_all':     dict(sampling='occupancy', credit='all'),
    'SOEVO_eps_sel': dict(sampling='epsilon',   credit='selective'),
    'SOEVO_eps_all': dict(sampling='epsilon',   credit='all'),
    'OPS_uniform':   dict(sampling='uniform',   credit='all'),
}

ARM_LABEL = {
    'SOEVO_sel':     'SO-EVO',
    'SOEVO_all':     'SO-EVO crediting all four',
    'SOEVO_eps_sel': 'selective, epsilon-greedy',
    'SOEVO_eps_all': 'all four, epsilon-greedy',
    'OPS_uniform':   'EVO with the rewritten operators',
}

ARM_COLORS = {
    'SOEVO_sel':     '#E63946',
    'SOEVO_all':     '#F4A261',
    'SOEVO_eps_sel': '#06D6A0',
    'SOEVO_eps_all': '#F4A261',
    'OPS_uniform':   '#8D99AE',
    'EVO':           '#1D6FA4',
}


def make_arm(arm_name):
    """Return a run_trial-compatible callable for the named arm."""
    cfg = ARMS[arm_name]

    def _fn(func, lb, ub, D, N=50, max_FEs=150000, VTR=1e-12,
            seed=None, record_curve=False, ns=20,
            eps_init=0.40, eps_end=0.01, **kw):
        return soevo(func, lb, ub, D, N=N, max_FEs=max_FEs, VTR=VTR, ns=ns,
                     sampling=cfg['sampling'], credit=cfg['credit'],
                     eps_init=eps_init, eps_end=eps_end,
                     seed=seed, record_curve=record_curve)
    _fn.__name__ = f'arm_{arm_name}'
    return _fn


ARM_FN = {name: make_arm(name) for name in ARMS}
