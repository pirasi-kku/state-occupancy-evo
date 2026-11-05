# -*- coding: utf-8 -*-
"""
soevo_constrained.py
====================
SO-EVO and EVO with pluggable constraint handling, used for the CEC 2020
engineering problems.

Only the comparison used for greedy acceptance changes between rules; the
operators, the occupancy sampling and update, and the evaluation count are
those of the unconstrained implementation.

    cst='penalty'      static penalty: greedy acceptance on
                       F(x) = f(x) + P * sum(max(0, g_i(x)))
    cst='feasibility'  Deb's feasibility rules: a feasible point beats an
                       infeasible one, feasible points are ranked by
                       objective, infeasible points by total violation
    cst='epsilon'      Deb's rules with violations below eps(t) counted as
                       feasible, eps decayed to zero over the first
                       `eps_frac` of the evaluations

The internal quantities of EVO (enrichment bound EB, stability level SL, best
and worst individuals) use the penalised scalar under every rule, so the EVO
machinery is unchanged; only acceptance follows the chosen rule.
"""

import numpy as np

from evo_core import _nearest_neighbour, _bh_clip


# ==============================================================================
#  Constraint handling
# ==============================================================================

def total_violation(con_fn, x):
    """sum of max(0, g_i(x)); 0.0 means feasible. Non-finite -> heavily infeasible."""
    if con_fn is None:
        return 0.0
    with np.errstate(all='ignore'):
        g = np.asarray(con_fn(x), dtype=float)
    if g.size == 0:
        return 0.0
    g = np.where(np.isfinite(g), g, 1e6)
    return float(np.sum(np.maximum(0.0, g)))


def accepts(mode, obj_new, vio_new, obj_cur, vio_cur, pen_new, pen_cur, eps=0.0):
    """
    True if the candidate should replace the incumbent.

    mode='penalty'      compare penalised scalars (static penalty)
    mode='feasibility'  Deb's rules
    mode='epsilon'      Deb's rules with violations <= eps counted as feasible
    """
    if mode == 'penalty':
        return pen_new < pen_cur

    if mode == 'epsilon':
        vio_new = 0.0 if vio_new <= eps else vio_new
        vio_cur = 0.0 if vio_cur <= eps else vio_cur

    fn_ok, fc_ok = (vio_new <= 0.0), (vio_cur <= 0.0)
    if fn_ok and fc_ok:
        return obj_new < obj_cur          # both feasible -> objective
    if fn_ok and not fc_ok:
        return True                       # feasible beats infeasible
    if not fn_ok and fc_ok:
        return False
    return vio_new < vio_cur              # both infeasible -> less violation


def _eps_schedule(FEs, max_FEs, eps0, eps_frac=0.4, cp=3.0):
    """eps decays from eps0 to 0 over the first eps_frac of the evaluations."""
    Tc = eps_frac * max_FEs
    if FEs >= Tc:
        return 0.0
    return float(eps0 * (1.0 - FEs / Tc) ** cp)


# ==============================================================================
#  SO-EVO with constraint handling
# ==============================================================================

def soevo_cst(obj_fn, con_fn, lb, ub, D, N=50, max_FEs=20000,
              ns=20, sampling='occupancy', credit='selective',
              eps_init=0.40, eps_end=0.01,
              cst='feasibility', penalty=1e6,
              seed=None, clip_lb=None, clip_ub=None):
    """
    sampling: 'occupancy' (the published method) or 'epsilon'
    (epsilon-greedy variant).

    Returns dict with best feasible/penalised results and diagnostics:
        best_obj    objective of the best FEASIBLE solution found (nan if none)
        best_pen    best penalised value f + penalty * violation
        feasible    True if any feasible solution was found
        best_vio    violation of the returned best point
        FEs         evaluations used
    """
    if seed is not None:
        np.random.seed(seed)
    clip_lb = lb if clip_lb is None else clip_lb
    clip_ub = ub if clip_ub is None else clip_ub

    # coefficient states and one occupancy vector per coefficient, as in soevo.py
    parts = np.array([(k + 1) / (ns + 1) for k in range(ns + 1)])
    Pa = np.ones(ns + 1); Pb = np.ones(ns + 1)
    Pg = np.ones(ns + 1); Pd = np.ones(ns + 1)

    def _sample_plain(occ):
        return int(np.random.choice(ns + 1, p=occ / occ.sum()))

    def _sample_eps(occ, e):
        if np.random.rand() < e:
            return int(np.random.randint(ns + 1))
        return int(np.random.choice(ns + 1, p=occ / occ.sum()))

    def evaluate(x):
        xc = np.clip(x, clip_lb, clip_ub)
        o = float(obj_fn(xc))
        v = total_violation(con_fn, xc)
        return o, v, o + penalty * v

    pop = lb + (ub - lb) * np.random.rand(N, D)
    obj = np.zeros(N); vio = np.zeros(N); pen = np.zeros(N)
    for i in range(N):
        obj[i], vio[i], pen[i] = evaluate(pop[i])
    FEs = N

    eps0 = float(np.median(vio)) if cst == 'epsilon' else 0.0

    while FEs < max_FEs:
        eps_now = _eps_schedule(FEs, max_FEs, eps0) if cst == 'epsilon' else 0.0

        if sampling == 'occupancy':
            e_greedy = None
        else:
            e_greedy = eps_init + (eps_end - eps_init) * (FEs / max_FEs)

        BS = pop[int(np.argmin(pen))]
        WS_val = float(np.max(pen)); BS_val = float(np.min(pen))
        EB = float(np.mean(pen))
        XCP = np.mean(pop, axis=0)

        for i in range(N):
            SLi = (pen[i] - BS_val) / (WS_val - BS_val + 1e-300)
            XNg = pop[_nearest_neighbour(pop, i)]

            if sampling == 'occupancy':
                ka = _sample_plain(Pa); a = parts[ka]
                kb = _sample_plain(Pb); b = parts[kb]
                kg = _sample_plain(Pg); g = parts[kg]
                kd = _sample_plain(Pd); d = parts[kd]
            else:
                ka = _sample_eps(Pa, e_greedy); a = parts[ka]
                kb = _sample_eps(Pb, e_greedy); b = parts[kb]
                kg = _sample_eps(Pg, e_greedy); g = parts[kg]
                kd = _sample_eps(Pd, e_greedy); d = parts[kd]

            if pen[i] > EB:
                SB = np.random.rand()
                if SLi > SB:
                    X1 = pop[i] + a * (BS - pop[i]) * SLi
                    X2 = pop[i] + b * (BS - pop[i]) + g * (XNg - pop[i])
                    u1, u2 = ('a',), ('b', 'g')
                else:
                    X1 = pop[i] + (a * BS - b * XCP) / (SLi + 1e-300)
                    X2 = pop[i] + g * (BS - pop[i]) - d * XNg
                    u1, u2 = ('a', 'b'), ('g', 'd')
                X1 = _bh_clip(X1, lb, ub); X2 = _bh_clip(X2, lb, ub)
                o1, v1, p1 = evaluate(X1)
                o2, v2, p2 = evaluate(X2)
                FEs += 2
                # keep the better candidate under the SAME rule used for acceptance
                if accepts(cst, o1, v1, o2, v2, p1, p2, eps_now):
                    Xn, on, vn, pn, used = X1, o1, v1, p1, u1
                else:
                    Xn, on, vn, pn, used = X2, o2, v2, p2, u2
            else:
                r = np.random.uniform(lb, ub, D)
                Xn = _bh_clip(pop[i] + d * r, lb, ub)
                on, vn, pn = evaluate(Xn); FEs += 1
                used = ('d',)

            if accepts(cst, on, vn, obj[i], vio[i], pn, pen[i], eps_now):
                pop[i] = Xn; obj[i] = on; vio[i] = vn; pen[i] = pn
                if credit == 'all':
                    Pa[ka] += 1; Pb[kb] += 1; Pg[kg] += 1; Pd[kd] += 1
                else:
                    if 'a' in used: Pa[ka] += 1
                    if 'b' in used: Pb[kb] += 1
                    if 'g' in used: Pg[kg] += 1
                    if 'd' in used: Pd[kd] += 1

            if FEs >= max_FEs:
                break

    feas = vio <= 0.0
    if feas.any():
        j = int(np.argmin(np.where(feas, obj, np.inf)))
        return dict(best_obj=float(obj[j]), best_pen=float(pen[j]),
                    feasible=True, best_vio=0.0, FEs=FEs)
    j = int(np.argmin(vio))
    return dict(best_obj=float('nan'), best_pen=float(np.min(pen)),
                feasible=False, best_vio=float(vio[j]), FEs=FEs)


# ==============================================================================
#  EVO with the same constraint handling (control)
# ==============================================================================

def evo_cst(obj_fn, con_fn, lb, ub, D, N=50, max_FEs=20000,
            cst='feasibility', penalty=1e6, seed=None,
            clip_lb=None, clip_ub=None):
    """
    EVO with the same constraint handling and return value as soevo_cst(), so
    the two differ only in their operators and coefficient sampling.
    """
    if seed is not None:
        np.random.seed(seed)
    clip_lb = lb if clip_lb is None else clip_lb
    clip_ub = ub if clip_ub is None else clip_ub

    def evaluate(x):
        xc = np.clip(x, clip_lb, clip_ub)
        o = float(obj_fn(xc))
        v = total_violation(con_fn, xc)
        return o, v, o + penalty * v

    pop = lb + (ub - lb) * np.random.rand(N, D)
    obj = np.zeros(N); vio = np.zeros(N); pen = np.zeros(N)
    for i in range(N):
        obj[i], vio[i], pen[i] = evaluate(pop[i])
    FEs = N
    eps0 = float(np.median(vio)) if cst == 'epsilon' else 0.0

    while FEs < max_FEs:
        eps_now = _eps_schedule(FEs, max_FEs, eps0) if cst == 'epsilon' else 0.0
        BS = pop[int(np.argmin(pen))]
        WS_val = float(np.max(pen)); BS_val = float(np.min(pen))
        EB = float(np.mean(pen)); XCP = np.mean(pop, axis=0)

        for i in range(N):
            SLi = (pen[i] - BS_val) / (WS_val - BS_val + 1e-300)
            XNg = pop[_nearest_neighbour(pop, i)]
            if pen[i] > EB:
                SB = np.random.rand()
                if SLi > SB:
                    ai1 = np.random.randint(1, D + 1)
                    ai2 = np.random.randint(1, ai1 + 1)
                    X1 = pop[i].copy(); X1[ai2 - 1:ai1] = BS[ai2 - 1:ai1]
                    gi1 = np.random.randint(1, D + 1)
                    gi2 = np.random.randint(1, gi1 + 1)
                    X2 = pop[i].copy(); X2[gi2 - 1:gi1] = XNg[gi2 - 1:gi1]
                else:
                    r1, r2 = np.random.rand(), np.random.rand()
                    X1 = pop[i] + (r1 * BS - r2 * XCP) / (SLi + 1e-300)
                    r3, r4 = np.random.rand(), np.random.rand()
                    X2 = pop[i] + r3 * BS - r4 * XNg
                X1 = _bh_clip(X1, lb, ub); X2 = _bh_clip(X2, lb, ub)
                o1, v1, p1 = evaluate(X1); o2, v2, p2 = evaluate(X2); FEs += 2
                if accepts(cst, o1, v1, o2, v2, p1, p2, eps_now):
                    Xn, on, vn, pn = X1, o1, v1, p1
                else:
                    Xn, on, vn, pn = X2, o2, v2, p2
            else:
                r = np.random.uniform(lb, ub, D)
                Xn = _bh_clip(pop[i] + r, lb, ub)
                on, vn, pn = evaluate(Xn); FEs += 1

            if accepts(cst, on, vn, obj[i], vio[i], pn, pen[i], eps_now):
                pop[i] = Xn; obj[i] = on; vio[i] = vn; pen[i] = pn
            if FEs >= max_FEs:
                break

    feas = vio <= 0.0
    if feas.any():
        j = int(np.argmin(np.where(feas, obj, np.inf)))
        return dict(best_obj=float(obj[j]), best_pen=float(pen[j]),
                    feasible=True, best_vio=0.0, FEs=FEs)
    j = int(np.argmin(vio))
    return dict(best_obj=float('nan'), best_pen=float(np.min(pen)),
                feasible=False, best_vio=float(vio[j]), FEs=FEs)
