# -*- coding: utf-8 -*-
"""
evo_core.py
===========
Benchmark functions, the base Energy Valley Optimizer (EVO), the six comparison
algorithms (ACO-R, HS, FA, MVO, ISA, CSA) and the small runner utilities shared
by the SO-EVO experiments.

Every algorithm has the signature

    algo(func, lb, ub, D, N=50, max_FEs=150000, VTR=..., seed=None,
         record_curve=False, ...)  ->  (nfe_to_vtr_or_None, best_fitness, curve)

and seeds NumPy's global generator from `seed`, so a run is reproducible from
its seed alone.
"""

import os
import numpy as np

# ==============================================================================
#  Benchmark functions F01-F20 (the benchmark of the original EVO study)
# ==============================================================================
# Global minimum f* = 0 except F04 (-1), F05 (-(D-1)), F18 (1) and F20 (-1).
# Use optimum(fname, D), which also covers the dimension-dependent F05.

def evo_f1(x):   # Ackley 1
    x = np.asarray(x, dtype=float); D = x.size
    return float(-20*np.exp(-0.2*np.sqrt(np.sum(x**2)/D))
                 - np.exp(np.sum(np.cos(2*np.pi*x))/D) + 20 + np.e)

def evo_f2(x):   # Alpine 1
    x = np.asarray(x, dtype=float)
    return float(np.sum(np.abs(x*np.sin(x) + 0.1*x)))

def evo_f3(x):   # Chung Reynolds
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2)**2)

def evo_f4(x):   # Exponential, minimum -1 at x = 0
    x = np.asarray(x, dtype=float)
    return float(-np.exp(-0.5*np.sum(x**2)))

def evo_f5(x):   # Inverted cosine wave, minimum -(D-1) at x = 0
    x = np.asarray(x, dtype=float)
    s = 0.0
    for i in range(x.size - 1):
        s += np.exp(-(x[i]**2 + x[i+1]**2 + 0.5*x[i]*x[i+1])/8.0) \
             * np.cos(4*np.sqrt(x[i]**2 + x[i+1]**2 + 0.5*x[i]*x[i+1]))
    return float(-s)

def evo_f6(x):   # Pinter
    x = np.asarray(x, dtype=float); D = x.size
    i  = np.arange(1, D+1, dtype=float)
    xp = np.roll(x,  1)   # x_{i-1}
    xn = np.roll(x, -1)   # x_{i+1}
    A  = xp*np.sin(x) + np.sin(xn)
    B  = xp**2 - 2*x + 3*xn - np.cos(x) + 1
    return float(np.sum(i*x**2)
                 + np.sum(20*i*np.sin(A)**2)
                 + np.sum(i*np.log10(1 + i*B**2)))

def evo_f7(x):   # Rastrigin
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10))

def evo_f8(x):   # Salomon
    x = np.asarray(x, dtype=float)
    r = np.sqrt(np.sum(x**2))
    return float(1 - np.cos(2*np.pi*r) + 0.1*r)

def evo_f9(x):   # Schwefel 1.2
    x = np.asarray(x, dtype=float)
    return float(np.sum(np.cumsum(x)**2))

def evo_f10(x):  # Schwefel 2.21
    x = np.asarray(x, dtype=float)
    return float(np.max(np.abs(x)))

def evo_f11(x):  # Griewank
    x = np.asarray(x, dtype=float); i = np.arange(1, x.size+1, dtype=float)
    return float(np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(i))) + 1)

def evo_f12(x):  # Powell Singular 2
    x = np.asarray(x, dtype=float)
    D = x.size; s = 0.0
    for i in range(0, D-3, 4):
        s += (x[i]+10*x[i+1])**2 + 5*(x[i+2]-x[i+3])**2 \
             + (x[i+1]-2*x[i+2])**4 + 10*(x[i]-x[i+3])**4
    return float(s)

def evo_f13(x):  # Schumer Steiglitz
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**4))

def evo_f14(x):  # Schwefel 2.4
    x = np.asarray(x, dtype=float); D = x.size
    i = np.arange(1, D+1, dtype=float)
    return float(np.sum((x - i)**4))

def evo_f15(x):  # Schwefel 2.25
    x = np.asarray(x, dtype=float)
    return float(np.sum((x[1:] - x[:-1]**2)**2) + (x[0]-1)**2)

def evo_f16(x):  # Sphere
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2))

def evo_f17(x):  # Step 3
    x = np.asarray(x, dtype=float)
    return float(np.sum(np.floor(np.abs(x))))

def evo_f18(x):  # Trigonometric 2, minimum 1 at x = 0.9
    x = np.asarray(x, dtype=float); D = x.size
    s = 0.0
    for i in range(D):
        s += 8*np.sin(7*(x[i]-0.9)**2)**2 \
             + 6*np.sin(14*(x[i]-0.9)**2)**2 \
             + (x[i]-0.9)**2
    return float(1 + s)

def evo_f19(x):  # Wavy
    x = np.asarray(x, dtype=float); D = x.size
    k = 10
    return float(1 - (1/D)*np.sum(np.cos(k*x)*np.exp(-x**2/2)))

def evo_f20(x):  # Xin-She Yang 6, minimum -1 at x = 0
    x = np.asarray(x, dtype=float)
    return float((np.sum(np.sin(x)**2) - np.exp(-np.sum(x**2)))
                 * np.exp(-np.sum(np.sin(np.sqrt(np.abs(x)))**2)))

# Function registry. `dim_group` is the dimension at which the original EVO
# study ran the function (F01-F10 at 50, F11-F20 at 100). The manuscript runs
# every function at D = 5, 10, 30, 50 and 100. Powell Singular 2 (F12) needs a
# dimension divisible by four; get_valid_dim() rounds up to the next one.

EVO_BENCHMARK = {
    'F01_Ackley1':         {'func': evo_f1,  'lb': -32,     'ub':  32,    'opt':  0.0,  'dim_group': 50,  'type': 'Multimodal'},
    'F02_Alpine1':         {'func': evo_f2,  'lb': -10,     'ub':  10,    'opt':  0.0,  'dim_group': 50,  'type': 'Multimodal'},
    'F03_ChungReynolds':   {'func': evo_f3,  'lb': -100,    'ub': 100,    'opt':  0.0,  'dim_group': 50,  'type': 'Unimodal'},
    'F04_Exponential':     {'func': evo_f4,  'lb': -1,      'ub':  1,     'opt': -1.0,  'dim_group': 50,  'type': 'Unimodal'},
    'F05_InvCosWave':      {'func': evo_f5,  'lb': -5,      'ub':  5,     'opt': None,  'dim_group': 50,  'type': 'Multimodal'},   # f* = -(D-1)
    'F06_Pinter':          {'func': evo_f6,  'lb': -10,     'ub':  10,    'opt':  0.0,  'dim_group': 50,  'type': 'Multimodal'},
    'F07_Rastrigin':       {'func': evo_f7,  'lb': -5.12,   'ub':  5.12,  'opt':  0.0,  'dim_group': 50,  'type': 'Multimodal'},
    'F08_Salomon':         {'func': evo_f8,  'lb': -100,    'ub': 100,    'opt':  0.0,  'dim_group': 50,  'type': 'Multimodal'},
    'F09_Schwefel12':      {'func': evo_f9,  'lb': -100,    'ub': 100,    'opt':  0.0,  'dim_group': 50,  'type': 'Unimodal'},
    'F10_Schwefel221':     {'func': evo_f10, 'lb': -100,    'ub': 100,    'opt':  0.0,  'dim_group': 50,  'type': 'Unimodal'},
    'F11_Griewank':        {'func': evo_f11, 'lb': -600,    'ub': 600,    'opt':  0.0,  'dim_group': 100, 'type': 'Multimodal'},
    'F12_PowellSingular2': {'func': evo_f12, 'lb': -4,      'ub':  5,     'opt':  0.0,  'dim_group': 100, 'type': 'Unimodal',   'req_div4': True},
    'F13_SchumerSteiglitz':{'func': evo_f13, 'lb': -100,    'ub': 100,    'opt':  0.0,  'dim_group': 100, 'type': 'Unimodal'},
    'F14_Schwefel24':      {'func': evo_f14, 'lb': -100,    'ub': 100,    'opt':  0.0,  'dim_group': 100, 'type': 'Unimodal'},
    'F15_Schwefel225':     {'func': evo_f15, 'lb': -100,    'ub': 100,    'opt':  0.0,  'dim_group': 100, 'type': 'Unimodal'},
    'F16_Sphere':          {'func': evo_f16, 'lb': -100,    'ub': 100,    'opt':  0.0,  'dim_group': 100, 'type': 'Unimodal'},
    'F17_Step3':           {'func': evo_f17, 'lb': -100,    'ub': 100,    'opt':  0.0,  'dim_group': 100, 'type': 'Unimodal'},
    'F18_Trigonometric2':  {'func': evo_f18, 'lb': -500,    'ub': 500,    'opt':  1.0,  'dim_group': 100, 'type': 'Multimodal'},
    'F19_Wavy':            {'func': evo_f19, 'lb': -np.pi,  'ub': np.pi,  'opt':  0.0,  'dim_group': 100, 'type': 'Multimodal'},
    'F20_XinSheYang6':     {'func': evo_f20, 'lb': -2*np.pi,'ub': 2*np.pi,'opt': -1.0,  'dim_group': 100, 'type': 'Multimodal'},
}


def get_valid_dim(fname, d):
    """
    Return the actual dimension to use for (fname, d).
    F12_PowellSingular2 requires D divisible by 4:
      round up to nearest multiple of 4 (e.g. 5→8, 10→12, 30→32, 50→52, 100→100).
    All other functions: return d unchanged.
    """
    info = EVO_BENCHMARK[fname]
    if info.get('req_div4', False):
        remainder = d % 4
        return d if remainder == 0 else d + (4 - remainder)
    return d


def get_dim_list(fname):
    """
    Dimensions of the original EVO study's grid for `fname`: up to 50 for
    F01-F10 and up to 100 for F11-F20, each passed through get_valid_dim().
    The manuscript additionally runs F01-F10 at D = 100.
    """
    info     = EVO_BENCHMARK[fname]
    original = info['dim_group']          # 50 or 100
    if original == 50:
        base = [d for d in DIM_LIST if d <= 50]
        if 50 not in base: base = base + [50]
    else:
        base = list(DIM_LIST)
        if 100 not in base: base = base + [100]
    return sorted({get_valid_dim(fname, d) for d in base})


# Dimensions used by get_dim_list()
DIM_LIST = [5, 10, 30, 50, 100]


# ==============================================================================
#  Boundary handling and nearest neighbour
# ==============================================================================

def _bh_clip(v, lb, ub):
    """Clip every coordinate to [lb, ub]; used by all algorithms here."""
    return np.clip(v, lb, ub)


def _nearest_neighbour(pop, i):
    """Return index of the nearest particle to pop[i] (Euclidean distance)."""
    diffs = pop - pop[i]
    dists = np.sum(diffs**2, axis=1)
    dists[i] = np.inf
    return int(np.argmin(dists))


# ==============================================================================
#  EVO (Azizi et al. 2023)
# ==============================================================================

def evo(func, lb, ub, D, N=50, max_FEs=150000, VTR=1e-12,
        seed=None, record_curve=False):
    """
    Energy Valley Optimizer, following Eqs. (5)-(11) of Azizi et al. (2023),
    Scientific Reports 13:226.
    Returns: (nfe_to_vtr_or_None, best_fitness, curve)
    """
    if seed is not None: np.random.seed(seed)
    pop     = lb + (ub - lb) * np.random.rand(N, D)
    fitness = np.array([func(pop[i]) for i in range(N)])
    FEs     = N
    curve   = [float(np.min(fitness))] if record_curve else None

    while FEs < max_FEs:
        BS_idx = int(np.argmin(fitness)); BS = pop[BS_idx]
        WS_val = float(np.max(fitness));  BS_val = float(np.min(fitness))
        NEL    = fitness.copy()
        EB     = float(np.mean(NEL))

        for i in range(N):
            SLi   = (NEL[i] - BS_val) / (WS_val - BS_val + 1e-300)
            ng_i  = _nearest_neighbour(pop, i)
            XNg   = pop[ng_i]
            XCP   = np.mean(pop, axis=0)

            if NEL[i] > EB:
                SB = np.random.rand()
                if SLi > SB:
                    # Alpha decay (Eq. 5): copy a random block of coordinates from BS
                    alpha_idx_I  = np.random.randint(1, D+1)
                    alpha_idx_II = np.random.randint(1, alpha_idx_I+1)
                    Xnew1 = pop[i].copy()
                    Xnew1[alpha_idx_II-1:alpha_idx_I] = BS[alpha_idx_II-1:alpha_idx_I]
                    # Gamma decay (Eq. 7): copy a random block of coordinates from XNg
                    gamma_idx_I  = np.random.randint(1, D+1)
                    gamma_idx_II = np.random.randint(1, gamma_idx_I+1)
                    Xnew2 = pop[i].copy()
                    Xnew2[gamma_idx_II-1:gamma_idx_I] = XNg[gamma_idx_II-1:gamma_idx_I]
                else:
                    # Beta decay (Eqs. 9 and 10)
                    r1, r2 = np.random.rand(), np.random.rand()
                    Xnew1 = pop[i] + (r1*BS - r2*XCP) / (SLi + 1e-300)
                    r3, r4 = np.random.rand(), np.random.rand()
                    Xnew2 = pop[i] + r3*BS - r4*XNg

                Xnew1 = _bh_clip(Xnew1, lb, ub)
                Xnew2 = _bh_clip(Xnew2, lb, ub)
                f1v   = func(Xnew1); f2v = func(Xnew2); FEs += 2
                Xbest_new, fbest_new = (Xnew1, f1v) if f1v <= f2v else (Xnew2, f2v)
            else:
                # Positron emission (Eq. 11)
                r     = np.random.uniform(lb, ub, D)
                Xbest_new = _bh_clip(pop[i] + r, lb, ub)
                fbest_new = func(Xbest_new); FEs += 1

            if fbest_new < fitness[i]:
                pop[i] = Xbest_new; fitness[i] = fbest_new

            if FEs >= max_FEs: break

        best_val = float(np.min(fitness))
        if best_val <= VTR:
            return FEs, best_val, curve
        if record_curve:
            curve.append(best_val)

    return None, float(np.min(fitness)), curve

# ==============================================================================
#  Comparison algorithms: ACO-R, HS, FA, MVO, ISA, CSA
# ==============================================================================
# Compact implementations of the core update rules of each method, with the
# control parameters listed in Table 3 of the manuscript. All share the
# (nfe_to_vtr_or_None, best_fitness, curve) return signature.

def aco_r(func, lb, ub, D, N=50, max_FEs=150000, VTR=1e-12,
          seed=None, record_curve=False, archive_size=None):
    """
    ACO-R, ant colony optimization for continuous domains (Socha & Dorigo 2008).
    Keeps an archive of the best solutions and samples new ones from a
    Gaussian kernel around a rank-weighted archive member.
    """
    if seed is not None: np.random.seed(seed)
    if archive_size is None: archive_size = N
    q = 1e-4; xi = 0.85
    arch = lb + (ub - lb) * np.random.rand(archive_size, D)
    farch = np.array([func(arch[k]) for k in range(archive_size)])
    FEs = archive_size
    order = np.argsort(farch); arch = arch[order]; farch = farch[order]
    curve = [float(farch[0])] if record_curve else None

    weights = np.array([np.exp(-((k)**2)/(2*(q*archive_size)**2))
                        for k in range(archive_size)], dtype=float)
    while FEs < max_FEs:
        weights = weights / weights.sum()
        new_sols = []
        for _ in range(N):
            l = int(np.random.choice(archive_size, p=weights))
            # Gaussian std: mean of distances to other archive solutions
            sigma = np.array([xi * np.sum(np.abs(arch[:, j] - arch[l, j])) / (archive_size - 1)
                               for j in range(D)])
            sigma = np.maximum(sigma, 1e-10)
            s = arch[l] + sigma * np.random.randn(D)
            s = _bh_clip(s, lb, ub)
            fs = func(s); FEs += 1
            new_sols.append((s, fs))
            if FEs >= max_FEs: break
        for s, fs in new_sols:
            if fs < farch[-1]:
                arch[-1] = s; farch[-1] = fs
        order = np.argsort(farch); arch = arch[order]; farch = farch[order]
        if farch[0] <= VTR: return FEs, float(farch[0]), curve
        if record_curve: curve.append(float(farch[0]))
    return None, float(farch[0]), curve


def harmony_search(func, lb, ub, D, N=50, max_FEs=150000, VTR=1e-12,
                   HMCR=0.95, PAR=0.3, bw=0.2,
                   seed=None, record_curve=False):
    """Harmony Search (Geem et al. 2001)."""
    if seed is not None: np.random.seed(seed)
    hm = lb + (ub - lb) * np.random.rand(N, D)
    fhm = np.array([func(hm[k]) for k in range(N)])
    FEs = N
    curve = [float(np.min(fhm))] if record_curve else None
    while FEs < max_FEs:
        x_new = np.zeros(D)
        for j in range(D):
            if np.random.rand() < HMCR:
                x_new[j] = hm[np.random.randint(N), j]
                if np.random.rand() < PAR:
                    x_new[j] += bw * (2*np.random.rand()-1)
            else:
                x_new[j] = lb + (ub-lb)*np.random.rand()
        x_new = _bh_clip(x_new, lb, ub)
        f_new = func(x_new); FEs += 1
        worst_idx = int(np.argmax(fhm))
        if f_new < fhm[worst_idx]:
            hm[worst_idx] = x_new; fhm[worst_idx] = f_new
        if np.min(fhm) <= VTR: return FEs, float(np.min(fhm)), curve
        if record_curve: curve.append(float(np.min(fhm)))
    return None, float(np.min(fhm)), curve


def firefly_algorithm(func, lb, ub, D, N=50, max_FEs=150000, VTR=1e-12,
                      alpha=0.25, beta0=0.2, gamma=1.0,
                      seed=None, record_curve=False):
    """Firefly Algorithm (Yang 2010)."""
    if seed is not None: np.random.seed(seed)
    pop = lb + (ub-lb)*np.random.rand(N, D)
    fitness = np.array([func(pop[i]) for i in range(N)])
    FEs = N
    curve = [float(np.min(fitness))] if record_curve else None
    while FEs < max_FEs:
        new_pop = pop.copy(); new_fit = fitness.copy()
        for i in range(N):
            for j in range(N):
                if fitness[j] < fitness[i]:
                    r = float(np.sqrt(np.sum((pop[i]-pop[j])**2)))
                    beta = beta0 * np.exp(-gamma * r**2)
                    x_new = pop[i] + beta*(pop[j]-pop[i]) \
                            + alpha*(np.random.rand(D)-0.5)*(ub-lb)
                    x_new = _bh_clip(x_new, lb, ub)
                    f_new = func(x_new); FEs += 1
                    if f_new < new_fit[i]:
                        new_pop[i] = x_new; new_fit[i] = f_new
                    if FEs >= max_FEs: break
            if FEs >= max_FEs: break
        pop = new_pop; fitness = new_fit
        if np.min(fitness) <= VTR: return FEs, float(np.min(fitness)), curve
        if record_curve: curve.append(float(np.min(fitness)))
    return None, float(np.min(fitness)), curve


def mvo(func, lb, ub, D, N=50, max_FEs=150000, VTR=1e-12,
        WEPmin=0.2, WEPmax=1.0,
        seed=None, record_curve=False):
    """
    Multi-Verse Optimizer (Mirjalili et al. 2016).
    WEP = Wormhole Existence Probability, TDR = Travelling Distance Rate.
    """
    if seed is not None: np.random.seed(seed)
    pop = lb + (ub-lb)*np.random.rand(N, D)
    fitness = np.array([func(pop[i]) for i in range(N)])
    FEs = N
    curve = [float(np.min(fitness))] if record_curve else None
    max_iter = (max_FEs - N) // N + 1
    t = 0
    while FEs < max_FEs:
        t += 1
        WEP = WEPmin + t*(WEPmax-WEPmin)/max_iter
        TDR = 1 - (t/max_iter)**(1/6)
        best_idx = int(np.argmin(fitness)); best = pop[best_idx]
        order = np.argsort(fitness)
        norm_fit = (fitness - fitness.min())
        denom = norm_fit.max() + 1e-300
        norm_fit = norm_fit / denom
        for i in range(N):
            for j in range(D):
                r1 = np.random.rand()
                if r1 < norm_fit[i]:
                    # White hole / black hole tunnel
                    idx = np.random.choice(order)
                    pop[i, j] = pop[idx, j]
                r2 = np.random.rand()
                if r2 < WEP:
                    r3 = np.random.rand()
                    if r3 < 0.5:
                        pop[i, j] = best[j] + TDR*((ub-lb)*np.random.rand()+lb)
                    else:
                        pop[i, j] = best[j] - TDR*((ub-lb)*np.random.rand()+lb)
            pop[i] = _bh_clip(pop[i], lb, ub)
            f_new = func(pop[i]); FEs += 1
            fitness[i] = f_new
            if FEs >= max_FEs: break
        if np.min(fitness) <= VTR: return FEs, float(np.min(fitness)), curve
        if record_curve: curve.append(float(np.min(fitness)))
    return None, float(np.min(fitness)), curve


def isa(func, lb, ub, D, N=50, max_FEs=150000, VTR=1e-12,
        alpha=0.2,
        seed=None, record_curve=False):
    """
    Interior Search Algorithm (Gandomi 2014), in a simplified form with two
    moves: a Gaussian step around the best solution scaled by the distance to
    it, and a step towards a random member of the population plus Gaussian
    noise.
    """
    if seed is not None: np.random.seed(seed)
    pop = lb + (ub-lb)*np.random.rand(N, D)
    fitness = np.array([func(pop[i]) for i in range(N)])
    FEs = N
    curve = [float(np.min(fitness))] if record_curve else None
    while FEs < max_FEs:
        best_idx = int(np.argmin(fitness)); best = pop[best_idx]
        for i in range(N):
            # step around the best solution, or towards a random member
            r = np.random.rand()
            if r < 0.5:
                x_new = best + alpha * np.random.randn(D) * np.abs(pop[i] - best)
            else:
                j = np.random.randint(N)
                while j == i: j = np.random.randint(N)
                lam = np.random.rand()
                x_new = pop[i] + lam*(pop[j] - pop[i]) + alpha*np.random.randn(D)
            x_new = _bh_clip(x_new, lb, ub)
            f_new = func(x_new); FEs += 1
            if f_new < fitness[i]:
                pop[i] = x_new; fitness[i] = f_new
            if FEs >= max_FEs: break
        if np.min(fitness) <= VTR: return FEs, float(np.min(fitness)), curve
        if record_curve: curve.append(float(np.min(fitness)))
    return None, float(np.min(fitness)), curve


def csa(func, lb, ub, D, N=50, max_FEs=150000, VTR=1e-12,
        pa=0.25, nd=0.15,
        seed=None, record_curve=False):
    """
    Cuckoo Search (Yang & Deb 2009), with Levy flights drawn by Mantegna's
    algorithm.
    """
    if seed is not None: np.random.seed(seed)

    def _levy(D, beta=1.5):
        import math as _math
        sigma = (_math.gamma(1+beta)*np.sin(np.pi*beta/2) /
                 (_math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
        u = np.random.randn(D) * sigma
        v = np.abs(np.random.randn(D))
        return u / v**(1/beta)

    pop = lb + (ub-lb)*np.random.rand(N, D)
    fitness = np.array([func(pop[i]) for i in range(N)])
    FEs = N
    curve = [float(np.min(fitness))] if record_curve else None
    while FEs < max_FEs:
        best_idx = int(np.argmin(fitness)); best = pop[best_idx]
        # Levy flight
        for i in range(N):
            step = _levy(D)
            x_new = pop[i] + step * (pop[i] - best)
            x_new = _bh_clip(x_new, lb, ub)
            f_new = func(x_new); FEs += 1
            if f_new < fitness[i]:
                pop[i] = x_new; fitness[i] = f_new
            if FEs >= max_FEs: break
        # Abandon fraction pa of nests
        for i in range(N):
            if np.random.rand() < pa:
                j, k = np.random.choice([x for x in range(N) if x != i], 2, replace=False)
                x_new = _bh_clip(pop[i] + nd*(pop[j]-pop[k]), lb, ub)
                f_new = func(x_new); FEs += 1
                if f_new < fitness[i]:
                    pop[i] = x_new; fitness[i] = f_new
            if FEs >= max_FEs: break
        if np.min(fitness) <= VTR: return FEs, float(np.min(fitness)), curve
        if record_curve: curve.append(float(np.min(fitness)))
    return None, float(np.min(fitness)), curve


# ==============================================================================
#  Runner utilities
# ==============================================================================

def optimum(fname, D):
    """Global minimum value f* of benchmark `fname` at dimension D."""
    if fname == 'F05_InvCosWave':
        return -(D - 1.0)
    return float(EVO_BENCHMARK[fname]['opt'])


def run_case(algo_fn, fname, D, N, max_FEs, seed, vtr=1e-10,
             record_curve=False, **kw):
    """
    One run on a benchmark case with the success test of the manuscript,
    f - f* <= vtr. Every algorithm stops when its raw best value is <= the VTR
    it is given, so the absolute target f* + vtr is passed; for functions with
    f* != 0 (F04, F05, F18, F20) passing vtr alone would test f <= vtr instead.
    """
    info = EVO_BENCHMARK[fname]
    return run_trial(algo_fn, info['func'], info['lb'], info['ub'], D, N,
                     max_FEs, optimum(fname, D) + vtr, seed,
                     record_curve=record_curve, **kw)


def run_trial(algo_fn, func, lb, ub, D, N, max_FEs, VTR, seed,
              record_curve=False, **kw):
    """
    One run. VTR is the ABSOLUTE target on the raw objective: the run succeeds
    when the best value is <= VTR. For a benchmark case use run_case(), which
    sets VTR = f* + 1e-10.
    """
    result = algo_fn(func, lb, ub, D, N=N, max_FEs=max_FEs, VTR=VTR,
                     seed=seed, record_curve=record_curve, **kw)
    nfe = result[0]; val = result[1]
    curve = result[2] if len(result) > 2 else None
    return {'nfe_to_vtr': nfe, 'best_fitness': float(val),
            'curve': curve, 'success': nfe is not None}


def aggregate_fitness(trials):
    vals = [t['best_fitness'] for t in trials]
    return {'best': float(np.min(vals)), 'mean': float(np.mean(vals)),
            'worst': float(np.max(vals)), 'std': float(np.std(vals)),
            'raw': vals}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True); return path


def save_csv(df, path):
    ensure_dir(os.path.dirname(path) or '.'); df.to_csv(path, index=False)
    print(f"  [saved] {path}")

