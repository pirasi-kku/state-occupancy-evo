# -*- coding: utf-8 -*-
"""
_harness.py
===========
Shared machinery for the experiment scripts, so every experiment uses identical
seeding, statistics and file conventions.

Design rules enforced here:

1. MATCHED SEEDS. Every arm uses the same seed for a given (problem, D, run),
   so all arms start from an identical initial population and the Wilcoxon
   signed-rank test is a genuine paired test (common random numbers).

2. NFE-to-VTR IS THE METRIC for the convergence-speed experiments, matching the
   paper. Runs that never reach the VTR are excluded from the NFE mean and
   reported separately as the success rate, so speed is never credited at the
   expense of reliability.

3. RESUME. A per-case CSV that already exists is loaded and skipped, so an
   interrupted sweep resumes where it stopped.
"""

import os, sys, datetime
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon as scipy_wilcoxon

from evo_core import (
    evo, aco_r, harmony_search, firefly_algorithm, mvo, isa, csa,
    run_trial, run_case, optimum, ensure_dir, save_csv,
    EVO_BENCHMARK, get_dim_list,
)
from soevo import ARMS, ARM_FN, ARM_LABEL, ARM_COLORS


# ==============================================================================
#  Shared config defaults
# ==============================================================================
# Protocol of the manuscript: VTR 1e-10, N = 50, MaxFEs = 150,000, 50 runs,
# n_s = 20. The VTR is relative to the optimum: a run succeeds when
# f - f* <= 1e-10 (evo_core.run_case passes the absolute target f* + VTR).
DEFAULT_VTR     = 1e-10
DEFAULT_N_POP   = 50
DEFAULT_MAX_FES = 150_000
DEFAULT_N_RUNS  = 50
DEFAULT_NS      = 20
DEFAULT_EPS     = (0.40, 0.01)
ALPHA           = 0.05

ARM_LIST = ['SOEVO_sel', 'SOEVO_all', 'SOEVO_eps_sel', 'SOEVO_eps_all']

COMPARATOR_FN = {
    'EVO': evo, 'ACO': aco_r, 'HS': harmony_search, 'FA': firefly_algorithm,
    'MVO': mvo, 'ISA': isa, 'CSA': csa,
}
COMPARATOR_LIST = ['EVO', 'ACO', 'HS', 'FA', 'MVO', 'ISA', 'CSA']


def algo_callable(name):
    """Resolve an algorithm name to a run_trial-compatible callable."""
    if name in ARM_FN:
        return ARM_FN[name]
    if name in COMPARATOR_FN:
        return COMPARATOR_FN[name]
    raise KeyError(f"unknown algorithm {name!r}")


def algo_kwargs(name, ns=DEFAULT_NS, eps=DEFAULT_EPS):
    """Per-algorithm keyword arguments."""
    if name in ARMS:
        kw = {'ns': ns}
        if ARMS[name]['sampling'] == 'epsilon':
            kw['eps_init'], kw['eps_end'] = eps
        return kw
    return {}


# ==============================================================================
#  Seeding — one shared table, reused by every algorithm
# ==============================================================================

def matched_seeds(n_problems, n_runs, base=20240101):
    """seeds[problem_index][run] — identical across algorithms."""
    rng = np.random.default_rng(base)
    return rng.integers(0, 2**31 - 1, size=(n_problems, n_runs)).tolist()


# ==============================================================================
#  Statistics
# ==============================================================================

def nfe_stats(trials, n_runs):
    """Aggregate NFE over converged runs only; report SR separately."""
    ok = [t['nfe_to_vtr'] for t in trials if t['nfe_to_vtr'] is not None]
    if not ok:
        return dict(NS=0, SR=0.0, mean_NFE=np.nan, std_NFE=np.nan,
                    median_NFE=np.nan, min_NFE=np.nan, max_NFE=np.nan)
    arr = np.asarray(ok, dtype=float)
    return dict(NS=len(ok), SR=100.0 * len(ok) / n_runs,
                mean_NFE=float(arr.mean()), std_NFE=float(arr.std()),
                median_NFE=float(np.median(arr)),
                min_NFE=float(arr.min()), max_NFE=float(arr.max()))


def fitness_stats(trials):
    v = np.asarray([t['best_fitness'] for t in trials], dtype=float)
    return dict(mean_fit=float(v.mean()), std_fit=float(v.std()),
                best_fit=float(v.min()), worst_fit=float(v.max()),
                median_fit=float(np.median(v)))


def paired_wilcoxon_nfe(trials_ref, trials_other, alpha=ALPHA):
    """
    Paired Wilcoxon on NFE-to-VTR over runs where BOTH converged.
    '+' ref faster, '-' other faster, '0' no significant difference.
    Returns (symbol, pvalue, n_paired).
    """
    pairs = [(a['nfe_to_vtr'], b['nfe_to_vtr'])
             for a, b in zip(trials_ref, trials_other)
             if a['nfe_to_vtr'] is not None and b['nfe_to_vtr'] is not None]
    if len(pairs) < 5:
        return '0', float('nan'), len(pairs)
    va = np.asarray([p[0] for p in pairs], dtype=float)
    vb = np.asarray([p[1] for p in pairs], dtype=float)
    # If every paired difference is zero the two algorithms behaved identically
    # on this case. scipy returns p = NaN there, and a naive `p >= alpha` test
    # is False for NaN, which would silently report a tie as a loss.
    if np.all(va == vb):
        return '0', float('nan'), len(pairs)
    try:
        _, p = scipy_wilcoxon(va, vb, alternative='two-sided')
        if not np.isfinite(p) or p >= alpha:
            return '0', float(p), len(pairs)
        return ('+' if np.mean(va) < np.mean(vb) else '-'), float(p), len(pairs)
    except Exception:
        return '0', float('nan'), len(pairs)


def paired_wilcoxon_fitness(trials_ref, trials_other, alpha=ALPHA):
    """Paired Wilcoxon on final fitness (for fixed-budget experiments)."""
    va = np.asarray([t['best_fitness'] for t in trials_ref], dtype=float)
    vb = np.asarray([t['best_fitness'] for t in trials_other], dtype=float)
    if len(va) != len(vb) or len(va) < 5:
        return '0', float('nan')
    if np.all(va == vb):            # identical behaviour, not a loss
        return '0', float('nan')
    try:
        _, p = scipy_wilcoxon(va, vb, alternative='two-sided')
        if not np.isfinite(p) or p >= alpha:
            return '0', float(p)
        return ('+' if np.mean(va) < np.mean(vb) else '-'), float(p)
    except Exception:
        return '0', float('nan')


def penalised_cost(trials, max_FEs):
    """Mean NFE with non-converged runs charged the full budget."""
    vals = [t['nfe_to_vtr'] if t['nfe_to_vtr'] is not None else max_FEs
            for t in trials]
    return float(np.mean(vals))


def friedman_ranks(cost_matrix, algo_names):
    """
    cost_matrix: (n_algos, n_cases), lower is better.
    Returns (DataFrame of mean ranks, chi2, p).

    TIES GET AVERAGE RANKS. This matters more than it sounds: on a benchmark
    where many cases saturate, most algorithms return bit-identical values, and
    assigning them distinct ranks by array position hands rank 1 to whichever
    algorithm happens to be listed first. An earlier version of this function
    did exactly that and produced a ranking that was an artefact of list order.
    """
    from scipy.stats import friedmanchisquare, rankdata
    n_a, n_c = cost_matrix.shape
    ranks = np.zeros_like(cost_matrix, dtype=float)
    for j in range(n_c):
        ranks[:, j] = rankdata(cost_matrix[:, j], method='average')
    avg = ranks.mean(axis=1)
    try:
        chi2, p = friedmanchisquare(*[cost_matrix[i, :] for i in range(n_a)])
    except Exception:
        chi2, p = np.nan, np.nan
    df = pd.DataFrame({'Algorithm': algo_names, 'MeanRank': avg}
                      ).sort_values('MeanRank').reset_index(drop=True)
    return df, float(chi2), float(p)


def nemenyi_cd(k, n, alpha=0.05):
    """Critical difference for the Nemenyi post-hoc test."""
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
         8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}
    if k not in q:
        return float('nan')
    return q[k] * np.sqrt(k * (k + 1) / (6.0 * n))


# ==============================================================================
#  Logging
# ==============================================================================

class Tee:
    def __init__(self, path):
        ensure_dir(os.path.dirname(path) or '.')
        self._f = open(path, 'w', encoding='utf-8', buffering=1)
        self._s = sys.stdout
    def write(self, m): self._s.write(m); self._f.write(m)
    def flush(self):    self._s.flush();  self._f.flush()
    def close(self):    sys.stdout = self._s; self._f.close()


def start_log(out_base, tag):
    ensure_dir(out_base)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    tee = Tee(os.path.join(out_base, f'{tag}_log_{ts}.txt'))
    sys.stdout = tee
    return tee, ts


def banner(title, **cfg):
    print("=" * 72)
    print(f"  {title}")
    for k, v in cfg.items():
        print(f"  {k} = {v}")
    print("=" * 72)
    print("\n  Arms:")
    for a in ARM_LIST:
        print(f"    {a:18s}  {ARM_LABEL[a]}")
    print()
