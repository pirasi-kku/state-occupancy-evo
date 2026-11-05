# -*- coding: utf-8 -*-
"""
make_figures.py
===============
Generates every figure of the manuscript from the CSV files in ../data/ and
writes PDF and PNG files into ../figures/. Every figure is computed at run
time and prints its source file.

Run:
    python make_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
RES  = os.path.join(BASE, "data")
FIG  = os.path.join(BASE, "figures")


def R(*p):
    return os.path.join(RES, *p)


def out(name):
    return os.path.join(FIG, name)


def rel(p):
    return os.path.relpath(p, BASE)


#  Palette: SO-EVO red, EVO blue.
SOEVO, EVO = "#E63946", "#1D6FA4"

#  Key of SO-EVO in the CSVs; must agree with make_tables.py.
METHOD = 'SOEVO_sel'            # occupancy + selective credit -> "SO-EVO"

plt.rcParams.update({'font.size': 13})


def save(fig, stem, bbox=None):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(out(stem + ".pdf"), bbox_inches=bbox)
    fig.savefig(out(stem + ".png"), dpi=150, bbox_inches=bbox)
    plt.close(fig)


def load_main():
    """NFE-to-VTR master, with the positional dimension slot used by the tables."""
    src = R('results_field', 'field_master.csv')
    m = pd.read_csv(src)
    slots = {}
    for f, g in m.groupby('Function'):
        for i, d in enumerate(sorted(g.D.unique()), 1):
            slots[(f, d)] = i
    m['slot'] = [slots[(f, d)] for f, d in zip(m.Function, m.D)]
    print(f"  [field]  {rel(src)}  ({m.groupby(['Function','D']).ngroups} cases"
          f" x {m.Algorithm.nunique()} algorithms)")
    return m


# ============================================================================
#  Figure 1: per-case evaluation-count ratio EVO/SO-EVO, by dimension
# ============================================================================
def fig_speedup(m):
    piv = (m.pivot_table(index=['Function', 'D', 'slot'], columns='Algorithm',
                         values='mean_NFE')
             .reset_index())
    piv['ratio'] = piv['EVO'] / piv[METHOD]
    piv = piv[np.isfinite(piv.ratio)]
    slots = sorted(piv.slot.unique())
    data = [piv[piv.slot == s].ratio.values for s in slots]
    means = [d.mean() for d in data]
    labels = ['5', '10', '30', '50', '100']
    print("  [speedup]  results_field/field_master.csv  -> slot means " +
          ", ".join(f"{v:.1f}x" for v in means))

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(len(slots))
    bp = ax.boxplot(data, positions=x, widths=.55, patch_artist=True,
                    medianprops=dict(color='white', lw=1.8),
                    flierprops=dict(marker='', ls='none'),
                    whiskerprops=dict(color='#666'), capprops=dict(color='#666'))
    for p in bp['boxes']:
        p.set(facecolor=SOEVO, alpha=.55, edgecolor='#8f2029')
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        ax.scatter(x[i] + rng.uniform(-.16, .16, len(d)), d, s=16, color='#8f2029',
                   alpha=.75, zorder=3, linewidths=0)
    ax.plot(x, means, '-s', color=EVO, lw=2.0, ms=8, zorder=4, label='mean')
    tops = [d.max() for d in data]
    for i, v in enumerate(means):
        ax.annotate(f'{v:.1f}×', (x[i], tops[i]), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=11.5, fontweight='bold',
                    color=EVO)
    ax.set_ylim(0, max(tops) * 1.14)
    ax.axhline(1, color='#888', ls=':', lw=1.4)
    ax.text(len(slots) - .55, 1.6, 'parity', fontsize=9.5, color='#888')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel('Dimension  $D$')
    ax.set_ylabel('Evaluations to VTR, EVO / SO-EVO')
    ax.set_title('The advantage grows with dimension', fontsize=13.5)
    ax.legend(frameon=False, fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=.3)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_speedup")


# ============================================================================
#  Figure 2: Friedman / Nemenyi critical-difference diagrams
# ============================================================================
def _cd_axis(ax, ranks, CD, title, lo=1, hi=8):
    """One CD diagram. `ranks` is a list of (display name, mean rank), any order."""
    items = sorted(ranks, key=lambda t: t[1])
    ax.set_xlim(lo - .05, hi + .05); ax.set_ylim(1.45, 5); ax.axis('off')
    AX = 4.2
    ax.plot([lo, hi], [AX, AX], 'k', lw=1.2)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [AX, AX + .12], 'k', lw=1)
        ax.text(t, AX + .30, str(t), ha='center', fontsize=9)
    ax.text((lo + hi) / 2, AX + .66, title, ha='center', fontsize=10)
    ax.plot([lo, lo + CD], [AX - .32, AX - .32], 'k', lw=2)
    ax.plot([lo, lo], [AX - .38, AX - .26], 'k')
    ax.plot([lo + CD, lo + CD], [AX - .38, AX - .26], 'k')
    ax.text(lo + CD / 2, AX - .58, f'CD = {CD:.3f}', ha='center', fontsize=8)

    half = (len(items) + 1) // 2
    left, right = items[:half], items[half:][::-1]

    def col(n):
        return ((SOEVO, 'bold') if n == 'SO-EVO' else
                (EVO, 'bold') if n == 'EVO' else ('black', 'normal'))

    ytop, step = AX - .9, .5
    for i, (n, rk) in enumerate(left):
        y = ytop - i * step; c, fw = col(n)
        ax.plot([rk, rk], [AX, y], color=c, lw=1.1)
        ax.plot([rk, lo - .05], [y, y], color=c, lw=1.1)
        ax.text(lo - .12, y, f'{n} ({rk:.2f})', ha='right', va='center',
                fontsize=9, color=c, fontweight=fw)
    for i, (n, rk) in enumerate(right):
        y = ytop - i * step; c, fw = col(n)
        ax.plot([rk, rk], [AX, y], color=c, lw=1.1)
        ax.plot([rk, hi + .05], [y, y], color=c, lw=1.1)
        ax.text(hi + .12, y, f'{n} ({rk:.2f})', ha='left', va='center',
                fontsize=9, color=c, fontweight=fw)

    vals = [r for _, r in items]
    bars, j = [], 0
    while j < len(vals):
        kk = j
        while kk + 1 < len(vals) and vals[kk + 1] - vals[j] < CD:
            kk += 1
        if kk > j:
            bars.append((vals[j], vals[kk]))
        j += 1
    maximal = [b for b in bars
               if not any(b != o and b[0] >= o[0] and b[1] <= o[1] for o in bars)]
    for idx, (a, b) in enumerate(maximal):
        ax.plot([a - .04, b + .04], [AX - .12 - idx * .16] * 2, 'k', lw=3,
                solid_capstyle='round')


def _read_ranks(path):
    fr = pd.read_csv(path).sort_values('MeanRank')
    ranks = [('SO-EVO' if a == METHOD else a, float(v))
             for a, v in zip(fr.Algorithm, fr.MeanRank)]
    return ranks, float(fr.CD.iloc[0])


def fig_cd():
    p1 = R('results_field', 'field_friedman.csv')
    p2 = R('results_field_fixedfe', 'field_fixedfe_friedman.csv')
    r1, cd1 = _read_ranks(p1)
    r2, cd2 = _read_ranks(p2)
    print(f"  [CD]  {rel(p1)} (CD={cd1:.3f}) , {rel(p2)} (CD={cd2:.3f})")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 5.4))
    _cd_axis(ax1, r1, cd1,
             '(a)  Mean Friedman rank on penalized expected cost, 100 cases '
             '(lower is better)')
    _cd_axis(ax2, r2, cd2,
             '(b)  Mean Friedman rank on final objective value after 150,000 '
             'evaluations, 90 cases')
    fig.tight_layout()
    save(fig, "fig_cd_diagram", bbox='tight')


# ============================================================================
#  Figure 4: wall-clock time
# ============================================================================
def fig_timing():
    src = R('results_timing', 'timing_master.csv')
    t = pd.read_csv(src)
    Ds = sorted(t.D.unique())
    evo = [t[(t.D == D) & (t.Algorithm == 'EVO')].mean_secs.mean() for D in Ds]
    soe = [t[(t.D == D) & (t.Algorithm == METHOD)].mean_secs.mean() for D in Ds]
    sp = [e / s for e, s in zip(evo, soe)]
    nfe_ratio = [t[(t.D == D) & (t.Algorithm == 'EVO')].mean_NFE.mean() /
                 t[(t.D == D) & (t.Algorithm == METHOD)].mean_NFE.mean() for D in Ds]
    nfun = t[t.Algorithm == 'EVO'].groupby('D').Function.nunique().min()
    print(f"  [timing]  {rel(src)}  ({nfun} functions x 10 runs per cell)  -> "
          "speedups " + ", ".join(f"{s:.1f}x" for s in sp))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.6),
                                 gridspec_kw=dict(width_ratios=[1.15, 1]))
    x = np.arange(len(Ds))
    a1.plot(x, evo, '-o', color=EVO, lw=2.4, ms=9, label='EVO')
    a1.plot(x, soe, '-s', color=SOEVO, lw=2.4, ms=9, label='SO-EVO')
    for i in range(len(Ds)):
        a1.annotate(f'{sp[i]:.2f}×', (x[i], evo[i]), textcoords="offset points",
                    xytext=(0, 11), ha='center', fontsize=11.5,
                    fontweight='bold', color='#333')
    a1.set_xticks(x); a1.set_xticklabels([int(d) for d in Ds])
    a1.set_xlabel('Dimension  $D$')
    a1.set_ylabel('Mean wall-clock time to VTR (s)')
    a1.set_title('(a)  Time to reach the VTR', fontsize=13)
    a1.legend(frameon=False, fontsize=12, loc='upper left')
    a1.grid(axis='y', alpha=.3); a1.set_ylim(0, max(evo) * 1.18)
    a1.spines[['top', 'right']].set_visible(False)

    w = .38
    a2.bar(x - w / 2, nfe_ratio, w, color=SOEVO, label='evaluations')
    a2.bar(x + w / 2, sp, w, color=EVO, label='seconds')
    a2.axhline(1, color='#888', ls=':', lw=1.4)
    a2.set_xticks(x); a2.set_xticklabels([int(d) for d in Ds])
    a2.set_xlabel('Dimension  $D$')
    a2.set_ylabel('Ratio EVO / SO-EVO')
    a2.set_title('(b)  Where the gain comes from', fontsize=13)
    a2.legend(frameon=False, fontsize=11, loc='upper left')
    a2.grid(axis='y', alpha=.3)
    a2.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_timing")


# ============================================================================
#  Figure 3: sensitivity to n_s
# ============================================================================
FUNC_SHORT = {'F01_Ackley1': 'Ackley 1', 'F07_Rastrigin': 'Rastrigin',
              'F09_Schwefel12': 'Schwefel 1.2', 'F11_Griewank': 'Griewank',
              'F12_PowellSingular2': 'Powell Sing. 2', 'F16_Sphere': 'Sphere'}


def fig_sensitivity():
    src = R('results_sensitivity', 'sensitivity_ns.csv')
    d = pd.read_csv(src)
    d = d[(d.Arm == METHOD) & (d.Sweep == 'ns')]
    vals = sorted(d.Value.unique())
    agg = d.groupby('Value').agg(nfe=('mean_NFE', 'mean'), sr=('SR', 'mean')).sort_index()
    spread = 100 * (agg.nfe.max() - agg.nfe.min()) / agg.nfe.min()
    ncase = d.groupby(['Function', 'D']).ngroups
    print(f"  [sensitivity]  {rel(src)}  ({ncase} cases, {d.Function.nunique()} "
          f"functions)  -> spread {spread:.1f}%, SR "
          f"{agg.sr.min():.0f}-{agg.sr.max():.0f}%")

    mk = ['o', 's', '^', 'D', 'v', 'P']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.6))
    x = np.arange(len(vals))
    for i, f in enumerate(sorted(d.Function.unique())):
        g = d[d.Function == f].groupby('Value').mean_NFE.mean().sort_index()
        a1.plot(x, g.values, marker=mk[i % len(mk)], lw=1.8, ms=7,
                label=FUNC_SHORT.get(f, f))
    a1.set_xticks(x); a1.set_xticklabels([int(v) for v in vals])
    a1.set_xlabel('$n_s$ (number of coefficient states)')
    a1.set_ylabel('Mean NFE-to-VTR')
    a1.set_title('(a)  Per function', fontsize=13)
    a1.axvline(list(vals).index(20), color=SOEVO, ls=':', lw=1.5)
    a1.grid(alpha=.3); a1.spines[['top', 'right']].set_visible(False)
    a1.legend(frameon=False, fontsize=9.5, loc='center right')

    a2.plot(x, agg.nfe.values, '-s', color=SOEVO, lw=2.4, ms=9)
    for i, v in enumerate(agg.nfe.values):
        a2.annotate(f'{v:,.0f}', (x[i], v), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=10.5, color='#333')
    a2.axvline(list(vals).index(20), color=SOEVO, ls=':', lw=1.5)
    a2.text(list(vals).index(20) + .08, agg.nfe.min(), 'default', fontsize=10,
            color=SOEVO)
    a2.set_xticks(x); a2.set_xticklabels([int(v) for v in vals])
    a2.set_xlabel('$n_s$ (number of coefficient states)')
    a2.set_ylabel(f'Mean NFE-to-VTR over {ncase} cases')
    a2.set_title(f'(b)  All {ncase} cases: {spread:.1f}% spread, '
                 f'SR {agg.sr.min():.0f}% throughout', fontsize=12.5)
    a2.set_ylim(agg.nfe.min() * .96, agg.nfe.max() * 1.06)
    a2.grid(alpha=.3); a2.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_sensitivity")


# ============================================================================
def main():
    os.makedirs(FIG, exist_ok=True)
    print("Regenerating the figures from ../data/:")
    m = load_main()
    fig_speedup(m)
    fig_cd()
    fig_timing()
    fig_sensitivity()
    print("Done -> ../figures/")


if __name__ == '__main__':
    main()
