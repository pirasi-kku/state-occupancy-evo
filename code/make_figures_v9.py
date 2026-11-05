# -*- coding: utf-8 -*-
"""
make_figures_v9.py
==================
Generates every figure of the manuscript from the CSV files in ../data/ and
writes PDF and PNG files into ../figures/. Every figure is computed at run
time and prints its source file.

Run:
    python make_figures_v9.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ----------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
RES  = os.path.join(BASE, "data")        # supplementary layout
FIG  = os.path.join(BASE, "figures")


def R(*p):
    return os.path.join(RES, *p)


def out(name):
    return os.path.join(FIG, name)


def rel(p):
    return os.path.relpath(p, BASE)


#  Palette and naming conventions.
SOEVO, EVO, GREEN, RED, BLUE = "#E63946", "#1D6FA4", "#2A9D8F", "#C1272D", "#2C6E9B"
GREY = "#cfcfcf"

#  Arm keys as they appear in the CSVs, and the display names used in the
#  manuscript. These must agree with make_tables_v9.py.
METHOD = 'SOEVO_sel'            # occupancy + selective credit -> "SO-EVO"
REWALL = 'SOEVO_all'            # occupancy + credit to all four
COMPARATORS = ['EVO', 'ACO', 'HS', 'FA', 'MVO', 'ISA', 'CSA']

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
#  FIGURE 1 — mechanism schematic (conceptual; not data-derived)
# ============================================================================
def fig_mechanism():
    fig = plt.figure(figsize=(11, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=.62, wspace=.22)
    xs = np.arange(21)

    aA = fig.add_subplot(gs[0, 0])
    aA.bar(xs, np.ones(21), color=EVO, width=.8)
    aA.set_title('(A)  EVO — uniform draw', fontsize=14, fontweight='bold',
                 color=EVO, pad=8)
    aA.set_xlabel('coefficient value  $r\\in(0,1]$', fontsize=12)
    aA.set_ylabel('$P(r)$', fontsize=12)
    aA.set_yticks([]); aA.set_xticks([])
    aA.spines[['top', 'right', 'left']].set_visible(False)
    aA.text(.5, -.30, 'every value equally likely · no memory across iterations',
            transform=aA.transAxes, ha='center', fontsize=10, style='italic',
            color='#555')

    aB = fig.add_subplot(gs[0, 1])
    h = np.array([3.0, 3.2, 2.6, 2.9, 2.3, 2.5, 2.0, 1.8, 2.1, 1.5, 1.3, 1.4,
                  1.0, .9, 1.1, .7, .6, .7, .5, .55, .5])
    aB.bar(xs, h, color=SOEVO, width=.8)
    aB.axhline(np.mean(h), ls='--', color=EVO, lw=1.6)
    aB.text(20, np.mean(h) + .12, 'EVO uniform', color=EVO, ha='right', fontsize=10)
    aB.set_title('(B)  SO-EVO — occupancy over states', fontsize=14,
                 fontweight='bold', color=SOEVO, pad=8)
    aB.set_xlabel('coefficient state  $c_k$, $k=0,\\dots,n_s$', fontsize=12)
    aB.set_ylabel('$P(k)$', fontsize=12)
    aB.set_yticks([]); aB.set_xticks([])
    aB.spines[['top', 'right', 'left']].set_visible(False)
    aB.text(.5, -.30, 'rewarded states gain weight · every state keeps $O\\geq1$',
            transform=aB.transAxes, ha='center', fontsize=10, style='italic',
            color='#555')

    aC = fig.add_subplot(gs[1, :])
    aC.set_xlim(0, 100); aC.set_ylim(0, 40); aC.axis('off')
    aC.text(0, 39, '(C)  Sampling and selective credit, one individual, one iteration',
            fontsize=14, fontweight='bold', va='top')

    def box(x, y, w, hh, text, fc, fs=12):
        aC.add_patch(FancyBboxPatch((x, y), w, hh,
                                    boxstyle="round,pad=0.5,rounding_size=2",
                                    fc=fc, ec='none'))
        aC.text(x + w / 2, y + hh / 2, text, ha='center', va='center',
                color='white', fontsize=fs, fontweight='bold')

    def arr(p1, p2, c='#444', lw=2.0):
        aC.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=17,
                                     lw=lw, color=c,
                                     connectionstyle="arc3,rad=0"))

    box(2, 21, 26, 12,
        'Draw $k_\\alpha,k_\\beta,k_\\gamma,k_\\delta$\nfrom '
        '$O_\\alpha,O_\\beta,O_\\gamma,O_\\delta$', BLUE, 12)
    box(35, 21, 27, 12,
        'Branch move; keep the\nbetter of $X_1,X_2$;\n$U$ = its coefficients',
        GREEN, 11.5)
    box(69, 21, 28, 12,
        'accept? reward only\n$O_\\phi[k_\\phi]{+}{=}1$, $\\phi\\in U$\n'
        'reject: $O$ unchanged', SOEVO, 11)
    arr((28, 27), (35, 27)); arr((62, 27), (69, 27))
    arr((83, 21), (83, 12)); arr((83, 12), (15, 12)); arr((15, 12), (15, 21))
    aC.text(49, 13.4, 'next iteration', ha='center', fontsize=10.5, color='#444',
            style='italic')
    aC.text(49, 5.6,
            'The coefficients drawn for the discarded candidate, and those the '
            'branch never used,\nare left unchanged: only the states that '
            'scaled the surviving move are credited.',
            ha='center', va='center', fontsize=11, color='#444')

    fig.suptitle('The SO-EVO mechanism: state-occupancy-guided coefficient sampling',
                 fontsize=15, fontweight='bold', y=1.0)
    print("  [mechanism]  conceptual schematic (no data source)")
    save(fig, "fig_mechanism_v9", bbox='tight')


# ============================================================================
#  FIGURE 2 — per-case speedup over EVO, by dimension
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
    save(fig, "fig_speedup_v9")


# ============================================================================
#  FIGURE 3 — Wilcoxon verdicts over the 100 cases
# ============================================================================
def fig_wilcoxon():
    src = R('results_pairwise_method', 'pairwise_all100.csv')
    d = pd.read_csv(src)
    refs = [('EVO', 'SO-EVO vs.\nEVO'),
            (REWALL, 'SO-EVO vs.\nSO-EVO crediting all four')]
    plus, zero, minus, labs = [], [], [], []
    for col, lab in refs:
        c = d[col].value_counts()
        plus.append(int(c.get('+', 0)))
        zero.append(int(c.get('0', 0)))
        minus.append(int(c.get('-', 0)))
        labs.append(lab)
    print(f"  [wilcoxon]  {rel(src)}  ({len(d)} cases)  -> " +
          ", ".join(f"{l.split(chr(10))[-1]} {p}/{z}/{mm}"
                    for l, p, z, mm in zip(labs, plus, zero, minus)))

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    y = np.arange(len(labs))
    ax.barh(y, plus, color=SOEVO, label='SO-EVO faster ($+$)')
    ax.barh(y, zero, left=plus, color=GREY, label='No difference (0)')
    ax.barh(y, minus, left=np.array(plus) + np.array(zero), color=EVO,
            label='SO-EVO slower ($-$)')
    for i in range(len(labs)):
        ax.text(plus[i] / 2, i, plus[i], ha='center', va='center', color='white',
                fontweight='bold')
        ax.text(plus[i] + zero[i] / 2, i, zero[i], ha='center', va='center',
                color='#444', fontweight='bold')
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=11); ax.invert_yaxis()
    ax.set_xlabel(f'Number of cases (out of {len(d)})')
    ax.set_xlim(0, len(d))
    ax.set_title('Paired Wilcoxon signed-rank verdicts on NFE-to-VTR',
                 fontsize=13.5)
    ax.legend(ncol=3, frameon=False, fontsize=10, loc='upper center',
              bbox_to_anchor=(.5, -.20))
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_wilcoxon_v9", bbox='tight')


# ============================================================================
#  FIGURE 4 — credit rule, mean NFE
# ============================================================================
def fig_credit(m):
    arms = [(METHOD, 'selective'),
            (REWALL, 'all four'),
            ('SOEVO_eps_sel', 'selective,\n$\\epsilon$-greedy'),
            ('SOEVO_eps_all', 'all four,\n$\\epsilon$-greedy')]
    # Average every arm over ONE case set: the cases on which all four arms
    # and EVO reach the VTR in at least one run (2026-10-02, review item A1).
    arms = [(a, lab) for a, lab in arms if a in set(m.Algorithm)]
    _algos = [a for a, _ in arms] + ['EVO']
    _piv = (m[m.Algorithm.isin(_algos)]
            .pivot_table(index=['Function', 'D'], columns='Algorithm',
                         values='mean_NFE').reindex(columns=_algos))
    _common = _piv.dropna(how='any').index
    m = m.set_index(['Function', 'D']).loc[_common].reset_index()
    print(f"  [credit]  common case set: {len(_common)} cases")
    labs, vals, cols = [], [], []
    for a, lab in arms:
        s = m[m.Algorithm == a]
        if s.empty:
            continue
        labs.append(lab); vals.append(s.mean_NFE.mean())
        cols.append(SOEVO if 'selective' in lab else '#f2a0a6')
    evo_val = m[m.Algorithm == 'EVO'].mean_NFE.mean()
    print("  [credit]  results_field/field_master.csv  -> " +
          ", ".join(f"{l.replace(chr(10),' ')} {v:,.0f}" for l, v in zip(labs, vals)) +
          f" ; EVO {evo_val:,.0f}")

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(labs))
    ax.bar(x, vals, width=.62, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 40, f'{v:,.0f}', ha='center', fontsize=11.5,
                fontweight='bold', color='#333')
    gain = 100 * (vals[1] - vals[0]) / vals[1]
    ybar = max(vals) * 1.10
    ax.annotate('', xy=(0, ybar), xytext=(1, ybar),
                arrowprops=dict(arrowstyle='<->', color='#444', lw=1.4))
    ax.text(.5, ybar * 1.015, f'{gain:.1f}% fewer evaluations', ha='center',
            fontsize=11, fontweight='bold', color='#444')
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=11)
    ax.set_ylabel('Mean evaluations to VTR')
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title('Crediting only the surviving move is the cheaper rule',
                 fontsize=13.5)
    ax.text(.99, .96, f'EVO, same cases: {evo_val:,.0f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=10.5, color=EVO, style='italic')
    ax.grid(axis='y', alpha=.3)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_credit_v9")


# ============================================================================
#  FIGURE 5 — Friedman / Nemenyi critical-difference diagrams
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
    save(fig, "fig_cd_diagram_v9", bbox='tight')


# ============================================================================
#  FIGURE 6 — wall-clock timing
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
    save(fig, "fig_timing_v9")


# ============================================================================
#  FIGURE 7 — sensitivity to n_s
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
    save(fig, "fig_sensitivity_v9")


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
