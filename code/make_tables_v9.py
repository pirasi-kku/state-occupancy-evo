# -*- coding: utf-8 -*-
"""
make_tables_v9.py
=================
Generates every LaTeX result table of the manuscript from the CSV files in
../data/, so no number is transcribed by hand. Writes one .tex file per table
into ../tables/.

The per-case tables collapse the dimension axis: every function is evaluated at
its smallest dimension, then 10, 30, 50, and 10 of them also at 100 (Powell
Singular 2 needs a dimension divisible by four and uses 8/12/32/52/100), so
columns are labelled D1..D5 by position with the actual values in the caption.
The appendix tables list every algorithm on every case.

Run:
    python make_tables_v9.py
"""

import os, sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_RES  = os.path.join(_ROOT, 'data')        # supplementary layout
OUT   = os.path.join(_ROOT, 'tables')
os.makedirs(OUT, exist_ok=True)

METHOD = 'SOEVO_sel'            # occupancy + selective credit
REWALL = 'SOEVO_all'            # occupancy + credit to all four
COMPARATORS = ['ACO', 'HS', 'FA', 'MVO', 'ISA', 'CSA']

FUNC_LABEL = {
    'F01_Ackley1': 'Ackley 1', 'F02_Alpine1': 'Alpine 1',
    'F03_ChungReynolds': 'Chung Reynolds', 'F04_Exponential': 'Exponential',
    'F05_InvCosWave': 'Inverted Cosine Wave', 'F06_Pinter': 'Pint\\\'er',
    'F07_Rastrigin': 'Rastrigin', 'F08_Salomon': 'Salomon',
    'F09_Schwefel12': 'Schwefel 1.2', 'F10_Schwefel221': 'Schwefel 2.21',
    'F11_Griewank': 'Griewank', 'F12_PowellSingular2': 'Powell Singular 2',
    'F13_SchumerSteiglitz': 'Schumer Steiglitz', 'F14_Schwefel24': 'Schwefel 2.4',
    'F15_Schwefel225': 'Schwefel 2.25', 'F16_Sphere': 'Sphere',
    'F17_Step3': 'Step 3', 'F18_Trigonometric2': 'Trigonometric 2',
    'F19_Wavy': 'Wavy', 'F20_XinSheYang6': 'Xin-She Yang 6',
}


def load_main():
    """
    Main benchmark, one row per (function, dimension, algorithm).

    results_field/field_master.csv is the single source: B1_field_merge.py
    assembles it from every campaign and it carries all 100 cases for all
    eleven algorithms. Nothing here re-merges the campaign folders, so no table
    can be built from a different subset than any other table.
    """
    return pd.read_csv(os.path.join(_RES, 'results_field', 'field_master.csv'))


def load_fixedfe():
    """
    Fixed-budget benchmark: mean and best final objective value after the full
    150,000 evaluations, one row per (function, dimension, algorithm). Covers
    the 90 cases at D<=50 together with D=100 for F11--F20.
    """
    p = os.path.join(_RES, 'results_field_fixedfe', 'field_fixedfe_master.csv')
    return pd.read_csv(p) if os.path.exists(p) else None


def load_friedman(sub, fname):
    p = os.path.join(_RES, sub, fname)
    return pd.read_csv(p) if os.path.exists(p) else None


def load_timing():
    """Timing results: the whole grid timed in one session, SO-EVO and EVO."""
    p = os.path.join(_RES, 'results_timing', 'timing_master.csv')
    if not os.path.exists(p):
        raise FileNotFoundError(p + '')
    return pd.read_csv(p)


def table_timing():
    """Wall-clock table, generated rather than hand-written."""
    t = load_timing()
    dims = sorted(t.D.unique())
    ev, me = {}, {}
    for D in dims:
        s = t[t.D == D]
        ev[D] = s[s.Algorithm == 'EVO']
        me[D] = s[s.Algorithm == METHOD]

    def row(label, fn, bold_last=False):
        cells = []
        for i, D in enumerate(dims):
            v = fn(D)
            txt = '--' if v != v else f'{v:.2f}'
            if bold_last and i == len(dims) - 1 and v == v:
                txt = f'\\textbf{{{txt}}}'
            cells.append(txt)
        return f'{label} & ' + ' & '.join(cells) + ' \\\\'

    nfun = t[t.Algorithm == 'EVO'].groupby('D').Function.nunique().min()
    rows = [
        row('EVO (s)',      lambda D: ev[D].mean_secs.mean()),
        row('\\pevo{} (s)', lambda D: me[D].mean_secs.mean()),
        row('Speedup',      lambda D: ev[D].mean_secs.mean() / me[D].mean_secs.mean(), True),
        row('NFE ratio',    lambda D: ev[D].mean_NFE.mean() / me[D].mean_NFE.mean(), True),
    ]
    head = '$D$ & ' + ' & '.join(str(int(d)) for d in dims) + ' \\\\'
    cap = ('Wall-clock time to reach the VTR and the corresponding evaluation '
           f'counts, averaged over {nfun} functions at each dimension and 10 runs '
           'each. Speedup is EVO/SO-EVO in seconds; NFE ratio is the same '
           'quotient in evaluations. All cells were measured in one session.')
    body = header(cap, 'tab:timing', 'l' + 'r' * len(dims), head)
    return body + '\n'.join(rows) + '\n' + FOOTER


def dim_slots(df):
    """Map each (function, D) to a positional slot 1..5."""
    slots = {}
    for f, g in df.groupby('Function'):
        for i, d in enumerate(sorted(g.D.unique()), 1):
            slots[(f, d)] = i
    return slots


def fmt_nfe(v, sr):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '--'
    s = f'{v:,.0f}'.replace(',', '\\,')
    if sr is not None and np.isfinite(sr) and sr < 100:
        s = s + f' ({sr:.0f}\\%)'
    return s


def header(caption, label, colspec, head, size='small'):
    return (f'\\begin{{table}}[htbp]\n\\centering\\{size}\n'
            f'\\caption{{{caption}}}\n\\label{{{label}}}\n'
            f'\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{head}\n\\midrule\n')


FOOTER = '\\bottomrule\n\\end{tabular}\n\\end{table}\n'


# ==============================================================================
#  Table: SO-EVO vs EVO, mean NFE per case
# ==============================================================================

def table_main_percase():
    m = load_main()
    slots = dim_slots(m)
    m['slot'] = [slots[(f, d)] for f, d in zip(m.Function, m.D)]
    verdict = f'{REWALL}_vs_this'

    # Two rows per function (SO-EVO above, EVO below) and five dimension
    # columns: the ten-column SO/EVO layout ran past the text width once the
    # success rate moved from a superscript into parentheses (2026-11-05).
    rows = []
    for f in sorted(m.Function.unique()):
        g = m[m.Function == f]
        so, ev = [], []
        for s in range(1, 6):
            a = g[(g.slot == s) & (g.Algorithm == METHOD)]
            b = g[(g.slot == s) & (g.Algorithm == 'EVO')]
            if a.empty:
                so.append(''); ev.append('')
                continue
            so.append(fmt_nfe(a.mean_NFE.values[0], a.SR.values[0]))
            ev.append(fmt_nfe(b.mean_NFE.values[0], b.SR.values[0])
                      if not b.empty else '--')
        rows.append(f'{f.split("_")[0]} & SO-EVO & ' + ' & '.join(so) + ' \\\\')
        rows.append(' & EVO & ' + ' & '.join(ev) + ' \\\\[2pt]')

    head = 'ID & & $D_1$ & $D_2$ & $D_3$ & $D_4$ & $D_5$ \\\\'
    cap = ('Mean number of function evaluations to reach the VTR, SO-EVO '
           'against EVO, over 50 runs; function names are given in '
           'Table~\\ref{tab:functions}. Dimension slots are $D_1$--$D_5$ = '
           '$5,10,30,50,100$, except Powell Singular~2 which requires a '
           'dimension divisible by four and uses $8,12,32,52,100$. A blank cell '
           'marks a dimension not run for that function. A percentage in '
           'parentheses gives the success rate when it '
           'is below $100\\%$; ``--\'\' marks a case in which no run reached '
           'the VTR.')
    body = header(cap, 'tab:main_percase', 'll' + 'r' * 5, head,
                  size='footnotesize')
    return body + '\n'.join(rows) + '\n' + FOOTER


# ==============================================================================
#  Table: speedup of SO-EVO over EVO per case
# ==============================================================================

def table_speedup():
    m = load_main()
    slots = dim_slots(m)
    m['slot'] = [slots[(f, d)] for f, d in zip(m.Function, m.D)]

    rows = []
    for f in sorted(m.Function.unique()):
        g = m[m.Function == f]
        cells = []
        for s in range(1, 6):
            a = g[(g.slot == s) & (g.Algorithm == METHOD)]
            b = g[(g.slot == s) & (g.Algorithm == 'EVO')]
            if a.empty or b.empty:
                cells.append(''); continue
            x, y = a.mean_NFE.values[0], b.mean_NFE.values[0]
            if not (np.isfinite(x) and np.isfinite(y)) or x <= 0:
                cells.append('--'); continue
            r = y / x
            cells.append(f'\\textbf{{{r:.1f}}}' if r >= 2 else f'{r:.1f}')
        rows.append(f'{FUNC_LABEL.get(f, f)} & ' + ' & '.join(cells) + ' \\\\')

    head = 'Function & $D_1$ & $D_2$ & $D_3$ & $D_4$ & $D_5$ \\\\'
    cap = ('Evaluation-count ratio EVO/SO-EVO per case; values above one favor '
           'SO-EVO. Bold marks a ratio of at least two. Dimension slots as in '
           'Table~\\ref{tab:main_percase}; ``--\'\' marks a case in which one '
           'of the two never reached the VTR.')
    body = header(cap, 'tab:speedup', 'l' + 'r' * 5, head)
    return body + '\n'.join(rows) + '\n' + FOOTER


# ==============================================================================
#  Table: credit-assignment ablation
# ==============================================================================

def common_cases(m, algos):
    """
    (Function, D) pairs on which every algorithm in `algos` has a finite
    mean_NFE, i.e. reached the VTR in at least one run. Means compared across
    algorithms are taken over this one set, so no average mixes case sets
    (2026-10-02, review item A1).
    """
    sub = m[m.Algorithm.isin(algos)]
    piv = sub.pivot_table(index=['Function', 'D'], columns='Algorithm',
                          values='mean_NFE')
    piv = piv.reindex(columns=algos)
    return piv.dropna(how='any').index


def table_credit():
    m = load_main()
    arms = [(METHOD, 'selective'), (REWALL, 'all four'),
            ('SOEVO_eps_sel', 'selective, $\\epsilon$-greedy sampling'),
            ('SOEVO_eps_all', 'all four, $\\epsilon$-greedy sampling'),
            ('EVO', 'EVO (no occupancy learning)')]
    arms = [(a, lab) for a, lab in arms if a in set(m.Algorithm)]
    common = common_cases(m, [a for a, _ in arms])
    mc = m.set_index(['Function', 'D']).loc[common].reset_index()
    rows = []
    for a, lab in arms:
        s = mc[mc.Algorithm == a]
        nfe = s.mean_NFE.mean(); sr = s.SR.mean()
        rows.append(f'{lab} & {nfe:,.0f} & {sr:.1f} \\\\'.replace(',', '\\,'))
    head = 'Credit rule & Mean NFE & Mean SR (\\%) \\\\'
    ncase = len(common)
    cap = (f'Credit assignment, averaged over the {ncase} cases on which every '
           'row reaches the VTR in at least one run. Mean NFE is taken over the '
           'successful runs. Every row uses the '
           'same operators, sampling and maximum number of function evaluations, '
           'and differs only in which '
           'occupancy vectors are incremented on an accepted move.')
    body = header(cap, 'tab:credit', 'lrr', head)
    return body + '\n'.join(rows) + '\n' + FOOTER


# ==============================================================================
#  Table: engineering
# ==============================================================================

def table_engineering():
    p = os.path.join(_RES, 'results_engineering', 'engineering_master.csv')
    if not os.path.exists(p):
        return None
    e = pd.read_csv(p)
    # Static penalty for every row except the two marked "Deb's rules": crediting
    # all four was run only under Deb's feasibility rules, so SO-EVO is shown
    # under both rules to keep the credit comparison like for like. EVO is the
    # EVO_penalty arm, which shares SO-EVO's constrained implementation.
    keep = ['SOEVO_sel_penalty', 'SOEVO_sel_feasibility', 'SOEVO_all_feasibility',
            'EVO_penalty', 'ACO', 'HS', 'FA', 'MVO', 'ISA', 'CSA']
    lab = {'SOEVO_sel_penalty': 'SO-EVO',
           'SOEVO_sel_feasibility': "SO-EVO, Deb's rules",
           'SOEVO_all_feasibility': "SO-EVO (all four), Deb's rules",
           'EVO_penalty': 'EVO'}
    rows = []
    probs = sorted(e.Problem.unique())
    for a in keep:
        s = e[e.Arm == a]
        if s.empty:
            continue
        cells = []
        for pr in probs:
            r = s[s.Problem == pr]
            if r.empty or not np.isfinite(r.gap_pct.values[0]):
                cells.append('--')
            else:
                cells.append(f'{r.gap_pct.values[0]:.1f}')
        rows.append(f'{lab.get(a, a)} & ' + ' & '.join(cells) + ' \\\\')
    short = [pr.split('_')[0] for pr in probs]
    head = 'Algorithm & ' + ' & '.join(short) + ' \\\\'
    cap = ('Gap to the published best-known value (\\%) on the CEC~2020 '
           'engineering problems, best over 30 runs, feasible runs only. '
           'Every row uses the static penalty for acceptance except the two '
           'marked Deb\'s rules, which use the feasibility rules of '
           '\\citet{Deb2000}. '
           '``--\'\' marks a problem on which no run ended feasible.')
    body = header(cap, 'tab:engineering', 'l' + 'r' * len(probs), head,
                  size='footnotesize')
    return body + '\n'.join(rows) + '\n' + FOOTER


# ==============================================================================
#  Table: paired Wilcoxon verdicts
# ==============================================================================

def table_pairwise():
    p = os.path.join(_RES, 'results_pairwise_method', 'pairwise_all100.csv')
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    rows = []
    for col, lab in [('EVO', 'EVO'),
                     (REWALL, 'SO-EVO crediting all four')]:
        if col not in d.columns:
            continue
        c = d[col].value_counts()
        rows.append(f'{lab} & {int(c.get("+", 0))} & {int(c.get("0", 0))} '
                    f'& {int(c.get("-", 0))} \\\\')
    head = ('Reference & \\pevo{} faster & No difference & \\pevo{} slower \\\\')
    n_all = int(d[REWALL].isin(['+', '0', '-']).sum()) if REWALL in d.columns else 0
    cap = (f'Paired Wilcoxon signed-rank verdicts on NFE-to-VTR over the '
           f'{len(d)} cases ({n_all} for the all-four variant, which was not run '
           'on F01--F10 at $D=100$) at the 0.05 significance level, paired by run index over the '
           'runs in which both algorithms reached the VTR. A case on which the '
           'two return identical values on every run is counted as no '
           'difference.')
    body = header(cap, 'tab:pairwise', 'lccc', head)
    return body + '\n'.join(rows) + '\n' + FOOTER


# ==============================================================================
#  Table: sensitivity to n_s
# ==============================================================================

def table_sensitivity():
    p = os.path.join(_RES, 'results_sensitivity', 'sensitivity_ns.csv')
    if not os.path.exists(p):
        raise FileNotFoundError(p + '')
    d = pd.read_csv(p)
    d = d[(d.Arm == METHOD) & (d.Sweep == 'ns')]
    if d.empty:
        return None
    g = d.groupby('Value').agg(nfe=('mean_NFE', 'mean'), sr=('SR', 'mean'))
    g = g.sort_index()
    vals = [int(v) for v in g.index]
    r1 = 'Mean NFE & ' + ' & '.join(f'{v:,.0f}'.replace(',', '\\,')
                                    for v in g.nfe) + ' \\\\'
    r2 = 'Mean SR (\\%) & ' + ' & '.join(f'{v:.0f}' for v in g.sr) + ' \\\\'
    head = '$n_s$ & ' + ' & '.join(str(v) for v in vals) + ' \\\\'
    nf = d.Function.nunique()
    ncase = d[['Function', 'D']].drop_duplicates().shape[0]
    cap = (f'Sensitivity to the number of states, averaged over {ncase} cases '
           f'({nf} functions at four dimensions) with 30 runs each.')
    body = header(cap, 'tab:sensitivity', 'l' + 'r' * len(vals), head)
    return body + r1 + '\n' + r2 + '\n' + FOOTER


# ==============================================================================
#  Tables: Friedman mean ranks
# ==============================================================================

ALGO_LABEL = {METHOD: '\\pevo{}', 'EVO': 'EVO'}


def _rank_table(fr, label, caption):
    if fr is None:
        return None
    fr = fr.sort_values('MeanRank')
    names = [ALGO_LABEL.get(a, a) for a in fr.Algorithm]
    vals = []
    for i, v in enumerate(fr.MeanRank):
        vals.append(f'\\textbf{{{v:.2f}}}' if i == 0 else f'{v:.2f}')
    head = '& ' + ' & '.join(names) + ' \\\\'
    row = 'Mean rank & ' + ' & '.join(vals) + ' \\\\'
    body = header(caption, label, 'l' + 'c' * len(names), head)
    return body + row + '\n' + FOOTER


def table_friedman():
    fr = load_friedman('results_field', 'field_friedman.csv')
    if fr is None:
        return None
    cd = fr.CD.iloc[0]
    n = load_main().groupby(['Function', 'D']).ngroups
    cap = (f'Friedman mean ranks on penalized expected cost over the {n} cases, '
           f'lower is better. Critical difference {cd:.3f} at the 0.05 '
           'significance level (Nemenyi).')
    return _rank_table(fr, 'tab:friedman', cap)


def table_friedman_fixedfe():
    fr = load_friedman('results_field_fixedfe', 'field_fixedfe_friedman.csv')
    if fr is None:
        return None
    cd = fr.CD.iloc[0]
    f = load_fixedfe()
    n = f.groupby(['Function', 'D']).ngroups if f is not None else 90
    cap = (f'Friedman mean ranks on the final objective value after '
           f'$150{{,}}000$ evaluations, over the {n} cases, lower is better. Critical difference '
           f'{cd:.3f} at the 0.05 significance level (Nemenyi).')
    return _rank_table(fr, 'tab:friedman_fixedfe', cap)


# ==============================================================================
#  Tables: fixed budget, final objective value
# ==============================================================================

def fmt_fit(v):
    """Final objective values span 10^-16 to 10^5, so a fixed format is no use."""
    if v is None or not np.isfinite(v):
        return '--'
    if v == 0:
        return '0'
    s = f'{v:.1e}'                       # e.g. 3.9e-15
    mant, exp = s.split('e')
    e = int(exp)
    return f'{mant}e{{{"-" if e < 0 else "+"}}}{abs(e)}'


def table_fixedfe_percase():
    f = load_fixedfe()
    if f is None:
        return None
    slots = dim_slots(f)
    f = f.copy()
    f['slot'] = [slots[(fn, d)] for fn, d in zip(f.Function, f.D)]
    nslot = int(f.slot.max())

    # Two rows per function (SO-EVO above, EVO below), as in
    # table_main_percase (2026-11-05).
    rows = []
    for fn in sorted(f.Function.unique()):
        g = f[f.Function == fn]
        so, ev = [], []
        for s in range(1, nslot + 1):
            a = g[(g.slot == s) & (g.Algorithm == METHOD)]
            b = g[(g.slot == s) & (g.Algorithm == 'EVO')]
            so.append(fmt_fit(a.mean_fit.values[0]) if not a.empty else '')
            ev.append(fmt_fit(b.mean_fit.values[0]) if not b.empty else '')
        rows.append(f'{fn.split("_")[0]} & SO-EVO & ' + ' & '.join(so) + ' \\\\')
        rows.append(' & EVO & ' + ' & '.join(ev) + ' \\\\[2pt]')

    head = 'ID & & ' + ' & '.join(f'$D_{i}$' for i in range(1, nslot + 1)) + ' \\\\'
    cap = ('Mean final objective value after $150{,}000$ function '
           'evaluations, SO-EVO against EVO. Dimension slots as in '
           'Table~\\ref{tab:main_percase}; a blank cell marks a case not run at '
           'that slot. Function identifiers are those of '
           'Table~\\ref{tab:functions}.')
    body = header(cap, 'tab:fixedfe_percase', 'll' + 'r' * nslot, head,
                  size='footnotesize')
    return body + '\n'.join(rows) + '\n' + FOOTER


def table_fixedfe_wins():
    """
    Who attains the best final value on a case. Most cases are shared, because
    several algorithms reach the numerical floor, so an outright win and a
    share of the best value are counted separately rather than awarding every
    tie to one algorithm.
    """
    f = load_fixedfe()
    if f is None:
        return None
    algos = [a for a in [METHOD, 'EVO'] + COMPARATORS if a in set(f.Algorithm)]
    piv = f.pivot_table(index=['Function', 'D'], columns='Algorithm',
                        values='mean_fit')[algos].dropna(how='any')
    mn = piv.min(axis=1)
    attains = piv.le(mn + 1e-12 * np.maximum(1.0, mn.abs()), axis=0)
    n_at_best = attains.sum(axis=1)

    rec = []
    for a in algos:
        out = int((attains[a] & (n_at_best == 1)).sum())
        sh = int((attains[a] & (n_at_best > 1)).sum())
        rec.append((a, out, sh))
    rec.sort(key=lambda r: (-(r[1] + r[2]), -r[1]))
    rows = [f'{"SO-EVO" if a == METHOD else a} & {o + s} & {o} \\\\'
            for a, o, s in rec]
    head = 'Algorithm & Attains best & Outright \\\\'
    cap = (f'Cases out of {len(piv)} on which each algorithm attains the best '
           'mean final objective value at MaxFEs${}=150{,}000$. Outright counts '
           'the cases on which it is strictly better than every other '
           'algorithm; the remainder are shared with at least one other '
           'algorithm at the same value.')
    body = header(cap, 'tab:fixedfe_wins', 'lcc', head)
    return body + '\n'.join(rows) + '\n' + FOOTER


def grouped_rows(rows):
    """rows: list of (function label, D, list of cell strings), sorted by
    function. The function name is written once per group as a \\multirow
    cell, rows inside a group end in \\\\* so a group is never split across
    pages, and a \\midrule separates consecutive functions (2026-11-05)."""
    out, i = [], 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j][0] == rows[i][0]:
            j += 1
        n = j - i
        for k in range(i, j):
            lab, d, cells = rows[k]
            first = f'\\multirow{{{n}}}{{*}}{{{lab}}}' if k == i else ''
            end = ' \\\\*' if k < j - 1 else ' \\\\'
            out.append(f'{first} & {int(d)} & ' + ' & '.join(cells) + end)
        if j < len(rows):
            out.append('\\midrule')
        i = j
    return out


def table_fixedfe_appendix():
    f = load_fixedfe()
    if f is None:
        return None
    algos = [a for a in [METHOD, 'EVO'] + COMPARATORS if a in set(f.Algorithm)]
    piv = f.pivot_table(index=['Function', 'D'], columns='Algorithm',
                        values='mean_fit')[algos]
    lines = grouped_rows([(FUNC_LABEL.get(fn, fn), d, [fmt_fit(v) for v in r.values])
                          for (fn, d), r in piv.iterrows()])
    names = ['SO-EVO' if a == METHOD else a for a in algos]
    head = 'Function & $D$ & ' + ' & '.join(names) + ' \\\\'
    # Portrait since 2026-11-05: footnotesize with 3pt column separation fits
    # the text width; LTcapwidth keeps the caption inside the margins.
    out = ['{\\footnotesize\\setlength{\\tabcolsep}{3pt}'
           '\\setlength{\\LTcapwidth}{\\textwidth}'
           '\\renewcommand{\\arraystretch}{0.97}',
           '\\begin{longtable}{ll' + 'r' * len(algos) + '}',
           '\\caption{Mean final objective value for every case and algorithm '
           'after $150{,}000$ function evaluations, without early termination; '
           'every value is the mean over all 50 runs.}'
           '\\label{tab:fixedfe_full}\\\\',
           '\\toprule', head, '\\midrule', '\\endfirsthead',
           '\\toprule', head, '\\midrule', '\\endhead',
           '\\bottomrule', '\\endfoot']
    out += lines
    out += ['\\end{longtable}', '}']
    return '\n'.join(out) + '\n'


# ==============================================================================
#  Appendix: full field, landscape longtable
# ==============================================================================

def table_field_appendix():
    p = os.path.join(_RES, 'results_field', 'field_master.csv')
    if not os.path.exists(p):
        return None
    f = pd.read_csv(p)
    algos = [METHOD, 'EVO'] + COMPARATORS
    algos = [a for a in algos if a in set(f.Algorithm)]
    piv = f.pivot_table(index=['Function', 'D'], columns='Algorithm',
                        values='mean_NFE')
    # reindex rather than select: an algorithm that never reaches the VTR (FA)
    # has an all-NaN column, which pivot_table drops; keep it as all dashes
    # so the table covers every algorithm its caption names (2026-11-05).
    piv = piv.reindex(columns=algos)
    # A case in which no algorithm ever reaches the VTR has no NFE for anyone
    # and is dropped by pivot_table. Put those rows back, as all-dashes, so the
    # table shows every case that was run rather than only the solvable ones.
    every = pd.MultiIndex.from_frame(
        f[['Function', 'D']].drop_duplicates().sort_values(['Function', 'D']))
    piv = piv.reindex(every)

    lines = grouped_rows([
        (FUNC_LABEL.get(fn, fn), d,
         [('--' if not np.isfinite(v) else f'{v:,.0f}'.replace(',', '\\,'))
          for v in r.values])
        for (fn, d), r in piv.iterrows()])

    names = ['SO-EVO' if a == METHOD else a for a in piv.columns]
    head = 'Function & $D$ & ' + ' & '.join(names) + ' \\\\'
    ncol = len(piv.columns)
    # Portrait since 2026-11-05: footnotesize with 3pt column separation fits
    # the text width; LTcapwidth keeps the caption inside the margins.
    out = ['{\\footnotesize\\setlength{\\tabcolsep}{3pt}'
           '\\setlength{\\LTcapwidth}{\\textwidth}'
           '\\renewcommand{\\arraystretch}{0.97}',
           '\\begin{longtable}{ll' + 'r' * ncol + '}',
           '\\caption{Mean number of function evaluations to reach the VTR for '
           'every case and algorithm, taken over the successful runs; the number '
           'of successful runs behind each mean is not shown here (success rates '
           'of SO-EVO and EVO are given in Table~\\ref{tab:main_percase}). '
           '``--\'\' marks a case in which no run '
           'reached the VTR.}\\label{tab:field_full}\\\\',
           '\\toprule', head, '\\midrule', '\\endfirsthead',
           '\\toprule', head, '\\midrule', '\\endhead',
           '\\bottomrule', '\\endfoot']
    out += lines
    out += ['\\end{longtable}', '}']
    return '\n'.join(out) + '\n'


# ==============================================================================
#  Table: the benchmark functions
# ==============================================================================

#  Type, range and optimum are properties of the function definition and are
#  written out here; the dimension list is read from the results, so the table
#  cannot disagree with the cases actually run.
FUNC_META = {
    'F01_Ackley1':          ('M', '$[-32,32]$',           '$0$'),
    'F02_Alpine1':          ('M', '$[-10,10]$',           '$0$'),
    'F03_ChungReynolds':    ('U', '$[-100,100]$',         '$0$'),
    'F04_Exponential':      ('U', '$[-1,1]$',             '$-1$'),
    'F05_InvCosWave':       ('M', '$[-5,5]$',             '$-(D{-}1)$'),
    'F06_Pinter':           ('M', '$[-10,10]$',           '$0$'),
    'F07_Rastrigin':        ('M', '$[-5.12,5.12]$',       '$0$'),
    'F08_Salomon':          ('M', '$[-100,100]$',         '$0$'),
    'F09_Schwefel12':       ('U', '$[-100,100]$',         '$0$'),
    'F10_Schwefel221':      ('U', '$[-100,100]$',         '$0$'),
    'F11_Griewank':         ('M', '$[-600,600]$',         '$0$'),
    'F12_PowellSingular2':  ('U', '$[-4,5]$',             '$0$'),
    'F13_SchumerSteiglitz': ('U', '$[-100,100]$',         '$0$'),
    'F14_Schwefel24':       ('U', '$[-100,100]$',         '$0$'),
    'F15_Schwefel225':      ('U', '$[-100,100]$',         '$0$'),
    'F16_Sphere':           ('U', '$[-100,100]$',         '$0$'),
    'F17_Step3':            ('U', '$[-100,100]$',         '$0$'),
    'F18_Trigonometric2':   ('M', '$[-500,500]$',         '$1$'),
    'F19_Wavy':             ('M', '$[-\\pi,\\pi]$',         '$0$'),
    'F20_XinSheYang6':      ('M', '$[-2\\pi,2\\pi]$',       '$-1$'),
}


def table_functions():
    m = load_main()
    dims = {f: sorted(g.D.unique()) for f, g in m.groupby('Function')}
    rows = []
    for f in sorted(FUNC_META):
        typ, rng, opt = FUNC_META[f]
        d = ','.join(str(int(x)) for x in dims.get(f, []))
        rows.append(f'{f.split("_")[0]} & {FUNC_LABEL[f]} & {typ} & {rng} '
                    f'& {opt} & ${d}$ \\\\')
    cap = ('The twenty benchmark functions. U: unimodal; M: multimodal. '
           'Sources:~\\citep{Ackley1987,Rastrigin1974,Salomon1996,Pinter1996,'
           'Schwefel1981}.')
    head = 'ID & Function & Type & Range $[lb,ub]$ & $f^*$ & Dimensions $D$ \\\\'
    return (header(cap, 'tab:functions', 'llcccc', head)
            + '\n'.join(rows) + '\n' + FOOTER)


# ==============================================================================

def main():
    jobs = [
        ('tab_functions.tex',    table_functions),
        ('tab_main_percase.tex', table_main_percase),
        ('tab_credit.tex',       table_credit),
        ('tab_fixedfe_percase.tex',  table_fixedfe_percase),
        ('tab_fixedfe_wins.tex',     table_fixedfe_wins),
        ('tab_fixedfe_appendix.tex', table_fixedfe_appendix),
        ('tab_engineering.tex',  table_engineering),
        ('tab_field_appendix.tex', table_field_appendix),
    ]
    for name, fn in jobs:
        try:
            body = fn()
        except Exception as exc:
            print(f'  [FAIL] {name}: {exc}')
            continue
        if body is None:
            print(f'  [skip] {name}: source data not present yet')
            continue
        with open(os.path.join(OUT, name), 'w', encoding='utf-8') as fh:
            fh.write(body)
        print(f'  wrote tables/{name}  ({len(body.splitlines())} lines)')


if __name__ == '__main__':
    main()
