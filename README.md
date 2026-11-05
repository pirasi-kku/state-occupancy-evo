# SO-EVO: A State-Occupancy-Guided Energy Valley Optimizer

Result files, source code and generated artefacts for

> **SO-EVO: A State-Occupancy-Guided Energy Valley Optimizer**
> Pirapong Singsathid, Department of Mathematics, Faculty of Science,
> Khon Kaen University.

Every table cell and every plotted point in the manuscript is computed from the
CSV files in `data/` by the two generator scripts in `code/`. Nothing is
transcribed by hand. Running those scripts on this folder reproduces the files
in `tables/` byte for byte and the figures in `figures/`.

```
.
├── data/       result files, grouped by experiment, one row per (case, algorithm)
├── code/       the algorithm, the experiment harness, the table and figure generators
├── tables/     the LaTeX tables in the manuscript, as generated
├── figures/    the figures in the manuscript, as generated (PDF, with PNG previews)
├── LICENSE
└── README.md   this file
```

## Naming

The CSV files use internal keys rather than display names. All SO-EVO variants
are implemented in `code/soevo.py` (dictionary `ARMS`).

| key in the CSVs   | manuscript name              | what it is                                            |
|-------------------|------------------------------|-------------------------------------------------------|
| `SOEVO_sel`       | **SO-EVO**                   | occupancy sampling, credit to the surviving move       |
| `SOEVO_all`       | SO-EVO crediting all four    | occupancy sampling, credit to all four coefficients    |
| `SOEVO_eps_sel`   | selective, ε-greedy          | ε-greedy sampling, selective credit                    |
| `SOEVO_eps_all`   | all four, ε-greedy           | ε-greedy sampling, credit to all four                  |
| `EVO`             | EVO                          | the base algorithm                                     |
| `ACO HS FA MVO ISA CSA` | same                   | the six comparators                                    |

In `engineering_master.csv` the SO-EVO and EVO keys carry a suffix naming the
constraint-handling rule of that run (`_penalty`, `_feasibility`, `_epsilon`);
the plain `EVO` key and the six comparators use the static penalty.

## `data/`

Main benchmark: twenty functions at five dimensions each (100 cases), 50 runs per
case, maximum number of function evaluations 150,000. A run succeeds when
$f - f^* \le 10^{-10}$ (the VTR), with $f^* = 0$ except F04 ($-1$), F05
($-(D-1)$), F18 ($1$) and F20 ($-1$); `evo_core.run_case` applies this test.

| file | rows | contents |
|---|---|---|
| `results_field/field_master.csv` | 1090 | mean NFE to VTR, success rate and penalized expected cost per case and algorithm. Tables 5, 6 and A1, Figure 1, and the evaluation ratios of Section 5.1 |
| `results_field/field_friedman.csv` | 8 | Friedman mean ranks and the Nemenyi critical difference over the 100 cases. Figure 2(a), Section 5.3 |
| `results_field/field_summary.csv` | 8 | per-algorithm means over the 100 cases |
| `results_pairwise_method/pairwise_all100.csv` | 100 | paired Wilcoxon verdict (+ / 0 / −), p-value and number of pairs per case, against EVO and against crediting all four. The counts reported in Sections 5.1 and 5.2 |

Runs without early termination: the same functions run to 150,000 evaluations.

| file | rows | contents |
|---|---|---|
| `results_field_fixedfe/field_fixedfe_master.csv` | 900 | mean and best final objective value per case and algorithm. Tables 7, 8 and B1 |
| `results_field_fixedfe/field_fixedfe_friedman.csv` | 8 | mean ranks and critical difference. Figure 2(b), Section 5.6 |
| `results_field_fixedfe/field_fixedfe_wins.csv` | 8 | how often each algorithm attains the best final value (the counts of Table 8) |

Supporting experiments:

| file | rows | contents |
|---|---|---|
| `results_sensitivity/sensitivity_ns.csv` | 120 | sweep of $n_s \in \{10,15,20,25,30\}$ over six functions at four dimensions, 30 runs each. Figure 3, Section 5.4 |
| `results_timing/timing_master.csv` | 50 | wall-clock seconds and evaluations to VTR for SO-EVO and EVO, five functions × five dimensions, 10 runs each, all measured in one session on one machine. Figure 4, Section 5.5 |
| `results_engineering/engineering_master.csv` | 104 | eight CEC 2020 constrained engineering problems, 30 runs each: feasibility rate, best and mean objective, gap to the best-known value. Table 9 |
| `results_engineering/feasibility_rates.csv`, `gaps.csv` | 8 each | the same results pivoted per problem |

Rerun of the four functions with $f^* \neq 0$:

| file | rows | contents |
|---|---|---|
| `results_vtrfix/vtrfix_master.csv` | 218 | F04, F05, F18, F20 at every dimension, every algorithm, with the absolute target used (`f_star`, `VTR_abs`) and the NFE of each of the 50 runs (`nfe_runs`, null for a run that did not succeed). These rows replace the corresponding rows of `field_master.csv` and feed the Wilcoxon verdicts for those 20 cases |

Wall-clock numbers are comparable only within `timing_master.csv`, which is why
the timing figure is built from that file alone.

## `code/`

| file | what it does |
|---|---|
| `soevo.py` | the algorithm: one parameterised implementation covering every variant; `sampling` selects the variant and `credit` the rule |
| `evo_core.py` | the benchmark functions, the base EVO implementation and the six comparators, plus shared helpers (nearest neighbour, boundary handling) |
| `soevo_constrained.py`, `constraint_handling.py`, `cec2020_problems.py` | the constrained variant, the constraint-handling rules and the eight engineering problems |
| `_harness.py` | the experiment harness: matched seeding, NFE statistics, paired Wilcoxon, penalized cost, Friedman ranks and the Nemenyi critical difference |
| `make_tables_v9.py` | regenerates every generated `.tex` file in `tables/` from `data/` |
| `make_figures_v9.py` | regenerates every figure in `figures/` from `data/` |

Runs are seeded from one shared seed table (`_harness.matched_seeds`), so every
algorithm sees the same starting populations on the same case and the paired
tests are paired by run index.

To regenerate the tables and figures, run from inside `code/`:

```
python make_tables_v9.py
python make_figures_v9.py
```

Both read from `../data/` and write into `../tables/` and `../figures/`, and
print the source file behind every artefact they write.

## `tables/` and `figures/`

`tab_params.tex` (Table 3) and `tab_eng_setup.tex` (Table 4) list experimental
settings and are written by hand; every other file in `tables/` is generated.
Table 1 (coefficients entering each candidate) is part of the method
description and is typed in the manuscript.

| table file | manuscript |
|---|---|
| `tab_functions.tex` | Table 2, the twenty benchmark functions |
| `tab_params.tex` | Table 3, algorithms and their control parameters |
| `tab_eng_setup.tex` | Table 4, engineering design problems |
| `tab_main_percase.tex` | Table 5, mean NFE to the VTR, SO-EVO against EVO |
| `tab_credit.tex` | Table 6, credit assignment |
| `tab_fixedfe_percase.tex` | Table 7, final objective value, SO-EVO against EVO |
| `tab_fixedfe_wins.tex` | Table 8, cases on which each algorithm attains the best final value |
| `tab_engineering.tex` | Table 9, gap to the best-known value on the engineering problems |
| `tab_field_appendix.tex` | Table A1, mean NFE to the VTR for every algorithm |
| `tab_fixedfe_appendix.tex` | Table B1, final objective value for every algorithm |

| figure file | manuscript |
|---|---|
| `fig_speedup_v9.pdf` | Fig. 1, per-case evaluation-count ratio EVO/SO-EVO by dimension |
| `fig_cd_diagram_v9.pdf` | Fig. 2, critical-difference diagrams |
| `fig_sensitivity_v9.pdf` | Fig. 3, sensitivity to $n_s$ |
| `fig_timing_v9.pdf` | Fig. 4, wall-clock cost |

Requirements: Python 3.10 or later with `numpy`, `scipy`, `pandas` and
`matplotlib`.

## License

MIT, see `LICENSE`.
