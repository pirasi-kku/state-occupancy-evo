# -*- coding: utf-8 -*-
"""
cec2020_problems.py
===================
Python port of eight problems from the CEC 2020 real-world constrained
optimization suite, transcribed from the reference MATLAB implementation
(cec20_func.m and Cal_par.m, A. Kumar, IIT (BHU) Varanasi;
https://github.com/P-N-Suganthan/2020-RW-Constrained-Optimisation):

    RC15 speed reducer            RC21 multiple disk clutch brake
    RC17 tension/compression      RC27 ten-bar truss (natural-frequency
         spring                        constraints, generalized eigenproblem)
    RC19 welded beam              RC28 rolling element bearing (maximization)
    RC20 three-bar truss          RC31 gear train

Each problem exposes:
    obj(x)    -> float          objective exactly as the MATLAB returns it
    con(x)    -> np.ndarray     inequality constraints, feasible when g <= 0
    lb, ub    -> np.ndarray     bounds from Cal_par.m
    D, n_con  -> int
    sense     -> 'min' | 'max'
    best_known-> float          published best-known value

For a minimising solver use `penalised(prob)` or negate when sense == 'max'.
"""

import numpy as np
from scipy.linalg import eigh


# ==============================================================================
#  RC15 — Weight Minimization of a Speed Reducer   (D=7, g=11)
# ==============================================================================

def rc15_obj(x):
    x1, x2, x3, x4, x5, x6, x7 = x
    return float(
        0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
        - 1.508 * x1 * (x6**2 + x7**2)
        + 7.477 * (x6**3 + x7**3)
        + 0.7854 * (x4 * x6**2 + x5 * x7**2))


def rc15_con(x):
    x1, x2, x3, x4, x5, x6, x7 = x
    g = np.empty(11)
    g[0] = -x1 * x2**2 * x3 + 27.0
    g[1] = -x1 * x2**2 * x3**2 + 397.5
    g[2] = -x2 * x6**4 * x3 * x4**(-3) + 1.93
    g[3] = -x2 * x7**4 * x3 / x5**3 + 1.93
    g[4] = 10 * x6**(-3) * np.sqrt(16.91e6 + (745.0 * x4 / (x2 * x3))**2) - 1100.0
    g[5] = 10 * x7**(-3) * np.sqrt(157.5e6 + (745.0 * x5 / (x2 * x3))**2) - 850.0
    g[6] = x2 * x3 - 40.0
    g[7] = -x1 / x2 + 5.0
    g[8] = x1 / x2 - 12.0
    g[9] = 1.5 * x6 - x4 + 1.9
    g[10] = 1.1 * x7 - x5 + 1.9
    return g


# ==============================================================================
#  RC17 — Tension/compression spring design (case 1)   (D=3, g=4 in MATLAB)
# ==============================================================================

def rc17_obj(x):
    x1, x2, x3 = x
    return float(x1**2 * x2 * (x3 + 2))


def rc17_con(x):
    x1, x2, x3 = x
    g = np.empty(4)
    g[0] = 1 - (x2**3 * x3) / (71785.0 * x1**4)
    g[1] = (4 * x2**2 - x1 * x2) / (12566.0 * (x2 * x1**3 - x1**4)) \
           + 1.0 / (5108.0 * x1**2) - 1
    g[2] = 1 - 140.45 * x1 / (x2**2 * x3)
    g[3] = (x1 + x2) / 1.5 - 1
    return g


# ==============================================================================
#  RC19 — Welded beam design   (D=4, g=5)
# ==============================================================================

def rc19_obj(x):
    x1, x2, x3, x4 = x
    return float(1.10471 * x1**2 * x2 + 0.04811 * x3 * x4 * (14 + x2))


def rc19_con(x):
    x1, x2, x3, x4 = x
    P, L, delta_max = 6000.0, 14.0, 0.25
    E, G = 30e6, 12e6
    T_max, sigma_max = 13600.0, 30000.0

    Pc = 4.013 * E * np.sqrt(x3**2 * x4**6 / 30.0) / L**2 \
         * (1 - x3 / (2 * L) * np.sqrt(E / (4 * G)))
    sigma = 6.0 * P * L / (x4 * x3**2)
    delta = 6.0 * P * L**3 / (E * x3**2 * x4)
    J = 2.0 * (np.sqrt(2) * x1 * x2 * (x2**2 / 4.0 + (x1 + x3)**2 / 4.0))
    R = np.sqrt(x2**2 / 4.0 + (x1 + x3)**2 / 4.0)
    M = P * (L + x2 / 2.0)
    ttt = M * R / J
    tt = P / (np.sqrt(2) * x1 * x2)
    t = np.sqrt(tt**2 + 2 * tt * ttt * x2 / (2 * R) + ttt**2)

    g = np.empty(5)
    g[0] = t - T_max
    g[1] = sigma - sigma_max
    g[2] = x1 - x4
    g[3] = delta - delta_max
    g[4] = P - Pc
    return g


# ==============================================================================
#  RC20 — Three-bar truss design   (D=2, g=3)
# ==============================================================================

def rc20_obj(x):
    x1, x2 = x
    return float((2 * np.sqrt(2) * x1 + x2) * 100.0)


def rc20_con(x):
    # The lower bound is 0 for both variables, and x1 = 0 makes the stress
    # denominators vanish (MATLAB would return Inf/NaN there). Such points are
    # physically meaningless (zero cross-section), so they are reported as
    # heavily infeasible rather than allowed to poison the comparison with NaN.
    x1, x2 = x
    g = np.empty(3)
    den = np.sqrt(2) * x1**2 + 2 * x1 * x2
    den2 = np.sqrt(2) * x2 + x1
    if den <= 0.0 or den2 <= 0.0:
        return np.full(3, 1e6)
    g[0] = (np.sqrt(2) * x1 + x2) / den * 2 - 2
    g[1] = x2 / den * 2 - 2
    g[2] = 1.0 / den2 * 2 - 2
    return np.where(np.isfinite(g), g, 1e6)


# ==============================================================================
#  RC21 — Multiple disk clutch brake design   (D=5, g=8 in MATLAB)
# ==============================================================================

def rc21_obj(x):
    x1, x2, x3, x4, x5 = x
    rho = 0.0000078
    return float(np.pi * (x2**2 - x1**2) * x3 * (x5 + 1) * rho)


def rc21_con(x):
    x1, x2, x3, x4, x5 = x
    Mf, Ms, Iz, n = 3.0, 40.0, 55.0, 250.0
    Tmax, s, delta = 15.0, 1.5, 0.5
    Vsrmax, pmax, mu, Lmax, delR = 10.0, 1.0, 0.6, 30.0, 20.0

    Rsr = 2.0 / 3.0 * (x2**3 - x1**3) / (x2**2 * x1**2)
    Vsr = np.pi * Rsr * n / 30.0
    A = np.pi * (x2**2 - x1**2)
    Prz = x4 / A
    w = np.pi * n / 30.0
    Mh = 2.0 / 3.0 * mu * x4 * x5 * (x2**3 - x1**3) / (x2**2 - x1**2)
    T = Iz * w / (Mh + Mf)

    g = np.empty(8)
    g[0] = -x2 + x1 + delR
    g[1] = (x5 + 1) * (x3 + delta) - Lmax
    g[2] = Prz - pmax
    g[3] = Prz * Vsr - pmax * Vsrmax
    g[4] = Vsr - Vsrmax
    g[5] = T - Tmax
    g[6] = s * Ms - Mh
    g[7] = -T
    return g


# ==============================================================================
#  RC27 — 10-bar truss design with frequency constraints   (D=10, g=3)
#         port of function_fitness / ConsBar10 / Cal_K_and_M / eigens
# ==============================================================================

_RC27_GCOORD = np.array([[18.288, 18.288, 9.144, 9.144, 0.0, 0.0],
                         [9.144,  0.0,    9.144, 0.0,   9.144, 0.0]])
# MATLAB 1-based node numbers per element (2 x 10)
_RC27_ELEMENT = np.array([[3, 1, 4, 2, 3, 1, 4, 3, 2, 1],
                          [5, 3, 6, 4, 4, 2, 5, 6, 3, 4]]) - 1   # -> 0-based
_RC27_E = 6.98e10
_RC27_RHO = 2770.0
_RC27_ADDED_MASS = 454.0


def _rc27_lengths():
    g = _RC27_GCOORD
    le = np.empty(10)
    for i in range(10):
        n1, n2 = _RC27_ELEMENT[0, i], _RC27_ELEMENT[1, i]
        le[i] = np.hypot(g[0, n2] - g[0, n1], g[1, n2] - g[1, n1])
    return le


_RC27_LE = _rc27_lengths()


def rc27_obj(x):
    """Weight = sum(rho * le_i * A_i)."""
    A = np.asarray(x, dtype=float)
    return float(np.sum(_RC27_RHO * _RC27_LE * A))


def _rc27_K_M(A):
    """Assemble 2D truss stiffness and (consistent) mass matrices."""
    g = _RC27_GCOORD
    sdof = 6 * 2
    K = np.zeros((sdof, sdof))
    M = np.zeros((sdof, sdof))
    for iel in range(10):
        n1, n2 = _RC27_ELEMENT[0, iel], _RC27_ELEMENT[1, iel]
        dx = g[0, n2] - g[0, n1]
        dy = g[1, n2] - g[1, n1]
        le = np.hypot(dx, dy)
        l_ij, m_ij = dx / le, dy / le
        Te = np.array([[l_ij, m_ij, 0.0, 0.0],
                       [0.0, 0.0, l_ij, m_ij]])
        ke_local = A[iel] * _RC27_E / le * np.array([[1.0, -1.0],
                                                     [-1.0, 1.0]])
        ke = Te.T @ ke_local @ Te
        me = _RC27_RHO * le * A[iel] * np.array([[2.0, 0.0, 1.0, 0.0],
                                                 [0.0, 2.0, 0.0, 1.0],
                                                 [1.0, 0.0, 2.0, 0.0],
                                                 [0.0, 1.0, 0.0, 2.0]]) / 6.0
        idx = np.array([2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1])
        K[np.ix_(idx, idx)] += ke
        M[np.ix_(idx, idx)] += me
    return K, M


def rc27_con(x):
    """g1 = 7/f1 - 1, g2 = 15/f2 - 1, g3 = 20/f3 - 1  (f in Hz)."""
    A = np.asarray(x, dtype=float)
    K, M = _rc27_K_M(A)
    sdof = K.shape[0]
    M = M + np.eye(sdof) * _RC27_ADDED_MASS      # non-structural mass

    # MATLAB bcdof = [(5:6)*2-1, (5:6)*2] -> 1-based dofs 9,11,10,12
    bcdof = np.array([9, 11, 10, 12]) - 1        # -> 0-based
    fdof = np.setdiff1d(np.arange(sdof), bcdof)

    Kf = K[np.ix_(fdof, fdof)]
    Mf = M[np.ix_(fdof, fdof)]
    try:
        w2 = eigh(Kf, Mf, eigvals_only=True)
    except Exception:
        return np.full(3, 1e6)
    w2 = np.sort(np.real(w2))
    w2 = np.clip(w2, 1e-30, None)
    freq = np.sqrt(w2) / (2.0 * np.pi)

    g = np.empty(3)
    g[0] = 7.0 / freq[0] - 1.0
    g[1] = 15.0 / freq[1] - 1.0
    g[2] = 20.0 / freq[2] - 1.0
    return g


# ==============================================================================
#  RC28 — Rolling element bearing   (D=10, g=9)   MAXIMISATION
# ==============================================================================

def _rc28_fc(x):
    Dm, Db, Z, fi, fo = x[0], x[1], round(x[2]), x[3], x[4]
    gamma = Db / Dm
    a = (1.0 + (1.04 * ((1 - gamma) / (1 + gamma))**1.72
                * (fi * (2 * fo - 1) / (fo * (2 * fi - 1)))**0.41)**(10.0 / 3.0))**(-0.3)
    b = (gamma**0.3 * (1 - gamma)**1.39 / (1 + gamma)**(1.0 / 3.0))
    c = (2 * fi / (2 * fi - 1))**0.41
    return 37.91 * a * b * c


def rc28_obj(x):
    """Dynamic load capacity C (to be MAXIMISED)."""
    Db = x[1]
    Z = round(x[2])
    fc = _rc28_fc(x)
    if Db > 25.4:
        return float(3.647 * fc * Z**(2.0 / 3.0) * Db**1.4)
    return float(fc * Z**(2.0 / 3.0) * Db**1.8)


def rc28_con(x):
    Dm, Db = x[0], x[1]
    Z = round(x[2])
    fi, fo = x[3], x[4]
    KDmin, KDmax, eps, e, chi = x[5], x[6], x[7], x[8], x[9]
    D, d, Bw = 160.0, 90.0, 30.0
    T = D - d - 2.0 * Db

    num = ((D - d) * 0.5 - 0.75 * T)**2 + (0.5 * D - 0.25 * T - Db)**2 \
          - (0.5 * d + 0.25 * T)**2
    den = 2.0 * (0.5 * (D - d) - 0.75 * T) * (0.5 * D - 0.25 * T - Db)
    phi_o = 2.0 * np.pi - 2.0 * np.arccos(np.clip(num / den, -1.0, 1.0))

    g = np.empty(9)
    g[0] = Z - 1 - phi_o / (2.0 * np.arcsin(np.clip(Db / Dm, -1.0, 1.0)))
    g[1] = KDmin * (D - d) - 2.0 * Db
    g[2] = 2.0 * Db - KDmax * (D - d)
    g[3] = chi * Bw - Db
    g[4] = 0.5 * (D + d) - Dm
    g[5] = Dm - (0.5 + e) * (D + d)
    g[6] = eps * Db - 0.5 * (D - Dm - Db)
    g[7] = 0.515 - fi
    g[8] = 0.515 - fo
    return g


# ==============================================================================
#  RC31 — Gear train design   (D=4, no inequality constraints)
# ==============================================================================

def rc31_obj(x):
    x1, x2, x3, x4 = x
    return float((1.0 / 6.931 - (x1 * x2) / (x3 * x4))**2)


def rc31_con(x):
    return np.zeros(0)


# ==============================================================================
#  Registry — bounds taken verbatim from Cal_par.m,
#             best_known from Table 3 of Problem-Definitions.pdf
# ==============================================================================

CEC2020 = {
    'RC15_SpeedReducer': dict(
        obj=rc15_obj, con=rc15_con, D=7, n_con=11, sense='min',
        lb=np.array([2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0]),
        ub=np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5]),
        best_known=2.9944244658e+03,
        desc='Weight Minimization of a Speed Reducer'),

    'RC17_Spring': dict(
        obj=rc17_obj, con=rc17_con, D=3, n_con=4, sense='min',
        lb=np.array([0.05, 0.25, 2.00]),
        ub=np.array([2.0, 1.3, 15.0]),
        best_known=1.2665232788e-02,
        desc='Tension/compression spring design (case 1)'),

    'RC19_WeldedBeam': dict(
        obj=rc19_obj, con=rc19_con, D=4, n_con=5, sense='min',
        lb=np.array([0.125, 0.1, 0.1, 0.1]),
        ub=np.array([2.0, 10.0, 10.0, 2.0]),
        best_known=1.6702177263e+00,
        desc='Welded beam design'),

    'RC20_ThreeBarTruss': dict(
        obj=rc20_obj, con=rc20_con, D=2, n_con=3, sense='min',
        lb=np.array([0.0, 0.0]),
        ub=np.array([1.0, 1.0]),
        best_known=2.6389584338e+02,
        desc='Three-bar truss design problem'),

    'RC21_ClutchBrake': dict(
        obj=rc21_obj, con=rc21_con, D=5, n_con=8, sense='min',
        lb=np.array([60.0, 90.0, 1.0, 0.0, 2.0]),
        ub=np.array([80.0, 110.0, 3.0, 1000.0, 9.0]),
        best_known=2.3524245790e-01,
        desc='Multiple disk clutch brake design problem'),

    'RC27_TenBarTruss': dict(
        obj=rc27_obj, con=rc27_con, D=10, n_con=3, sense='min',
        lb=np.full(10, 0.645e-4),
        ub=np.full(10, 50e-4),
        best_known=5.2445076066e+02,
        desc='10-bar truss design with frequency constraints'),

    # NOTE ON SENSE: Problem-Definitions.pdf states "Maximize" for the load
    # capacity, but the competition harness minimises whatever cec20_func.m
    # returns, and it returns C unnegated. Verified numerically: the published
    # f(x*) = 14614.14 lies BELOW the minimum feasible value found by random
    # search (22314) and is approached from above when minimising (17035 after
    # 40k evaluations), whereas the feasible maximum is ~58148. The benchmark
    # therefore treats RC28 as a minimisation of the returned value.
    'RC28_RollingBearing': dict(
        obj=rc28_obj, con=rc28_con, D=10, n_con=9, sense='min',
        lb=np.array([125.0, 10.5, 4.51, 0.515, 0.515, 0.4, 0.6, 0.3, 0.02, 0.6]),
        ub=np.array([150.0, 31.5, 50.49, 0.6, 0.6, 0.5, 0.7, 0.4, 0.1, 0.85]),
        best_known=1.4614135715e+04,
        desc='Rolling element bearing'),

    'RC31_GearTrain': dict(
        obj=rc31_obj, con=rc31_con, D=4, n_con=0, sense='min',
        lb=np.full(4, 12.0),
        ub=np.full(4, 60.0),
        best_known=0.0,
        desc='Gear train design problem'),
}


# ==============================================================================
#  Helpers
# ==============================================================================

def violation(prob, x):
    """Total inequality violation, 0 when feasible."""
    with np.errstate(all='ignore'):
        g = prob['con'](np.asarray(x, dtype=float))
    if g.size == 0:
        return 0.0
    g = np.where(np.isfinite(g), g, 1e6)
    return float(np.sum(np.maximum(0.0, g)))


def signed_obj(prob, x):
    """Objective oriented for MINIMISATION (negated when sense == 'max')."""
    v = prob['obj'](np.asarray(x, dtype=float))
    return -v if prob['sense'] == 'max' else v


def penalised(prob, x, coeff=1e6):
    return signed_obj(prob, x) + coeff * violation(prob, x)


def gap_to_best(prob, obj_value):
    """Relative gap to the published best-known value, on the natural scale."""
    bk = prob['best_known']
    if bk == 0:
        return abs(obj_value)
    return abs(obj_value - bk) / abs(bk)
