"""
Implementation of the six-port calibration algorithm from:

  K. Staszek, "Six-Port Calibration Utilizing Matched Load and Unknown
  Calibration Loads," IEEE Trans. Microw. Theory Techn., vol. 66, no. 10,
  pp. 4617-4626, Oct. 2018.

Equation numbers in the comments refer to that paper.

Model (eq. 10):   p_{i,k} = q_i * |1 + Abar_i * gamma_k|^2 ,  i = 1..3
where gamma_k = Gamma_k / Gamma_1 (loads normalized to the first one) and
Abar_i = A_i * Gamma_1.

Public API
----------
calibrate(p_matched, P, expected_slope=...)  -> dict with 'q', 'A_cal', ...
measure_gamma(cal, P)                        -> complex reflection coefficient(s)
solve_gamma(A, r)                            -> low-level 3-circle solver
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar


# ----------------------------------------------------------------------------
# Low-level solver: given three complex "centers" A_i and three measured radii
# r_i = |1 + A_i*gamma|, solve the linear system for gamma.
# This is exactly eq. (39)-(45) of the paper (and identically (54)-(60) for
# the measurement stage).
# ----------------------------------------------------------------------------
def solve_gamma(A, r):
    """Solve |1 + A_i*gamma| = r_i (i = 1,2,3) for gamma.

    Parameters
    ----------
    A : (3,) complex array - the centers (Abar_i or candidate values).
    r : (3,) or (3, K) float array - measured radii r_{i,k}, eq. (13).

    Returns
    -------
    gamma : scalar complex or (K,) complex array.
    """
    
    A = np.asarray(A, dtype=complex)
    r = np.asarray(r, dtype=float)
    a1, a2, a3 = A
    m = np.abs(A) ** 2

    C = 2 * a2.real / m[1] - 2 * a1.real / m[0]          # (40)
    Dc = 2 * a1.imag / m[0] - 2 * a2.imag / m[1]         # (41)
    F = 2 * a3.real / m[2] - 2 * a2.real / m[1]          # (43)
    G = 2 * a2.imag / m[1] - 2 * a3.imag / m[2]          # (44)
    E = (r[1] ** 2 - 1.0) / m[1] - (r[0] ** 2 - 1.0) / m[0]   # (42)
    H = (r[2] ** 2 - 1.0) / m[2] - (r[1] ** 2 - 1.0) / m[1]   # (45)

    den = C * G - Dc * F
    return (E * G - Dc * H + 1j * (C * H - E * F)) / den      # (39)


# ----------------------------------------------------------------------------
# Sec. III-D: geometric construction of the B points for candidate {d12, d13}
# ----------------------------------------------------------------------------
def _triangle_d(r, d12, d13, phi3):
    """For each load k and each candidate angle phi3 build the B_{i,k} points
    (eq. 14-16, 20-22) and return the resulting normalized distances
    d12^k, d13^k (eq. 19).

    r    : (3, K) radii
    phi3 : (M,) candidate angles of B3 in [-pi, pi)
    Returns (dk12, dk13), each (K, M); NaN where the triangle is impossible.
    """
    r1 = r[0][:, None]
    r2 = r[1][:, None]
    r3 = r[2][:, None]
    p3 = phi3[None, :]

    with np.errstate(invalid="ignore", divide="ignore"):
        # law of cosines between B1 (on the real axis) and B3      (20)
        D13 = np.sqrt(np.maximum(r1 ** 2 + r3 ** 2 - 2 * r1 * r3 * np.cos(p3), 0.0))
        D12 = D13 * (d12 / d13)                                  # (21)
        cosp2 = (r1 ** 2 + r2 ** 2 - D12 ** 2) / (2 * r1 * r2)   # (22)
        invalid = np.abs(cosp2) > 1.0            # complex phi2 -> reject (22)
        p2 = np.arccos(np.clip(cosp2, -1.0, 1.0))                # phi2 in [0,180deg] (17)
        B2 = r2 * np.exp(1j * p2)                                # (15)
        B3 = r3 * np.exp(1j * p3)                                # (16)
        D23 = np.abs(B2 - B3)
        S = D12 + D23 + D13
        dk12 = D12 / S                                           # (19)
        dk13 = D13 / S

    dk12 = np.where(invalid | (S <= 0), np.nan, dk12)
    dk13 = np.where(invalid | (S <= 0), np.nan, dk13)
    return dk12, dk13


def _cost_Y(r, d12, d13, phi3):
    """Cost function Y(d12_est, d13_est), eq. (23)-(24)."""
    if d12 <= 1e-6 or d13 <= 1e-6 or d12 + d13 >= 1.0:
        return np.inf
    dk12, dk13 = _triangle_d(r, d12, d13, phi3)
    res = np.abs(dk12 - d12) + np.abs(dk13 - d13)                # (24)
    res = np.where(np.isfinite(res), res, np.inf)
    yk = res.min(axis=1)
    if not np.all(np.isfinite(yk)):
        return np.inf
    return float(yk.sum())                                       # (23)


# ----------------------------------------------------------------------------
# Sec. III-E: boundary conditions for the {d12, d13} search
# ----------------------------------------------------------------------------
def _d_bounds(r):
    """Eq. (27)-(36). Returns ((lo12, hi12), (lo13, hi13))."""
    r1, r2, r3 = r
    Dmin = {(1, 2): np.abs(r1 - r2), (1, 3): np.abs(r1 - r3), (2, 3): np.abs(r2 - r3)}
    Dmax = {(1, 2): r1 + r2, (1, 3): r1 + r3, (2, 3): r2 + r3}

    den_A = Dmin[(1, 2)] + Dmin[(1, 3)] + Dmin[(2, 3)]           # (29)
    den_B = Dmax[(1, 2)] + Dmax[(1, 3)] + Dmin[(2, 3)]           # (30)
    den_C = Dmin[(1, 2)] + Dmax[(1, 3)] + Dmax[(2, 3)]           # (31)
    den_D = Dmax[(1, 2)] + Dmin[(1, 3)] + Dmax[(2, 3)]           # (32)

    out = []
    for key in ((1, 2), (1, 3)):
        with np.errstate(divide="ignore", invalid="ignore"):
            cand = np.stack([Dmin[key] / den_A, Dmax[key] / den_B,
                             Dmin[key] / den_C, Dmax[key] / den_D])
        cand = np.where(np.isfinite(cand), cand, np.nan)
        kmin = np.nanmin(cand, axis=0)                           # (33)
        kmax = np.nanmax(cand, axis=0)                           # (34)
        lo = float(np.nanmax(kmin))                              # (35)
        hi = float(np.nanmin(kmax))                              # (36)
        lo = max(lo, 1e-3)
        hi = min(hi, 0.98)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.02, 0.65                                  # safe fallback
        out.append((lo, hi))
    return out[0], out[1]


def _bset_at(r_k1, d12, d13, phi3):
    """Build a B point set (eq. 25/26) for load k = 1 at a given phi3."""
    r1, r2, r3 = r_k1
    D13 = np.sqrt(max(r1 ** 2 + r3 ** 2 - 2 * r1 * r3 * np.cos(phi3), 0.0))
    D12 = D13 * (d12 / d13)
    cosp2 = np.clip((r1 ** 2 + r2 ** 2 - D12 ** 2) / (2 * r1 * r2), -1.0, 1.0)
    p2 = np.arccos(cosp2)
    return np.array([r1, r2 * np.exp(1j * p2), r3 * np.exp(1j * phi3)])


# ----------------------------------------------------------------------------
# Main calibration routine (Sec. III-A ... III-H)
# ----------------------------------------------------------------------------
def calibrate(p_matched, P, expected_slope="negative",
              n_phi_coarse=241, n_phi_fine=1441, n_grid=25, n_alpha=720):
    """Calibrate a six-port system from a matched-load reading and K readings
    of UNKNOWN calibration loads.

    Parameters
    ----------
    p_matched : (3,) powers measured with the matched load (Gamma = 0).
    P         : (3, K) powers for the K unknown loads (k = 1 is the reference
                load to which everything is normalized).
    expected_slope : 'negative' or 'positive' - a-priori sign of the linear
                trend of the loads' phases vs. k (Sec. III-G / eq. 49-50).
                For loads made of increasing transmission-line lengths, or a
                radar target moved away from the antenna, use 'negative'.

    Returns dict with:
        q       - (3,) power-normalization constants, eq. (3)
        A_cal   - (3,) calibrated Abar_i, eq. (53)
        gammas  - (K,) reconstructed normalized calibration loads
        d12,d13 - normalized triangle distances
        conjugated, residual, alpha, set_used - diagnostics
    """
    q = np.asarray(p_matched, dtype=float)                       # (3)
    P = np.asarray(P, dtype=float)
    r = np.sqrt(P / q[:, None])                                  # (13)
    K = r.shape[1]

    # ---------- Sec. III-D + III-E : find d12, d13 -------------------------
    phi3c = np.linspace(-np.pi, np.pi, n_phi_coarse, endpoint=False)
    (lo12, hi12), (lo13, hi13) = _d_bounds(r)

    def coarse_search(a12, b12, a13, b13, n):
        best_val, best_x = np.inf, None
        for d12 in np.linspace(a12, b12, n):
            for d13 in np.linspace(a13, b13, n):
                v = _cost_Y(r, d12, d13, phi3c)
                if v < best_val:
                    best_val, best_x = v, (d12, d13)
        return best_val, best_x

    best_val, best_x = coarse_search(lo12, hi12, lo13, hi13, n_grid)
    if best_x is None:  # bounds were too tight -> generic wide search
        best_val, best_x = coarse_search(0.01, 0.7, 0.01, 0.7, n_grid)
    if best_x is None:
        raise RuntimeError("d12/d13 search failed: no valid triangle found")

    phi3f = np.linspace(-np.pi, np.pi, n_phi_fine, endpoint=False)
    res = minimize(lambda x: _cost_Y(r, x[0], x[1], phi3f), best_x,
                   method="Nelder-Mead",
                   options=dict(xatol=1e-7, fatol=1e-13, maxiter=500))
    d12, d13 = float(res.x[0]), float(res.x[1])

    # ---------- two candidate B sets from load k = 1 (eq. 25-26) -----------
    dk12, dk13 = _triangle_d(r[:, :1], d12, d13, phi3f)
    y1 = (np.abs(dk12 - d12) + np.abs(dk13 - d13))[0]
    y1 = np.where(np.isfinite(y1), y1, np.inf)

    M = len(phi3f)
    minima = [i for i in range(M)
              if np.isfinite(y1[i])
              and y1[i] <= y1[(i - 1) % M] and y1[i] <= y1[(i + 1) % M]]
    minima.sort(key=lambda i: y1[i])
    # keep minima that are angularly distinct
    sel = []
    for i in minima:
        if all(abs(np.angle(np.exp(1j * (phi3f[i] - phi3f[j])))) > np.deg2rad(5)
               for j in sel):
            sel.append(i)
        if len(sel) == 2:
            break
    if not sel:
        raise RuntimeError("no valid B-point candidates for load k=1")

    step = phi3f[1] - phi3f[0]
    Bsets = []
    for i in sel:
        # local 1-D refinement of phi3 around the grid minimum
        def f1(p3):
            a, b = _triangle_d(r[:, :1], d12, d13, np.array([p3]))
            v = abs(a[0, 0] - d12) + abs(b[0, 0] - d13)
            return v if np.isfinite(v) else 1e9
        rr = minimize_scalar(f1, bounds=(phi3f[i] - step, phi3f[i] + step),
                             method="bounded", options=dict(xatol=1e-10))
        Bsets.append(_bset_at(r[:, 0], d12, d13, float(rr.x)))
    if len(Bsets) == 1:
        Bsets.append(Bsets[0])

    # ---------- Sec. III-F : rotation & selection (eq. 37-38, 46) ----------
    def rot_resid(B, alpha):
        a = B * np.exp(1j * alpha) - 1.0
        if np.min(np.abs(a)) < 1e-9:
            return np.inf
        g = solve_gamma(a, r)                                    # gamma_k via (39)
        if not np.all(np.isfinite(g)):
            return np.inf
        model = np.abs(1.0 + a[:, None] * g[None, :])
        return float(np.sum((model - r) ** 2))                   # (37)-(38)

    alphas = np.linspace(0.0, 2 * np.pi, n_alpha, endpoint=False)
    dalp = alphas[1] - alphas[0]
    best = (np.inf, None, None)   # (residual, B, alpha)
    for tag, B in enumerate(Bsets):
        vals = np.array([rot_resid(B, a) for a in alphas])
        i0 = int(np.argmin(vals))
        rr = minimize_scalar(lambda a: rot_resid(B, a),
                             bounds=(alphas[i0] - dalp, alphas[i0] + dalp),
                             method="bounded", options=dict(xatol=1e-10))
        if rr.fun < best[0]:
            best = (float(rr.fun), tag, float(rr.x))
    resid_rot, set_used, alpha = best
    B_rot = Bsets[set_used] * np.exp(1j * alpha) - 1.0           # (46)

    # ---------- Sec. III-G : complex-conjugation test (eq. 47-50) ----------
    g = solve_gamma(B_rot, r)
    inc = np.angle(g[1:] / g[:-1])
    Theta = np.angle(g[0]) + np.concatenate([[0.0], np.cumsum(inc)])  # (47)-(48)
    kk = np.arange(1, K + 1, dtype=float)
    a_slope = np.polyfit(kk, Theta, 1)[0]                        # (49)
    if expected_slope == "negative":
        conj_needed = a_slope > 0
    else:
        conj_needed = a_slope < 0
    B_rotC = np.conj(B_rot) if conj_needed else B_rot            # (50)

    # ---------- Sec. III-H : final refinement (eq. 51-53) ------------------
    def ref_resid(dphi):
        a = (B_rotC + 1.0) * np.exp(1j * dphi) - 1.0
        if np.min(np.abs(a)) < 1e-9:
            return 1e9
        gg = solve_gamma(a, r)
        if not np.all(np.isfinite(gg)):
            return 1e9
        model = np.abs(1.0 + a[:, None] * gg[None, :])
        return float(np.sum((model - r) ** 2))                   # (51)-(52)

    rr = minimize(ref_resid, np.zeros(3), method="Nelder-Mead",
                  options=dict(xatol=1e-9, fatol=1e-15, maxiter=800))
    A_cal = (B_rotC + 1.0) * np.exp(1j * rr.x) - 1.0             # (53)

    return dict(q=q, A_cal=A_cal, d12=d12, d13=d13,
                gammas=solve_gamma(A_cal, r),
                conjugated=bool(conj_needed), residual=float(rr.fun),
                alpha=alpha, set_used=set_used, slope=float(a_slope))


def measure_gamma(cal, P):
    """Measurement stage, eq. (54)-(60): reflection coefficient(s) normalized
    to the first calibration load. P: (3,) or (3, K) power readings."""
    r = np.sqrt(np.asarray(P, dtype=float) / cal["q"][:, None]
                if np.ndim(P) == 2 else
                np.asarray(P, dtype=float) / cal["q"])
    return solve_gamma(cal["A_cal"], r)
