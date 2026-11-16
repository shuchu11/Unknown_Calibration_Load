"""
Reproduction (by simulation) of:

  K. Staszek, "Calibration of a Six-Port-Based CW Radar Using Unknown
  Positions of a Target," MIKON 2022.

The radar echo is modeled per eq. (1) of the paper:
    Gamma(D) = Gamma_0 + Gamma_T(D) * exp(-2*(alpha + j*beta)*D)
and the six-port power detectors per the standard model
    P_i = q_i * |1 + A_i * Gamma|^2 .

The calibration algorithm of the 2018 TMTT paper (sixport_calibration.py) is
applied with the N target positions acting as the K unknown calibration
loads. The "matched load" reading is obtained with no target in the field of
view (Gamma' = Gamma - Gamma_0 = 0), exactly as the spiral center Gamma_0 is
defined in the 2022 paper.

Outputs:
  fig1_spirals.png      - ideal vs. raw (uncalibrated) measured spiral
  fig2_demo.png         - single calibration demo (N=8, L=1.5*lambda)
  fig3_error_maps.png   - max / RMS distance-error maps vs. N and L (Fig. 6)
  results_summary.json  - key numbers
"""
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sixport_calibration import calibrate, measure_gamma, solve_gamma

# ---------------------------------------------------------------- constants
rng = np.random.default_rng(2022)
c0 = 299_792_458.0
f0 = 2.35e9
lam = c0 / f0 * 1e3            # wavelength in mm  (~127.57 mm)
beta = 2 * np.pi / lam         # rad / mm

D = np.round(np.arange(100.0, 420.0 + 1e-9, 0.1), 4)   # 3201 positions (Sec. V)
D1 = D[0]

# ------------------------------------------------- "unknown" radar hardware
# imbalanced six-port (magnitude/phase imbalance + detector sensitivities)
A = np.array([1.02 * np.exp(1j * np.deg2rad(8)),
              0.87 * np.exp(1j * np.deg2rad(122)),
              1.10 * np.exp(1j * np.deg2rad(-127))])
q = np.array([0.21, 0.26, 0.185])
Gamma0 = 0.06 * np.exp(1j * np.deg2rad(46))   # TX/RX antenna isolation (spiral center)
SIGMA_DB = 0.02                               # power-reading noise (std, dB)


def gamma_prime(d):
    """Target echo Gamma'(D) = Gamma_T*exp(-2*gamma*D), eq. (1) minus Gamma_0.
    Magnitude decays with distance AND is modulated to emulate non-uniform
    near-field illumination / varying RCS (the case the method must handle)."""
    mag = 0.55 * (100.0 / d) ** 1.1 * (1.0 + 0.08 * np.sin(2 * np.pi * d / 90.0))
    return mag * np.exp(1j * (0.7 - 2 * beta * d))


def read_powers(gam):
    """Noisy power readings P1..P3 (normalized by PREF) for reflection gam."""
    gam = np.atleast_1d(gam)
    P = q[:, None] * np.abs(1.0 + A[:, None] * gam[None, :]) ** 2
    P = P * 10 ** (rng.normal(0.0, SIGMA_DB, P.shape) / 10.0)        # 加上功率雜訊
    return P


# one full measurement campaign (like the 3201 acquisitions in the paper)
P_sweep = read_powers(Gamma0 + gamma_prime(D))
# "matched load": no target in the field of view -> Gamma' = 0
P_matched = read_powers(np.array([Gamma0]))[:, 0]

# ground truth in the normalized frame the algorithm works in:
#   p_i = q_i' |1 + A_i' Gamma'|^2  with  A_i' = A_i/(1+A_i*Gamma0)
A_eff = A / (1.0 + A * Gamma0)
Abar_true = A_eff * gamma_prime(D1)           # Abar_i = A_i' * Gamma'_1


def run_calibration(N, L_mm):
    """Calibrate with N equally spaced unknown positions spanning L_mm,
    then measure the full sweep and convert phase to distance (eq. 2)."""
    pos = D1 + L_mm * np.arange(N) / (N - 1)
    idx = np.clip(np.round((pos - D1) / 0.1).astype(int), 0, len(D) - 1)
    cal = calibrate(P_matched, P_sweep[:, idx], expected_slope="negative")
    g = measure_gamma(cal, P_sweep)
    ph = np.unwrap(np.angle(g))
    ph -= ph[0]                       # gamma is normalized to position 1
    D_est = D1 - ph * lam / (4 * np.pi)          # eq. (2)
    return cal, D_est, D_est - D, idx


# ============================================================ figure 1
gam_p = gamma_prime(D)
# raw estimate with an "ideal" six-port assumption (120-deg spaced, equal q)
A_ideal = np.exp(1j * np.deg2rad([0.0, 120.0, -120.0]))
r_raw = np.sqrt(P_sweep / P_matched[:, None])
g_raw = solve_gamma(A_ideal, r_raw)

fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
sub = slice(0, len(D), 8)
ax[0].plot(gam_p.real[sub], gam_p.imag[sub], lw=1)    # 沒 gamma_0 ，僅加上振幅衰減，再轉回 Gamma
ax[0].set_title("ideal echo $\\Gamma'(D)$  (true spiral)") 
ax[1].plot(g_raw.real[sub], g_raw.imag[sub], lw=1, color="tab:red")  # 有 gamma_0 ，Gamma轉換成功率，搭配理想 A 再轉換回 Gamma
ax[1].set_title("uncalibrated estimate (ideal-six-port assumption)")
for a in ax:
    a.set_xlabel("Real"); a.set_ylabel("Imag"); a.axis("equal"); a.grid(alpha=.3)
fig.suptitle("D = 100...420 mm, f = 2.35 GHz  (cf. paper Figs. 2-4)")
fig.tight_layout()
fig.savefig("fig1_spirals.png", dpi=150)
plt.close(fig)

# ============================================================ figure 2: demo
t0 = time.time()
N_demo, L_demo = 8, 1.5 * lam
cal, D_est, err, idx = run_calibration(N_demo, L_demo)
t_demo = time.time() - t0

gam_true_norm = gam_p / gam_p[idx[0]]        # true loads normalized to pos. 1
g_meas = measure_gamma(cal, P_sweep)

fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
ax[0].plot(gam_true_norm.real[sub], gam_true_norm.imag[sub], lw=1,
           label="true $\\Gamma'/\\Gamma'_1$")
ax[0].plot(g_meas.real[sub], g_meas.imag[sub], "--", lw=1,
           label="measured (calibrated)")
ax[0].plot(cal["gammas"].real, cal["gammas"].imag, "o", ms=5,
           label="reconstructed $\\gamma_k$")
ax[0].legend(fontsize=8); ax[0].axis("equal"); ax[0].grid(alpha=.3)
ax[0].set_title(f"complex plane, N={N_demo}, L=1.5$\\lambda$")

ax[1].plot(D, D_est, lw=1)
ax[1].plot(D, D, "k--", lw=.8)
ax[1].set_xlabel("true distance (mm)"); ax[1].set_ylabel("measured distance (mm)")
ax[1].set_title("distance transfer"); ax[1].grid(alpha=.3)

ax[2].plot(D, err, lw=.8)
ax[2].set_xlabel("true distance (mm)"); ax[2].set_ylabel("error (mm)")
ax[2].set_title(f"max={np.max(np.abs(err)):.2f} mm "
                f"({np.max(np.abs(err))/lam:.4f}$\\lambda$), "
                f"RMS={np.sqrt(np.mean(err**2)):.2f} mm")
ax[2].grid(alpha=.3)
fig.tight_layout()
fig.savefig("fig2_demo.png", dpi=150)
plt.close(fig)

print(f"[demo] N=8 L=1.5λ  calib time {t_demo:.2f}s  "
      f"max|err|={np.max(np.abs(err)):.3f} mm  "
      f"RMS={np.sqrt(np.mean(err**2)):.3f} mm  conj={cal['conjugated']}")
print("Abar_true vs A_cal:")
for a, b in zip(Abar_true, cal["A_cal"]):
    print(f"  {a: .4f}  {b: .4f}  |diff|={abs(a-b):.2e}")

# ============================================================ figure 3: N-L maps
Ns = np.arange(3, 21)                     # 3..20 positions (Sec. V)
Lfr = np.arange(0.25, 2.5001, 0.25)       # spread 0.25...2.5 lambda
maxmap = np.full((len(Ns), len(Lfr)), np.nan)
rmsmap = np.full((len(Ns), len(Lfr)), np.nan)

t0 = time.time()
for a_i, N in enumerate(Ns):
    for b_i, Lf in enumerate(Lfr):
        try:
            _, _, e, _ = run_calibration(int(N), Lf * lam)
            maxmap[a_i, b_i] = np.max(np.abs(e))
            rmsmap[a_i, b_i] = np.sqrt(np.mean(e ** 2))
        except Exception as ex:
            print(f"  N={N} L={Lf:.2f}λ failed: {ex}")
    print(f"N={N:2d} done, elapsed {time.time()-t0:.0f}s")

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
extent = [Lfr[0] - 0.125, Lfr[-1] + 0.125, Ns[0] - 0.5, Ns[-1] + 0.5]
for a, m, vmax, title in ((ax[0], maxmap, 14.0, "maximum error (mm)"),
                          (ax[1], rmsmap, 4.0, "RMS error (mm)")):
    im = a.imshow(m, origin="lower", aspect="auto", extent=extent,
                  vmin=0, vmax=vmax, cmap="jet")
    a.axvline(0.5, color="w", ls="--", lw=1.5)
    Lline = np.linspace(0.25, 2.5, 100)
    a.plot(Lline, 4 * Lline + 2, "w--", lw=1.5)
    a.text(0.55, 19, "$L=\\lambda/2$", color="w", fontsize=9)
    a.text(1.6, 7.2, "$N=4L/\\lambda+2$", color="w", fontsize=9, rotation=38)
    a.set_xlabel("spread of target positions $L$ ($\\lambda$)")
    a.set_ylabel("number of target positions $N$")
    a.set_title(title)
    plt.colorbar(im, ax=a, extend="max")
fig.suptitle("distance measurement error, 2.35 GHz, D = 100...420 mm (cf. paper Fig. 6)")
fig.tight_layout()
fig.savefig("fig3_error_maps.png", dpi=150)
plt.close(fig)

# good region statistics: L >= 0.75λ and N >= 4L/λ + 2
good = np.zeros_like(maxmap, dtype=bool)
for a_i, N in enumerate(Ns):
    for b_i, Lf in enumerate(Lfr):
        good[a_i, b_i] = (Lf >= 0.75) and (N >= 4 * Lf + 2)

summary = dict(
    lambda_mm=lam,
    sigma_dB=SIGMA_DB,
    demo=dict(N=N_demo, L_lambda=1.5,
              max_err_mm=float(np.max(np.abs(err))),
              max_err_lambda=float(np.max(np.abs(err)) / lam),
              rms_err_mm=float(np.sqrt(np.mean(err ** 2)))),
    good_region=dict(criterion="L>=0.75λ and N>=4L/λ+2",
                     n_cases=int(good.sum()),
                     worst_max_err_mm=float(np.nanmax(maxmap[good])),
                     worst_max_err_lambda=float(np.nanmax(maxmap[good]) / lam),
                     median_rms_err_mm=float(np.nanmedian(rmsmap[good]))),
    Abar_true=[[x.real, x.imag] for x in Abar_true],
    A_cal=[[x.real, x.imag] for x in cal["A_cal"]],
)
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary["good_region"], indent=2))
print("total time %.0f s" % (time.time() - t0))
