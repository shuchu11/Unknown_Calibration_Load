"""Noiseless sanity check of the 2018 calibration algorithm:
generate powers from known Abar_i / gamma_k, calibrate treating the loads as
unknown, and verify Abar_i and gamma_k are recovered."""
import numpy as np
from sixport_calibration import calibrate, measure_gamma

rng = np.random.default_rng(7)

# True system constants (Abar_i = A_i * Gamma_1) and q_i
A_true = np.array([0.95 * np.exp(1j * np.deg2rad(35)),
                   0.70 * np.exp(1j * np.deg2rad(160)),
                   1.10 * np.exp(1j * np.deg2rad(-95))])
q_true = np.array([0.20, 0.25, 0.18])

# K unknown calibration loads, normalized so gamma_1 = 1, phases roughly
# DECREASING (like increasing transmission-line lengths / target distance)
K = 8
mags = rng.uniform(0.5, 1.3, K); mags[0] = 1.0
ph = np.concatenate([[0.0], -np.cumsum(rng.uniform(25, 70, K - 1))])
gam_true = mags * np.exp(1j * np.deg2rad(ph))

P = q_true[:, None] * np.abs(1 + A_true[:, None] * gam_true[None, :]) ** 2
p0 = q_true.copy()  # matched load: p_{i,0} = q_i, eq. (3)

cal = calibrate(p0, P, expected_slope="negative")

print("conjugated:", cal["conjugated"], " set:", cal["set_used"],
      " residual: %.3e" % cal["residual"])
print("d12=%.6f d13=%.6f" % (cal["d12"], cal["d13"]))
print("\nA_true  vs  A_cal")
for a, b in zip(A_true, cal["A_cal"]):
    print(f"  {a: .6f}   {b: .6f}   |diff|={abs(a-b):.2e}")
print("\nmax |gamma error| on calibration loads:",
      np.max(np.abs(cal["gammas"] - gam_true)))

# measure fresh random loads
gt = rng.uniform(0.2, 1.2, 5) * np.exp(1j * rng.uniform(-np.pi, np.pi, 5))
Pm = q_true[:, None] * np.abs(1 + A_true[:, None] * gt[None, :]) ** 2
gm = measure_gamma(cal, Pm)
print("max |gamma error| on 5 fresh test loads:", np.max(np.abs(gm - gt)))
