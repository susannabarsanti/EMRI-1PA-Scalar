"""
Standalone, CPU-only example: integrate the 1PA scalar-charge trajectory
(SchwarzCirc1PAScalar) with the true Lambda vs. the same class with Lambda=0
(i.e. the 1PA GR trajectory), and report the dephasing in Phi_phi induced by
the scalar charge at 1PA order. Only trajectory integration is needed here -
no waveform/response/PSD/GPU machinery - so this runs on CPU.
"""

import os
import sys

import numpy as np
from scipy.interpolate import interp1d

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared_modules"))
from Post_Adiabatic_Trajectories import SchwarzCirc0PAScalarS, SchwarzCirc1PAScalar

from few.trajectory.inspiral import EMRIInspiral

# Example scenario (same as Case2_1PAScalarVSGR).
M = 1e5; mu = 10; a = 0.0; e0 = 0.0; x_I0 = 1.0
Phi_phi0 = 2.5; Phi_theta0 = 0.0; Phi_r0 = 0.0
T = 1.0
pfin = 6.35
Lambda = 0.000625

traj_0PAScalar = EMRIInspiral(func=SchwarzCirc0PAScalarS)
traj_1PAScalar = EMRIInspiral(func=SchwarzCirc1PAScalar)

# p0 via backward integration from pfin at 0PA order.
_, p_traj_back, _, _, _, _, _ = traj_0PAScalar(
    M, mu, a, pfin, e0, x_I0, Lambda,
    Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T,
    integrate_backwards=True,
)
p0 = p_traj_back[-1]
print("p0 =", p0)

t_scalar, p_scalar, _, _, Phi_phi_scalar, _, _ = traj_1PAScalar(
    M, mu, a, p0, e0, x_I0, Lambda,
    Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T,
)
print("p_fin (1PA, Lambda) =", p_scalar[-1])
print("Phi_phi_fin (1PA, Lambda) =", Phi_phi_scalar[-1])

t_gr, p_gr, _, _, Phi_phi_gr, _, _ = traj_1PAScalar(
    M, mu, a, p0, e0, x_I0, 0.0,
    Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T,
)
print("p_fin (1PA, GR) =", p_gr[-1])
print("Phi_phi_fin (1PA, GR) =", Phi_phi_gr[-1])

interp_phi_scalar = interp1d(t_scalar, Phi_phi_scalar, kind="linear")
interp_phi_gr = interp1d(t_gr, Phi_phi_gr, kind="linear")

t_max = min(t_scalar[-1], t_gr[-1])
dephasing = interp_phi_scalar(t_max) - interp_phi_gr(t_max)

print("t_max =", t_max)
print("DEPHASING (scalar charge, 1PA order) =", dephasing)
