"""
Standalone example: generate an EMRI waveform (through the LISA response) for
the same scenario as Case1a (M=1e6, mu=50), for two different flux models -
FEW's built-in Schwarzschild-eccentric flux vs its built-in Kerr-eccentric-
equatorial flux (both GR, no scalar charge) - and plot their characteristic
strain against the LISA TDI PSD. This is the script that produced the
"plot_strain_SchwarzschildVSKerr_..." comparison in checks_Burke/.

p0 is obtained by backward-integrating the repo's own 1PA scalar-charge
trajectory (SchwarzCirc0PASpinScalar) from pfin to T, so the injected p0 is
consistent with the paper's scenario even though the two waveforms compared
here are plain GR (Schwarzschild/Kerr).
"""

import os
import sys

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits

import astropy.units as u
from astropy.cosmology import Planck18, z_at_value
cosmo = Planck18

from few.waveform import GenerateEMRIWaveform
from few.trajectory.ode import SchwarzEccFlux, KerrEccEqFlux
from few.trajectory.inspiral import EMRIInspiral

_repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared_modules"))
from Post_Adiabatic_Trajectories import SchwarzCirc0PASpinScalar
from PSD_code import write_psd_file, load_psd_from_file

run_direc = os.path.join(_repo_root, "psd_cache") + "/"
os.makedirs(run_direc, exist_ok=True)
out_dir = os.path.join(_repo_root, "raw_runs", "plot_waveform")
os.makedirs(out_dir, exist_ok=True)

YRSID_SI = 31558149.763545603
use_gpu = True
xp = cp
N_channels = 2

# Same physical scenario as Case1a (M=1e6, mu=50); p0 is recomputed below via
# backward integration rather than reusing Case1a's stored value.
M = 1e6; mu = 50; a = 0.0; e0 = 0.0; x_I0 = 1.0
SNR_choice = 200
qS = 0.8; phiS = 2.2; qK = 1.6; phiK = 1.2
Phi_phi0 = 2.5; Phi_theta0 = 0.0; Phi_r0 = 0.0
delta_t = 2.0; T = 1.0
pfin = 6.35
Lambda = 0.0025
Secondary_Spin = 0

tdi_gen = "2nd generation"
order = 25  # interpolation order

tdi_kwargs_esa = dict(
    orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
    order=order,
    tdi=tdi_gen,
    tdi_chan="AE",
)

index_lambda = 8
index_beta = 7
t0 = 20000.0  # throw away on both ends where the orbital information is unreliable


def get_redshift(distance):
    return z_at_value(cosmo.luminosity_distance, distance * u.Gpc).value


def zero_pad(data):
    N = len(data)
    pow_2 = xp.ceil(np.log2(N))
    return xp.pad(data, (0, int((2**pow_2) - N)), "constant")


def inner_prod(sig1_f, sig2_f, N_t, delta_t, PSD):
    prefac = 4 * delta_t / N_t
    sig2_f_conj = xp.conjugate(sig2_f)
    return prefac * xp.real(xp.sum((sig1_f * sig2_f_conj) / PSD))


# ===================== Compute p0 via backward integration =====================

traj_0PASpin = EMRIInspiral(func=SchwarzCirc0PASpinScalar)

_, p_traj_back, _, _, _, _, _ = traj_0PASpin(
    M, mu, a, pfin, e0, x_I0, Lambda, Secondary_Spin,
    Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T,
    integrate_backwards=True,
)
p0 = p_traj_back[-1]
print("p0 =", p0)

# ===================== Build the two (GR) waveform models =====================

inspiral_kwargs_schw = {"flux_output_convention": "pex", "func": SchwarzEccFlux}
inspiral_kwargs_kerr = {"flux_output_convention": "pex", "func": KerrEccEqFlux}

waveform_schwarzschild = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux",
    sum_kwargs=dict(pad_output=True),
    inspiral_kwargs=inspiral_kwargs_schw,
    use_gpu=use_gpu,
    return_list=False,
)

waveform_kerr = GenerateEMRIWaveform(
    "FastKerrEccentricEquatorialFlux",
    sum_kwargs=dict(pad_output=True),
    inspiral_kwargs=inspiral_kwargs_kerr,
    use_gpu=use_gpu,
    return_list=False,
)

EMRI_TDI_Schwarzschild = ResponseWrapper(
    waveform_schwarzschild, T, delta_t, index_lambda, index_beta, t0=t0,
    flip_hx=True, use_gpu=use_gpu, is_ecliptic_latitude=False,
    remove_garbage="zero", **tdi_kwargs_esa,
)

EMRI_TDI_Kerr = ResponseWrapper(
    waveform_kerr, T, delta_t, index_lambda, index_beta, t0=t0,
    flip_hx=True, use_gpu=use_gpu, is_ecliptic_latitude=False,
    remove_garbage="zero", **tdi_kwargs_esa,
)

# d_L = 1.0 for now; distance is rescaled below to match SNR_choice.
params_unnormed = [M, mu, a, p0, e0, x_I0, 1.0, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
add_params = [Lambda]

print("Generating the two waveforms...")
TDI_unnormed_schwarzschild = EMRI_TDI_Schwarzschild(*params_unnormed, *add_params, mode_selection_threshold=0.0)
TDI_unnormed_kerr = EMRI_TDI_Kerr(*params_unnormed, *add_params, mode_selection_threshold=0.0)

TDI_pad_unnormed_schwarzschild = [zero_pad(TDI_unnormed_schwarzschild[i]) for i in range(N_channels)]
TDI_pad_unnormed_kerr = [zero_pad(TDI_unnormed_kerr[i]) for i in range(N_channels)]

N_t = len(TDI_pad_unnormed_schwarzschild[0])

TDI_fft_unnormed_schwarzschild = xp.asarray([xp.fft.rfft(w) for w in TDI_pad_unnormed_schwarzschild])
TDI_fft_unnormed_kerr = xp.asarray([xp.fft.rfft(w) for w in TDI_pad_unnormed_kerr])

freq = xp.fft.rfftfreq(N_t, delta_t)
freq[0] = freq[1]  # retain the zeroth frequency
freq_np = xp.asnumpy(freq)

# ===================== PSD and distance normalization =====================

PSD_filename = "tdi2_AE_w_background.npy"
write_psd_file(model="scirdv1", channels="AE", tdi2=True, include_foreground=True,
               filename=run_direc + PSD_filename, stochastic_params=[T * YRSID_SI])
PSD_AE_interp = load_psd_from_file(run_direc + PSD_filename, xp=cp)
PSD_AE = PSD_AE_interp(freq_np)

SNR2_dL_1 = xp.asnumpy(xp.sum(xp.asarray([
    inner_prod(TDI_fft_unnormed_schwarzschild[i], TDI_fft_unnormed_schwarzschild[i], N_t, delta_t, PSD_AE[i])
    for i in range(N_channels)
])))
SNR_dL_1 = SNR2_dL_1 ** 0.5

dist = SNR_dL_1 / SNR_choice
check_redshift = get_redshift(dist)

TDI_fft_schwarzschild = (1 / dist) * TDI_fft_unnormed_schwarzschild
TDI_fft_kerr = (1 / dist) * TDI_fft_unnormed_kerr

SNR2 = xp.asnumpy(xp.sum(xp.asarray([
    inner_prod(TDI_fft_schwarzschild[i], TDI_fft_schwarzschild[i], N_t, delta_t, PSD_AE[i])
    for i in range(N_channels)
])))
SNR = SNR2 ** 0.5

print(f"Distance dist = {dist} Gpc, giving SNR = {SNR} (target was {SNR_choice})")
print(f"At this distance, redshift z = {check_redshift}")

# ===================== Plot the A channel =====================

plt.loglog(freq_np[1:], freq_np[1:] * abs(cp.asnumpy(TDI_fft_schwarzschild[0][1:])), label="Schwarzschild")
plt.loglog(freq_np[1:], freq_np[1:] * abs(cp.asnumpy(TDI_fft_kerr[0][1:])), label="Kerr")
plt.loglog(freq_np[1:], np.sqrt(freq_np[1:] * cp.asnumpy(PSD_AE[0][1:])), label="TDI2 PSD")
plt.xlabel(r"Frequency [Hz]", fontsize=20)
plt.ylabel(r"Characteristic strain", fontsize=20)
plt.title(fr"$(M, \mu, a, p_0, e_0, D_L)$ = ({M}, {mu}, {a}, {p0}, {e0}, {dist})")
plt.grid()
plt.xlim([1e-5, freq_np[-1]])
plt.legend(fontsize=8)

out_path = os.path.join(out_dir, f"plot_strain_SchwarzschildVSKerr_M{M}_mu{mu}_p0{p0}_e0{e0}_DL{dist}.png")
plt.savefig(out_path, bbox_inches="tight")
plt.close()
print("Saved:", out_path)
