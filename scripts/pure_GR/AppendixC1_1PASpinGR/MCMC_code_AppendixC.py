import os
import sys

import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt

from fastlisaresponse import ResponseWrapper     # Response

from lisatools.detector import EqualArmlengthOrbits

# Cosmology stuff
import astropy.units as u
from astropy.cosmology import Planck18, z_at_value; cosmo = Planck18

def get_redshift(distance):
    return (z_at_value(cosmo.luminosity_distance, distance * u.Gpc )).value

def get_distance(redshift):
    return cosmo.luminosity_distance(redshift).to(u.Gpc).value

# Import relevant EMRI packages
from few.waveform import GenerateEMRIWaveform
from few.trajectory.ode import PN5, SchwarzEccFlux, KerrEccEqFlux

from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_p_at_t
from few.utils.geodesic import get_separatrix

# Mode selection
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector
# Import features from eryn
from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.backends import HDFBackend

MAKE_PLOT = True
# PE_EMRI_params.py sits alongside this script (per-scenario values, injection).
from PE_EMRI_params import (M, mu, a, p0, e0, x_I0, SNR_choice, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T, pfin, delta_t, Lambda, Secondary_Spin)

# Post_Adiabatic_Trajectories.py and PSD_code.py are shared across all cases.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared_modules"))
# Appendix C (pure-GR spin reassessment) uses the same spin-scalar trajectories as Case III.
# Injection is always 1PASpinScalar with the scenario's true Lambda (always 0.0 for these
# scenarios - a real GR+spin system, no scalar charge at all) and true Secondary_Spin.
# Recovery always fixes Lambda to 0.0 (never sampled) - that is the "GR" in Appendix C - and
# may additionally fix Secondary_Spin to 0.0 (testing spin-neglect bias, not just PA-order bias).
from Post_Adiabatic_Trajectories import SchwarzCirc0PASpinScalar, SchwarzCirc1PASpinScalar
from PSD_code import (write_psd_file, load_psd_from_file, load_psd)

# =================== APPENDIX C: recovery model ===================
# RECOVERY_ORDER: "1PA" or "0PA" - which order's spin-scalar trajectory to recover with.
# FIX_SPIN_ZERO: True -> Secondary_Spin also fixed to 0.0 in recovery (pure Schwarzschild GR,
#                samples_..Inj{RECOVERY_ORDER}GRRec_...), matching the same convention Case II
#                uses for Lambda (same trajectory class family, hardcode the extra param to 0
#                rather than switching classes).
#                False -> Secondary_Spin free in recovery (samples_..Inj{RECOVERY_ORDER}SpinGRRec_...)
RECOVERY_ORDER = "1PA"    # <-- EDIT THIS per run: "1PA" or "0PA"
FIX_SPIN_ZERO = False      # <-- EDIT THIS per run: True or False

# DISTANCE_PRIOR_STYLE: "symmetric" (dist +/- n*Delta_theta_D, same as Case I/III) or
# "upper_only" (uniform_dist(0., dist + 25*Delta_theta_D) - specific to AppendixC3's
# DistanceUpperPrior25*deltaD runs; set per-scenario, not a general default).
DISTANCE_PRIOR_STYLE = "symmetric"   # <-- EDIT per scenario folder: "symmetric" or "upper_only"

recovery_label = f"{RECOVERY_ORDER}GR" if FIX_SPIN_ZERO else f"{RECOVERY_ORDER}SpinGR"

use_gpu = True

_repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
run_direc = os.path.join(_repo_root, "psd_cache") + "/"
os.makedirs(run_direc, exist_ok=True)

# Output folder (named after this script's own location, i.e. the case folder), used for the
# raw MCMC backend file, diagnostic plot, priors/config JSON, and failed-point log - instead
# of dumping everything flat into MCMC_scratch/ or run_direc (the historical mess). Each
# AppendixC1/C2/C3 folder is self-contained (its own PE_EMRI_params.py), same script.
script_dir = os.path.dirname(os.path.abspath(__file__))
case_name = os.path.basename(script_dir)
out_dir = os.path.join(_repo_root, 'raw_runs', case_name)
os.makedirs(out_dir, exist_ok=True)
YRSID_SI = 31558149.763545603

np.random.seed(1234)

tdi_gen = "2nd generation"

order = 25  # interpolation order (should not change the result too much)

tdi_kwargs_esa = dict(
    orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
    order=order,
    tdi=tdi_gen,
    tdi_chan="AE",
)  # could do "AET"

index_lambda = 8
index_beta = 7

# with longer signals we care less about this
t0 = 20000.0  # throw away on both ends when our orbital information is weird

TDI_channels = ['TDIA','TDIE']
N_channels = len(TDI_channels)


print("defining functions")
def zero_pad(data):
    '''
    Inputs: data stream of length N
    Returns: zero_padded data stream
    '''
    N = len(data)
    pow_2 = xp.ceil(np.log2(N))
    return xp.pad(data,(0,int((2**pow_2)-N)),'constant')

def inner_prod(sig1_f,sig2_f,N_t,delta_t,PSD):
    '''
    Compute stationary noise-weighted inner product
    Inputs: sig1_f and sig2_f are signals in frequency domain
            N_t length of padded signal in time domain
            delta_t sampling interval
            PSD Power spectral density

    Returns: Noise weighted inner product
    '''
    prefac = 4*delta_t / N_t
    sig2_f_conj = xp.conjugate(sig2_f)
    return prefac * xp.real(xp.sum((sig1_f * sig2_f_conj)/PSD))

##======================Likelihood and Posterior (change this)=====================

ERROR_LOG_FILE = os.path.join(out_dir, "failed_points.log")

def log_failed_point(params, error_msg):
    """Append a failed MCMC point to the log file."""
    with open(ERROR_LOG_FILE, "a") as f:
        f.write("ERROR: " + error_msg + "\n")
        f.write("PARAMS: " + ", ".join([f"{p:.8e}" for p in params]) + "\n")
        f.write("-" * 60 + "\n")

def llike(params):
    '''
    Inputs: Parameters to sample over
    Outputs: log-whittle likelihood
    Appendix C: Lambda is always fixed to 0.0 (never sampled - that is the "GR" in Appendix
    C). Secondary_Spin (params[9]) is free unless FIX_SPIN_ZERO, in which case it is also
    fixed to 0.0 and not sampled at all (ndim drops accordingly).
    '''
    try:
        # Intrinsic Parameters
        M_val = float(params[0])
        mu_val =  float(params[1])
        a_val =  a
        p0_val = float(params[2])
        e0_val = e0
        x_I0_val = x_I0

        # Luminosity distance
        D_val = float(params[3])

        # Angular Parameters
        qS_val = float(params[4])
        phiS_val = float(params[5])
        qK_val = float(params[6])
        phiK_val = float(params[7])

        # Angular parameters
        Phi_phi0_val = float(params[8])
        Phi_theta0_val = Phi_theta0
        Phi_r0_val = Phi_r0

        # Additional parameters - Lambda is always fixed to 0 (GR); Secondary_Spin is free
        # unless FIX_SPIN_ZERO, in which case it is also fixed to 0 (pure Schwarzschild GR).
        Lambda_val = 0.
        Secondary_Spin_val = 0. if FIX_SPIN_ZERO else float(params[9])
        try:
            waveform_prop = EMRI_TDI_Model_recovery(M_val, mu_val, a_val, p0_val, e0_val,
                                    x_I0_val, D_val, qS_val, phiS_val, qK_val, phiK_val,
                                        Phi_phi0_val, Phi_theta0_val, Phi_r0_val, Lambda_val, Secondary_Spin_val)
        except ValueError as e:
            # specifica per l'integratore che non converge
            print(f"[Integrator fail] {e} for params: {params}")
            log_failed_point(params, str(e))
            return -np.inf
        except Exception as e:
            print(f"[Waveform error] {e} for params: {params}")
            log_failed_point(params, str(e))
            return -np.inf


        # Taper and then zero pad.
        EMRI_AET_w_pad_prop = [zero_pad(waveform_prop[i]) for i in range(N_channels)]

        # Compute in frequency domain
        EMRI_AET_fft_prop = [xp.fft.rfft(item) for item in EMRI_AET_w_pad_prop]

        # Compute (d - h| d- h)
        diff_f_AET = [data_f_AET[k] - EMRI_AET_fft_prop[k] for k in range(N_channels)] #check is ZERO when d=0
        inn_prod = xp.asarray([inner_prod(diff_f_AET[k],diff_f_AET[k],N_t,delta_t,PSD_AE[k]) for k in range(N_channels)])

        # Return log-likelihood value as numpy val.
        llike_val_np = xp.asnumpy(-0.5 * (xp.sum(inn_prod)))
        return (llike_val_np)

    except Exception as e:
        # Stampa un warning opzionale per debugging
        print(f"[Warning] Likelihood failed for params {params}: {e}")
        # restituisci likelihood nulla (logL = -inf)
        return -np.inf

## =================== SET UP PARAMETERS =====================

N_channels = 2
xp = cp

## ===================== TRAJECTORIES (spin-scalar) ====================

traj_1PASpin = EMRIInspiral(func=SchwarzCirc1PASpinScalar)

# Forward check: verify the injected (1PA, true Lambda, true Secondary_Spin) trajectory
# reaches T years, using p0 as imported directly from PE_EMRI_params.py. Unlike Case I-IV,
# p0 is NOT recomputed via backward integration here: the appendix scenarios' p0 values were
# reconstructed from each historical run's true_values, so they are already the exact value
# used, and back-integration is unnecessary (and was not what the historical script did either
# - it had this step commented out and used a hardcoded numeric p0 instead).
t_traj, p_traj, e_traj, xI_traj, Phi_phi_traj, Phi_r_traj, Phi_theta_traj = traj_1PASpin(M, mu, a, p0, e0, x_I0, Lambda, Secondary_Spin,
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T)

traj_args = [M, mu, a, e_traj[0], xI_traj[0]]
print("p0 (from PE_EMRI_params.py) = ", p0)


print("Now going to load in class")
import cupy.cuda
device = cupy.cuda.Device(0)
print("Memoria libera PRIMA:", device.mem_info[0] / 1024**2, "MB")


inspiral_kwargs_0PA = {
    'flux_output_convention':'pex',
    'func':SchwarzCirc0PASpinScalar
    }

inspiral_kwargs_1PA = {
    'flux_output_convention':'pex',
    'func':SchwarzCirc1PASpinScalar
    }

Kerr_waveform_0PA = GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux",
        sum_kwargs=dict(pad_output=True),
        inspiral_kwargs = inspiral_kwargs_0PA,
        use_gpu=use_gpu,
        return_list=False,
    )

Kerr_waveform_1PA = GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux",
        sum_kwargs=dict(pad_output=True),
        inspiral_kwargs = inspiral_kwargs_1PA,
        use_gpu=use_gpu,
        return_list=False,
    )


# Build the response wrapper
print("Building the responses!")
EMRI_TDI_Model_1PA = ResponseWrapper(
        Kerr_waveform_1PA,
        T,
        delta_t,
        index_lambda,
        index_beta,
        t0=t0,
        flip_hx=True,  # set to True if waveform is h+ - ihx (FEW is)
        use_gpu=use_gpu,
        is_ecliptic_latitude=False,  # False if using polar angle (theta)
        remove_garbage="zero",  # removes the beginning of the signal that has bad information
        **tdi_kwargs_esa,
    )


EMRI_TDI_Model_0PA = ResponseWrapper(
        Kerr_waveform_0PA,
        T,
        delta_t,
        index_lambda,
        index_beta,
        t0=t0,
        flip_hx=True,  # set to True if waveform is h+ - ihx (FEW is)
        use_gpu=use_gpu,
        is_ecliptic_latitude=False,  # False if using polar angle (theta)
        remove_garbage="zero",  # removes the beginning of the signal that has bad information
        **tdi_kwargs_esa,
    )

# The recovery template used inside llike() - selected once via RECOVERY_ORDER at the top.
# (FIX_SPIN_ZERO only affects the Secondary_Spin_val passed into it, not which template is used.)
EMRI_TDI_Model_recovery = EMRI_TDI_Model_1PA if RECOVERY_ORDER == "1PA" else EMRI_TDI_Model_0PA

####=======================True Responsed waveform==========================

# Injection is always the 1PA model with the true Lambda (0.0) and true Secondary_Spin
# from PE_EMRI_params.py. We set d_L = 1.0 for now. We will choose luminosity distance
# according to a specific choice of SNR later.

params_unnormed = [M, mu, a, p0, e0, x_I0, 1.0, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
add_params = [Lambda, Secondary_Spin]

print("Memoria libera DOPO:", device.mem_info[0] / 1024**2, "MB")

import gc
gc.collect()
cp.get_default_memory_pool().free_all_blocks()

print("Memoria libera DOPO:", device.mem_info[0] / 1024**2, "MB")

print("Running the truth waveform")
Kerr_TDI_waveform_unnormed = EMRI_TDI_Model_1PA(*params_unnormed, *add_params, mode_selection_threshold = 0.0) # Inject waveform with all the modes!

# Taper and then zero_pad signal
Kerr_FEW_TDI_pad_unnormed = [zero_pad(Kerr_TDI_waveform_unnormed[i]) for i in range(N_channels)]

N_t = len(Kerr_FEW_TDI_pad_unnormed[0])


# Compute signal in frequency domain
Kerr_TDI_fft_unnormed = xp.asarray([xp.fft.rfft(waveform) for waveform in Kerr_FEW_TDI_pad_unnormed])

freq = xp.fft.rfftfreq(N_t,delta_t)
freq[0] = freq[1]   # To "retain" the zeroth frequency


# Define PSDs
# First, write PSD to a file.
PSD_filename = f"tdi2_AE_w_background_T{T}.npy"  # T-specific filename avoids race conditions between concurrently running scripts with different T
kwargs_PSD = {"stochastic_params": [T*YRSID_SI]} # We include the background

write_PSD = write_psd_file(model='scirdv1', channels='AE',
                           tdi2=True, include_foreground=True,
                           filename = run_direc + PSD_filename, **kwargs_PSD)

PSD_AE_interp = load_psd_from_file(run_direc + PSD_filename, xp=cp)

freq_np = xp.asnumpy(freq)

PSD_AE = PSD_AE_interp(freq_np)


SNR2_Kerr_FEW_dL_1 = xp.asnumpy(xp.sum(xp.asarray([inner_prod(Kerr_TDI_fft_unnormed[i],Kerr_TDI_fft_unnormed[i],N_t,delta_t,PSD_AE[i]) for i in range(N_channels)])))
SNR_Kerr_FEW_dL_1 = SNR2_Kerr_FEW_dL_1**(1/2)
print("SNR for Kerr_FEW with dL_1 ",SNR_Kerr_FEW_dL_1)

dist = (SNR_Kerr_FEW_dL_1/SNR_choice)

check_redshift = get_redshift(dist)

Kerr_TDI_fft = (1/dist) * Kerr_TDI_fft_unnormed

SNR2_Kerr_FEW = xp.asnumpy(xp.sum(xp.asarray([inner_prod(Kerr_TDI_fft[i],Kerr_TDI_fft[i],N_t,delta_t,PSD_AE[i]) for i in range(N_channels)])))
SNR_Kerr_FEW = SNR2_Kerr_FEW**(1/2)

print(f"New distance is dist = {dist}, giving SNR = {SNR_Kerr_FEW}")
print(f"At this value of distance, we find a redshift of z = {check_redshift}")

# ================== PLOT THE A CHANNEL ===================

if MAKE_PLOT == True:
    plt.loglog(freq_np[1:], freq_np[1:]*abs(cp.asnumpy(Kerr_TDI_fft[0][1:])), label = "Waveform frequency domain")
    plt.loglog(freq_np[1:], np.sqrt(freq_np[1:] * cp.asnumpy(PSD_AE[0][1:])), label = "TDI2 PSD")
    plt.xlabel(r'Frequency [Hz]', fontsize = 30)
    plt.ylabel(r'Magnitude',fontsize = 30)
    plt.title(fr'$(M, \mu, a, p_0, e_0, D_L)$ = {M,mu,p0,a, e0,dist}')
    plt.grid()
    plt.xlim([1e-5,freq_np[-1]])
    plt.savefig(os.path.join(out_dir, f"mcmc_charac_strain_{M,mu,p0,e0,dist}_Kerr.png"), bbox_inches = "tight")
    plt.close()


##=====================Noise Setting: Currently 0=====================

# Compute Variance and build noise realisation
variance_noise_AET = [N_t * PSD_AE[k] / (4*delta_t) for k in range(N_channels)]

variance_noise_AET[0] = 2*variance_noise_AET[0]
variance_noise_AET[-1] = 2*variance_noise_AET[-1]

noise_f_AET_real = [xp.random.normal(0,np.sqrt(variance_noise_AET[k])) for k in range(N_channels)]
noise_f_AET_imag = [xp.random.normal(0,np.sqrt(variance_noise_AET[k])) for k in range(N_channels)]

# Compute noise in frequency domain
noise_f_AET = xp.asarray([noise_f_AET_real[k] + 1j * noise_f_AET_imag[k] for k in range(N_channels)])

for i in range(N_channels):
    noise_f_AET[i][0] = noise_f_AET[i][0].real
    noise_f_AET[i][-1] = noise_f_AET[i][-1].real

data_f_AET = Kerr_TDI_fft + 0*noise_f_AET   # define the data

##===========================MCMC Settings============================

iterations = 30000  # The number of steps to run of each walker
burnin = 0 # I always set burnin when I analyse my samples
nwalkers = 50  #50 #members of the ensemble, like number of chains

ntemps = 1             # Number of temperatures used for parallel tempering scheme.

tempering_kwargs=dict(ntemps=ntemps)  # Sampler requires the number of temperatures as a dictionary

d = 0.1 # A parameter that can be used to dictate how close we want to start to the true parameters
# USEFUL CHECK: If d = 0 and noise_f = 0, llike(*params) = 0.0, exactly!!

# We start the sampler exceptionally close to the true parameters and let it run. This is reasonable
# if and only if we are quantifying how well we can measure parameters. We are not performing a search.

# Intrinsic Parameters

start_M = M*(1. + d * 1e-7 * np.random.randn(nwalkers,1))
start_mu = mu*(1. + d * 1e-7 * np.random.randn(nwalkers,1))

start_p0 = p0*(1. + d * 1e-8 * np.random.randn(nwalkers, 1))

# Luminosity distance
start_D = dist*(1 + d * 1e-6 * np.random.randn(nwalkers,1))

# Angular parameters
start_qS = qS*(1. + d * 1e-6 * np.random.randn(nwalkers,1))
start_phiS = phiS*(1. + d * 1e-6 * np.random.randn(nwalkers,1))
start_qK = qK*(1. + d * 1e-6 * np.random.randn(nwalkers,1))
start_phiK = phiK*(1. + d * 1e-6 * np.random.randn(nwalkers,1))

# Initial phases
start_Phi_Phi0 = Phi_phi0*(1. + d * 1e-6 * np.random.randn(nwalkers, 1))

start = np.hstack((start_M,start_mu, start_p0, start_D,
start_qS, start_phiS, start_qK, start_phiK, start_Phi_Phi0))

if not FIX_SPIN_ZERO:
    # Secondary_Spin is only sampled when not fixed to 0 (pure Schwarzschild GR variants
    # fix it instead). Lambda is never sampled at all in Appendix C.
    start_Secondary_Spin = Secondary_Spin*(1. + d * 1e-7 * np.random.randn(nwalkers,1))
    start = np.hstack((start, start_Secondary_Spin))


if ntemps > 1:
    # If we decide to use parallel tempering, we fall into this if statement. We assign each *group* of walkers
    # an associated temperature. We take the original starting values and "stack" them on top of each other.
    start = np.tile(start,(ntemps,1,1))

if np.size(start.shape) == 1:
    start = start.reshape(start.shape[-1], 1)
    ndim = 1
else:
    ndim = start.shape[-1]
print("ndim = ", ndim)

# ================= SET UP PRIORS ========================

n = 2500 # size of prior

Delta_theta_intrinsic = [M/(1e4), mu/(1e4), 1e-4, 1e-4, 1e-4, 1e-4]  # M, mu, a, p0, e0 Y0
Delta_theta_D = dist/np.sqrt(np.sum(SNR_Kerr_FEW))

priors_in = {
    # Intrinsic parameters
    0: uniform_dist(M - n*Delta_theta_intrinsic[0], M + n*Delta_theta_intrinsic[0]), # Primary Mass M
    1: uniform_dist(mu - n*Delta_theta_intrinsic[1], mu + n*Delta_theta_intrinsic[1]), # Secondary Mass mu
    2: uniform_dist(p0 - n*Delta_theta_intrinsic[3], p0 + n*Delta_theta_intrinsic[3]), # semi-latus rectum p0
    # Extrinsic parameters -- Angular parameters
    4: uniform_dist(0, np.pi), # Polar angle (sky position)
    5: uniform_dist(0, 2*np.pi), # Azimuthal angle (sky position)
    6: uniform_dist(0, np.pi),  # Polar angle (spin vec)
    7: uniform_dist(0, 2*np.pi), # Azimuthal angle (spin vec)
    # Initial phases
    8: uniform_dist(0, 2*np.pi), # Phi_phi0
}

if DISTANCE_PRIOR_STYLE == "upper_only":
    # AppendixC3-specific: one-sided upper-bounded prior (matches the historical
    # "DistanceUpperPrior25*deltaD" filename suffix).
    priors_in[3] = uniform_dist(0., dist + 25*Delta_theta_D) # distance D
else:
    priors_in[3] = uniform_dist(dist - n*Delta_theta_D, dist + n*Delta_theta_D) # distance D

if not FIX_SPIN_ZERO:
    # Secondary spin (Secondary_Spin) - free parameter unless FIX_SPIN_ZERO.
    # Lambda is never a free parameter in Appendix C (always fixed to 0 - that's the "GR").
    priors_in[9] = uniform_dist(-1., 1.)

priors = ProbDistContainer(priors_in, use_cupy = False)   # Set up priors so they can be used with the sampler.

# =================== SET UP PROPOSAL ==================

moves_stretch = StretchMove(a=2.0, use_gpu=True)

print("Memoria libera PRIMA DEL sampling:", device.mem_info[0] / 1024**2, "MB")
gc.collect()
cp.get_default_memory_pool().free_all_blocks()
print("Memoria libera DOPO:", device.mem_info[0] / 1024**2, "MB")

# Quick checks
if ntemps > 1:
    print("Value of starting log-likelihood points", llike(start[0][0]))
    if np.isinf(sum(priors.logpdf(np.asarray(start[0])))):
        print("You are outside the prior range, you fucked up")
        quit()
else:
    print("Value of starting log-likelihood points", llike(start[0]))

os.chdir(out_dir)  # out_dir was computed earlier, near run_direc

print("p0 = ",p0)
print("pfin = ",pfin)
print("M = ",M)
print("mu = ",mu)
print("SNR_choice = ",SNR_choice)
print("charge= ",np.sqrt(Lambda))
print("Lambda= ",Lambda)
print("Secondary_Spin= ",Secondary_Spin)
print("RECOVERY_ORDER= ", RECOVERY_ORDER)
print("FIX_SPIN_ZERO= ", FIX_SPIN_ZERO)
print("recovery_label= ", recovery_label)

# Save priors and run config as JSON, for reproducibility (e.g. computing Bayes factors
# later needs the exact prior bounds used).
import json

def serialize_prior(pr_obj):
    """Estrae gli estremi (low, high) da un uniform_dist."""
    return {"type": "uniform", "low": float(pr_obj.min_val), "high": float(pr_obj.max_val)}

priors_serialized = {str(k): serialize_prior(v) for k, v in priors_in.items()}
with open(os.path.join(out_dir, f"priors_{recovery_label}.json"), "w") as f:
    json.dump(priors_serialized, f, indent=4)

run_config = {
    "M": float(M), "mu": float(mu), "a": float(a), "p0": float(p0), "e0": float(e0),
    "x_I0": float(x_I0), "qS": float(qS), "phiS": float(phiS), "qK": float(qK),
    "phiK": float(phiK), "Phi_phi0": float(Phi_phi0), "Phi_theta0": float(Phi_theta0),
    "Phi_r0": float(Phi_r0), "delta_t": float(delta_t), "T": float(T), "pfin": float(pfin),
    "Lambda": float(Lambda), "Secondary_Spin": float(Secondary_Spin),
    "SNR_choice": float(SNR_choice), "dist": float(dist), "ntemps": ntemps,
    "nwalkers": nwalkers, "ndim": ndim, "iterations": iterations,
    "RECOVERY_ORDER": RECOVERY_ORDER, "FIX_SPIN_ZERO": FIX_SPIN_ZERO,
    "DISTANCE_PRIOR_STYLE": DISTANCE_PRIOR_STYLE,
}
with open(os.path.join(out_dir, f"run_config_{recovery_label}.json"), "w") as f:
    json.dump(run_config, f, indent=4)

fp = f"MCMC_samps_SPIN_1PASpinGRInj{recovery_label}Rec_M{M}_mu{mu}_p0{p0}_Lambda{Lambda}_SNR{SNR_choice}_dist{dist}_T{T}_pfin{pfin}_dt{delta_t}_dstart{d}_ntemps{ntemps}_Kerr_SecondarySpin{Secondary_Spin}.h5"
backend = HDFBackend(fp)

ensemble = EnsembleSampler(
                            nwalkers,
                            ndim,
                            llike,
                            priors,
                            backend = backend,                 # Store samples to a .h5 file
                            tempering_kwargs=tempering_kwargs,  # Allow tempering!
                            moves = moves_stretch
                            )
Reset_Backend = True # NOTE: CAREFUL HERE. ONLY TO USE IF WE RESTART RUNS!!!!
if Reset_Backend:
    os.remove(fp) # Manually get rid of backend
    backend = HDFBackend(fp) # Set up new backend
    ensemble = EnsembleSampler(
                            nwalkers,
                            ndim,
                            llike,
                            priors,
                            backend = backend,                 # Store samples to a .h5 file
                            tempering_kwargs=tempering_kwargs,  # Allow tempering!
                            moves = moves_stretch
                            )
else:
    start = backend.get_last_sample() # Start from last sample
out = ensemble.run_mcmc(start, iterations, progress=True)  # Run the sampler
