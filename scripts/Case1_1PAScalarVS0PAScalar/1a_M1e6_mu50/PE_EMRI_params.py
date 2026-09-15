###
# This is the main file that should be edited.
###
try:
    import cupy as cp
    import numpy as np
    xp = cp
    use_gpu = True
except ImportError:
    import numpy as np
    xp = np
    use_gpu = False

# Reconstructed for reproducibility: exact parameters used for this specific
# paper result, read back from the true_values embedded in the corresponding
# trimmed .h5 file in GOOD_RESULTS/data/. PE_EMRI_params.py is normally edited
# in place before each run, so historical per-run values are otherwise lost -
# see GOOD_RESULTS/scripts/README.md.

M = 1e6; mu = 50; a = 0.0; p0 = 12.568936166598405; e0 = 0.0; x_I0 = 1.0
SNR_choice = 200;
qS = 0.8; phiS = 2.2; qK = 1.6; phiK = 1.2;
Phi_phi0 = 2.5; Phi_theta0 = 0.0; Phi_r0 = 0.0;
delta_t = 2.0; T = 1.0;
pfin = 6.35
Lambda = 0.0025
Secondary_Spin = 0
