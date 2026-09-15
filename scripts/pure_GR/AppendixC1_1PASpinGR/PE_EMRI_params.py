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
# trimmed .h5 file in ../../data/ (or ../../../data/ for Case1/AppendixC
# scenario subfolders). This file is normally edited in place before each run,
# so it only ever reflects the parameters used to produce the .h5 file shipped
# alongside this script.

M = 1e6; mu = 100; a = 0.0; p0 = 15.7905; e0 = 0.0; x_I0 = 1.0
SNR_choice = 80;
qS = np.pi/3; phiS = np.pi/4; qK = 2.0; phiK = 5.0;
Phi_phi0 = 1.5; Phi_theta0 = 0.0; Phi_r0 = 0.0;
delta_t = 2.0; T = 1.5;
pfin = 6.2
Lambda = 0.0
Secondary_Spin = 0.5
