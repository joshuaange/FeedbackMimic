"""
Extending the baryonic feedback profile to higher redshifts from z=0
"""

import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from feedback_mimic import get_some_quantities, get_NFW_profile, get_realspace_profile, get_spectra, get_enclosed_mass, uniform_density_ft, Mtot_to_MDelta, get_rho_cga

PROF_INPUT_FILE   = "/home/jange/scratch/FeedbackMimic/results/profile_matching"
HALO_INPUT_FILE   = "/home/jange/scratch/FeedbackMimic/results/halo_model"
OUTPUT_DIR   = "/home/jange/scratch/FeedbackMimic/results/feedback_models"
 
Z_ARR      = np.linspace(0.0, 2.0, 81)
Z_OF_PROFILE_MATCHING = [0.0,0.5,1.0,1.5,2.0]
GAMMA_SCALINGS  = [0.0]
K_ARR  = np.logspace(-6, 6, 8192)
R_ARR  = np.logspace(-4, np.log10(1 / K_ARR.min()), 1500)

etaz = 0.7
FEEDBACK_MODEL = f"etaz{etaz}_preferred_cooled"

FEEDBACK_MODEL_TO_MATCH = FEEDBACK_MODEL
NUM_ITERATIONS_TO_MATCH = 25
M_transition = 2.5e11

# CGA parameters (must match match_profile.py)
A_star  = 0.09
M_0star = 2.5e11
eta_cga = 0.6

##############################################

print(FEEDBACK_MODEL)
for gamma_scaling in GAMMA_SCALINGS:
    print(f"\n{'='*60}\ngamma_scaling = {gamma_scaling}")
    for z in tqdm(Z_ARR):
        with open(HALO_INPUT_FILE + f"/{z}_data_LCDM_full2h_preferred.pkl", "rb") as f:
            data_LCDM = pickle.load(f)
        r_v, rho_crit, rho_char, concentration, r_s, Delta_vir, rho_0, rho_0_wh = get_some_quantities(data_LCDM)
        M_arr = data_LCDM['M']
        k_arr_HMCode = data_LCDM['power_spec_dic']['k']

        # ── CGA profile for this z ──────────────────────────────────────
        M200c_arr, r200c_arr = Mtot_to_MDelta(
            data_LCDM, r_v, M_arr, K_ARR, R_ARR, Del=200,
            reference_density="critical"
        )
        f_cga = A_star * (M_0star / M200c_arr)**eta_cga
        R_h   = 0.015 * r200c_arr

        rho_cga = np.zeros((np.size(M_arr), len(R_ARR)))
        for j in range(np.size(M_arr)):
            if M_arr[j] >= M_transition:
                rho_cga[j] = get_rho_cga(R_ARR, M_arr[j], M200c_arr[j], r200c_arr[j])
        rho_cga[R_ARR[None, :] > r_v[:, None]] = 0.0

        u_cga            = f_cga[:, None] * np.exp(-(K_ARR[None, :]       * R_h[:, None])**2)
        u_cga_HMCodek    = f_cga[:, None] * np.exp(-(k_arr_HMCode[None,:] * R_h[:, None])**2)

        # logic to pick Z_0 and Z_1, and skip if needed
        Z_0 = max((x for x in Z_OF_PROFILE_MATCHING if x <= z), default=None)
        Z_1 = min((x for x in Z_OF_PROFILE_MATCHING if x >= z), default=None)
        print(f"{Z_0} <= {z} <= {Z_1}")

        with open(HALO_INPUT_FILE + f"/{Z_0}_data_LCDM_full2h_preferred.pkl", "rb") as f:
            data_LCDM_z0 = pickle.load(f)
        r_v_z0, *_ = get_some_quantities(data_LCDM_z0)
        with open(HALO_INPUT_FILE + f"/{Z_1}_data_LCDM_full2h_preferred.pkl", "rb") as f:
            data_LCDM_z1 = pickle.load(f)
        r_v_z1, *_ = get_some_quantities(data_LCDM_z1)

        starting_profile_z0 = np.load(PROF_INPUT_FILE + f"/{FEEDBACK_MODEL_TO_MATCH}/utilde_results_gamma{gamma_scaling}_{NUM_ITERATIONS_TO_MATCH}it_z{Z_0}.npy", allow_pickle=True)[()]
        starting_profile_z1 = np.load(PROF_INPUT_FILE + f"/{FEEDBACK_MODEL_TO_MATCH}/utilde_results_gamma{gamma_scaling}_{NUM_ITERATIONS_TO_MATCH}it_z{Z_1}.npy", allow_pickle=True)[()]

        if Z_0 == Z_1:
            Delta        = starting_profile_z0[-1]['Delta']
            r_t_unbiased = starting_profile_z0[-1]['r_t_unbiased'] * r_v / r_v_z0
        else:
            t = (z - Z_0) / (Z_1 - Z_0)
            Delta        = starting_profile_z0[-1]['Delta'] + t * (starting_profile_z1[-1]['Delta'] - starting_profile_z0[-1]['Delta'])
            r_t_unbiased = (starting_profile_z0[-1]['r_t_unbiased'] * r_v / r_v_z0) + t * ((starting_profile_z1[-1]['r_t_unbiased'] * r_v / r_v_z1) - (starting_profile_z0[-1]['r_t_unbiased'] * r_v / r_v_z0))

        # ── Reconstruct NFW + constant components ───────────────────────
        u_k_unbiased_to_rt  = get_NFW_profile(r_t=r_t_unbiased, r_v=r_v, k_arr=K_ARR, r_s=r_s) * Delta
        u_k_unbiased_to_max = get_NFW_profile(r_t=max(R_ARR)*np.ones_like(r_v), r_v=r_v, k_arr=K_ARR, r_s=r_s) * Delta
        rho_unbiased_to_rt  = get_realspace_profile(u_k=u_k_unbiased_to_max,
                                                    r_arr=R_ARR, k_arr=K_ARR, M_arr=M_arr, r_t=r_t_unbiased)

        mass_contained                   = get_enclosed_mass(rho=rho_unbiased_to_rt, r_t=r_t_unbiased, r_arr=R_ARR)
        mass_unbiased_to_add_from_constant = M_arr - mass_contained
        u_k_constant_unbiased            = uniform_density_ft(mass_unbiased_to_add_from_constant, M_arr, r_t_unbiased, K_ARR)

        # ── Combine: CGA + (1 - f_cga) * (NFW + constant) ──────────────
        # matches: u_k_hat_unbiased = u_cga + (1-f_cga) * (u_k_unbiased_to_rt + u_k_constant_unbiased)
        u_k_hat_unbiased = u_cga + (1 - f_cga[:, None]) * (u_k_unbiased_to_rt + u_k_constant_unbiased)
        u_k_hat_unbiased[M_arr < M_transition] = get_NFW_profile(
            r_t=r_v[M_arr < M_transition], r_v=r_v[M_arr < M_transition],
            k_arr=K_ARR, r_s=r_s[M_arr < M_transition]
        )  # low-mass bins revert to NFW, matching match_profile behaviour

        u_k_hat_unbiased_max = u_cga + (1 - f_cga[:, None]) * (u_k_unbiased_to_max + u_k_constant_unbiased)
        rho_hat_unbiased = rho_cga + (1 - f_cga[:, None]) * get_realspace_profile(
            u_k=(u_k_unbiased_to_max + u_k_constant_unbiased), r_arr=R_ARR, k_arr=K_ARR, M_arr=M_arr, r_t=r_t_unbiased)

        # ── Power spectrum ───────────────────────────────────────────────
        u_k_hat_unbiased_HMCode = np.array([interp1d(K_ARR, row, kind='linear')(k_arr_HMCode) for row in u_k_hat_unbiased])
        P_tot, P_1h, P_2h, hmf, M, u_k_axionHMCode, _ = get_spectra(
            data_LCDM, utilde=u_k_hat_unbiased_HMCode, full_output=True)

        iteration_results = {
            "r_t_unbiased": r_t_unbiased,
            "u_k_hat_unbiased": u_k_hat_unbiased,
            "rho_hat_unbiased": rho_hat_unbiased,
            "P": P_tot,
            "P_1h": P_1h,
            "P_2h": P_2h,
            "hmf": hmf,
            "u_k_axionHMCode": u_k_axionHMCode,
        }

        np.save(OUTPUT_DIR + f"/{FEEDBACK_MODEL}/utilde_results_gamma{gamma_scaling}_z{z}.npy", [iteration_results])