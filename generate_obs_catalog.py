"""
generate_mock_catalogs.py
=========================
Original mock catalog generator, unchanged except for the addition of a
'wl_data' sub-dict in the catalog output.  The wl_data block is built
entirely from quantities already computed during generation (no new
computations, no unit changes) and is consumed by cluster_catalogue_survey
for the new cosmocnc-internal shear likelihood path.

Everything used by get_log_lik_externalShear is untouched:
  - Sigma_crit_arr convention:  compute_Sigma_crit_arr(...) / h_fid
  - get_tangential_shear call:  identical
  - gt_list, R_proj_slices:     identical
  - catalog top-level keys:     identical
"""

import os
import pickle
import numpy as np
from scipy import integrate, interpolate
from tqdm import tqdm
from astropy.cosmology import FlatLambdaCDM
import cosmocnc_axionHMcode_jax.cosmocnc_jax as cosmocnc
from axionHMcode.cosmology.variance import func_sigma_M
from axionHMcode.cosmology.overdensities import func_delta_c
from axionHMcode.halo_model.halo_mass_function import func_sheth_tormen

from feedback_mimic import (
    dn_dlog10Mtot, get_NFW_profile, get_realspace_profile,
    dV_dz, profile_to_q, Mtot_to_MDelta, get_some_quantities,
    get_FDM_profile, get_total_mass, get_tangential_shear, dV_dz
)

# ── paths ─────────────────────────────────────────────────────────────────────
HALO_INPUT_FILE     = "/home/jange/scratch/FeedbackMimic/results/halo_model"
FEEDBACK_INPUT_FILE = "/home/jange/scratch/FeedbackMimic/results/feedback_models"
OUTPUT_DIR          = "/home/jange/scratch/FeedbackMimic/results/mock_catalogs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── configuration (unchanged) ─────────────────────────────────────────────────
HMF_TYPES = ['FDM_axfraction05']
FDM_suffix = "_axfraction05" # must manually change this

WL_TYPE = "DES"

#['LCDM', 'LCDM_COSMOCNC_SR',  'LCDM_COSMOCNC', 
# 'LCDM_hat0.0', 'FDM_preferred']
# 'LCDM_hat0.0_axfraction05', 'FDM_axfraction05']

pressure_prescription = "ArnaudUPP"
# allows ArnaudUPP, AricoBCM, baryonPasting
# default is ArnaudUPP, others add a qualifier to the end of the catalog
fth_prescription = ""
# nelson2014 or green2020 -> only when not using ArnaudUPP
if pressure_prescription == "ArnaudUPP":
    PRESSURE_SUFFIX = ""
elif pressure_prescription == "AricoBCM":
    PRESSURE_SUFFIX = f"_AricoBCMPressure{fth_prescription}"
elif pressure_prescription == "baryonPasting":
    PRESSURE_SUFFIX = f"_baryonPastingPressure{fth_prescription}"
else:
    print("No valid pressure profile prescription given, defaulting to...")
    PRESSURE_SUFFIX = ""
print(f"Pressure Prescription {PRESSURE_SUFFIX}")

SR_params = "bestfitSR"
# PlanckSZiFi or bestfitSR

INCLUDE_SHEAR_PROFILES = True
etaz           = 0.7
FEEDBACK_MODEL = f"etaz{etaz}{FDM_suffix}_cooled"

N_MOCKS_PER_TYPE = 1
#SEEDS            = [51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75]
#SEEDS            = [50]
SEEDS = [70,71,72,73,74,75]

Q_TH  = 5.0
Q_MIN = 0.4 + 1e-6        # this just makes it be a smooth taper on the total mass of halos selected
Q_FLOOR = 0.4
FSKY  = 0.4
h_fid = 0.68

M_MIN_SAMPLE = 10**12.5
M_MAX_SAMPLE = 1e16
NUM_M_SAMPLE = 300
ln_M_target  = np.linspace(np.log(M_MIN_SAMPLE), np.log(M_MAX_SAMPLE), NUM_M_SAMPLE)
M_target     = np.exp(ln_M_target)

SIGMA_LN_Q_INTRINSIC = 0.173
SIGMA_Q_OBS          = 1.0

if WL_TYPE == "DES":
    SIGMA_SHAPE = 0.37
    N_SOURCE    = 5.5
    # z_min ~ 0.025, r_max = 3.2/1.025 ~ 3.12, n_bins = round((3.12-0.5)/0.3)+1 = 10
    N_R_MAX = 10   # matches external likelihood's max
    DELTA_R_BOCQUET = 0.3
    _z_source_bins=np.array([0.285, 0.476, 0.743, 0.942])
    wl_type_suffix = "_DES"
elif WL_TYPE == "Euclid":
    SIGMA_SHAPE = 0.26
    N_SOURCE = 30.0
    # z_min ~ 0.025, r_max = 3.2/1.025 ~ 3.12, n_bins = round((3.12-0.5)/0.3)+1 = 10
    N_R_MAX = 10   # matches external likelihood's max
    DELTA_R_BOCQUET = 0.3
    _z_source_bins=np.array([0.285, 0.476, 0.743, 0.942])
    wl_type_suffix = "_Euclid"
elif WL_TYPE == "deepEuclid":
    SIGMA_SHAPE = 0.26
    N_SOURCE = 30.0
    # z_min ~ 0.025, r_max = 3.2/1.025 ~ 3.12, n_bins = round((3.12-0.5)/0.3)+1 = 10
    N_R_MAX = 10   # matches external likelihood's max
    DELTA_R_BOCQUET = 0.3
    _z_source_bins = np.array([0.3195, 0.4941, 0.6206, 0.7338, 0.8440,
                                    0.9581, 1.0841, 1.2336, 1.4331, 1.7875])
    wl_type_suffix = "_deepEuclid"
print(f"WL Type {WL_TYPE}")

M_MIN        = 1e11
M_MAX        = 1e16
NUM_M        = 300
M_DELTA      = 200
M_REF_DENS   = "critical"
M_TRANSITION = 10**12.5

Z_ARR = np.linspace(0.0, 2.0, 81)
Z_ARR = Z_ARR[Z_ARR > 0]

K_ARR = np.logspace(-6, 6, 8192)
R_ARR = np.logspace(-4, np.log10(1 / K_ARR.min()), 1500)

H_FID = 0.68
#####################
if 'LCDM_COSMOCNC' in HMF_TYPES or 'LCDM_COSMOCNC_SR' in HMF_TYPES:
    import os
    os.environ["TOMOGRAPHIC_BINS"] = WL_TYPE
    from mcmc_likelihood import get_log_likelihood
    ln_L, nc = get_log_likelihood(
        catalogue_name = f"/home/jange/scratch/FeedbackMimic/results/mock_catalogs/LCDM{wl_type_suffix}_mock_seed50.pkl",
        omega_m  = 0.143344,
        A_s      = 2.1e-9,
        A_sz     = -4.352890598769685,
        alpha_sz = 1.1991734303623067,
        beta_sz  = 1.9996943667059135,
        b_WL_0   = 0.0,
        b_WL_M   = 1.0,
        return_nc = True,
        n_points = 1000,
        n_mass_points_coarse = 1000
    )
if 'LCDM_COSMOCNC_SR' in HMF_TYPES:
    from cosmocnc_axionHMcode_jax.cosmocnc_jax.surveys.survey_sr_so_sim import (
        precompute_q_prefactors, sr_q_so_sim_layer0, sr_q_so_sim_layer1
    )
    
    def q_via_survey_sr(M500c_h, z, nc, A_sz, alpha_sz, beta_sz, bias_sz, dof, h_fid=H_FID):
        """
        Computes q per-halo using the EXACT algorithm survey_sr_so_sim.py's
        sr_q_so_sim_layer0/layer1 use in the real likelihood, in place of
        feedback_mimic's profile_to_q integral method. Deliberately uses nc's
        OWN D_A/E_z (the same emulator-derived values the real likelihood
        consumes) rather than an independent reference -- this catalog is meant
        to inherit whatever the likelihood's own machinery produces, so any
        comparison against the LCDM_COSMOCNC catalog isolates exactly the
        profile-integral-vs-analytic-formula question, nothing else.
        """
        iz = np.argmin(np.abs(np.asarray(nc.redshift_vec) - z))
        E_z_val, D_A_val = float(nc.E_z[iz]), float(nc.D_A[iz])
        H0_val = h_fid * 100.
    
        prefactor_logy0, prefactor_M_500_to_theta = precompute_q_prefactors(
            E_z_val, H0_val, D_A_val, A_sz, bias_sz, alpha_sz, beta_sz)
    
        M500c_phys = np.asarray(M500c_h) / h_fid
        x0_500 = np.log(M500c_phys / 1e14)
        sr = nc.scaling_relations['q_so_sim']
        sigma_sz_poly = sr.sigma_sz_poly
        log_theta_min = sr.log_theta_min          # same bounds object the
        log_theta_max = sr.log_theta_max          # likelihood itself uses

        x1, _ = sr_q_so_sim_layer0(x0_500, prefactor_logy0, prefactor_M_500_to_theta,
                                     sigma_sz_poly, alpha_sz,
                                     log_theta_min, log_theta_max)
        q = sr_q_so_sim_layer1(x1, prefactor_logy0, prefactor_M_500_to_theta, dof)
        return np.asarray(q)
#####################
from cosmocnc_axionHMcode_jax.cosmocnc_jax.mass_conversion import _nfw_m, _nfw_m_prime
def nfw_m500c_conversions(c500c, n_iter=15):
    """Given c500c, returns (M200c/M500c, c200c) from the same Newton solve."""
    m_c500 = _nfw_m(c500c)
    def step(_, y):
        cy = c500c * y
        mc = _nfw_m(cy)
        F  = 2.5 * y**(-3) - m_c500 / mc
        Fp = -7.5 * y**(-4) + (m_c500 * c500c * _nfw_m_prime(cy)) / (mc**2)
        return y - F / Fp
    y0 = np.full_like(c500c, 1.5)
    y = y0
    for i in range(n_iter):
        y = step(i, y)
    M200_over_M500 = (y**3) / 2.5
    c200c = c500c * y
    return M200_over_M500, c200c
    
def build_cosmocnc_hmf_M200c(nc, z_request, ln_M200c_target_Msol_h, data):
    ln_M_nc = np.asarray(nc.ln_M)
    M_nc_phys   = np.exp(ln_M_nc) * 1e14
    M_nc_Msol_h = np.exp(ln_M_nc) * 1e14 * H_FID
    ln_M_nc_Msol_h = np.log(M_nc_Msol_h)
    redshift_vec_nc = np.asarray(nc.redshift_vec)
    iz_nc = np.argmin(np.abs(redshift_vec_nc - z_request))

    hmf_matrix_nc   = np.asarray(nc.hmf_matrix).copy()[iz_nc]
    hmf_matrix_nc  /= nc.cosmology.background_cosmology.differential_comoving_volume(z_request).value # dn/dlnM500c in physical units
    hmf_matrix_nc  /= M_nc_phys # dn/dM500c
    # divide by h^4 to get to h-units -> dn/dM200c in h units
    M200c = nfw_m500c_conversions(nc.profile_params['concentration_500c'])[0][iz_nc] * M_nc_phys # M200c, physical Msol
    dM500_dM200 = np.gradient(M_nc_phys) / np.gradient(M200c)
    hmf_matrix_nc  *= dM500_dM200 # dn/dlnM200c in physical units
    #hmf_matrix_nc  /= H_FID**4 # dn/dM200c in h units
    #hmf_matrix_nc  *= M200c*H_FID # dn/dlnM200c in h units
    hmf_matrix_nc  *= M200c
    hmf_on_target_grid = np.interp(
        ln_M200c_target_Msol_h, np.log(M200c * H_FID), hmf_matrix_nc,
        left=0., right=0.)
    return hmf_on_target_grid



# ── CDF-inversion sampler (unchanged) ─────────────────────────────────────────
'''
def _get_samples_pdf_1d(n_samples, x, cdf):
    cdf = cdf / np.max(cdf)
    u   = np.random.rand(n_samples)
    return np.interp(u, cdf, x)

def get_samples_pdf_2d(n_samples, x_arr, y_arr, pdf_xy):
    eps = 1e-12
    pdf_x  = np.sum(pdf_xy, axis=1) * (y_arr[1] - y_arr[0])
    #cdf_x  = np.cumsum(pdf_x) * (x_arr[1] - x_arr[0])
    cdf_x = scipy.integrate.cumulative_simpson(pdf_x, x=x_arr, initial=0.0)
    x_samp = _get_samples_pdf_1d(n_samples, x_arr, cdf_x)

    cdf_y_given_x = np.cumsum(pdf_xy, axis=1) * (y_arr[1] - y_arr[0])
    for ix in range(cdf_y_given_x.shape[0]):
        mx = cdf_y_given_x[ix, -1]
        if mx > eps:
            cdf_y_given_x[ix] /= mx

    interp_cdf = interpolate.RegularGridInterpolator(
        (x_arr, y_arr), cdf_y_given_x,
        method='linear', bounds_error=False, fill_value=None)

    u      = np.random.rand(n_samples)
    y_samp = np.zeros(n_samples)
    for i, (xi, ui) in enumerate(zip(x_samp, u)):
        cdf_i     = interp_cdf(np.column_stack([np.full_like(y_arr, xi), y_arr]))
        y_samp[i] = np.interp(ui, np.clip(cdf_i, 0, 1), y_arr)

    return x_samp, y_samp
'''
from scipy import integrate
from scipy.interpolate import PchipInterpolator

def _get_samples_pdf_1d(n_samples, x, cdf):
    """
    Inverse-transform sampling via monotone cubic (PCHIP) interpolation of
    a Simpson-integrated CDF, replacing linear interpolation of a
    cumsum-based CDF. Fixes a several-percent, statistically significant
    bias (excess near rising edges/peaks, deficit through curved tails)
    that the original method introduced at finite grid resolution.

    NOTE: signature now takes the PDF values (not a precomputed cdf) and
    the x-grid, since we need Simpson integration here rather than cumsum.
    """
    cdf_vals = integrate.cumulative_simpson(cdf, x=x, initial=0.0)
    cdf_vals = cdf_vals / cdf_vals[-1]
    # PCHIP requires strictly increasing x-data (here, the cdf values);
    # de-duplicate any flat/repeated cdf values (e.g. in zero-pdf regions)
    cdf_u, idx = np.unique(cdf_vals, return_index=True)
    x_u = x[idx]
    inv_cdf = PchipInterpolator(cdf_u, x_u, extrapolate=True)
    u = np.random.rand(n_samples)
    return inv_cdf(np.clip(u, cdf_u[0], cdf_u[-1]))


def get_samples_pdf_2d(n_samples, x_arr, y_arr, pdf_xy):
    eps = 1e-12
    pdf_x = integrate.simpson(pdf_xy, x=y_arr, axis=1)
    x_samp = _get_samples_pdf_1d(n_samples, x_arr, pdf_x)

    n_x = pdf_xy.shape[0]
    cdf_y_given_x = np.zeros_like(pdf_xy)
    for ix in range(n_x):
        row_cdf = integrate.cumulative_simpson(pdf_xy[ix], x=y_arr, initial=0.0)
        mx = row_cdf[-1]
        cdf_y_given_x[ix] = row_cdf / mx if mx > eps else row_cdf

    ix_nearest = np.argmin(np.abs(x_arr[:, None] - x_samp[None, :]), axis=0)
    u = np.random.rand(n_samples)
    y_samp = np.zeros(n_samples)

    # build one PCHIP per unique row, invert all samples sharing it at once
    for ix in np.unique(ix_nearest):
        row_cdf = cdf_y_given_x[ix]
        cdf_u, idx = np.unique(row_cdf, return_index=True)
        mask = (ix_nearest == ix)
        if len(cdf_u) < 2:
            y_samp[mask] = y_arr[idx[0]] if len(idx) else y_arr[0]
            continue
        y_u = y_arr[idx]
        inv_cdf = PchipInterpolator(cdf_u, y_u, extrapolate=True)
        y_samp[mask] = inv_cdf(np.clip(u[mask], cdf_u[0], cdf_u[-1]))

    return x_samp, y_samp

# ── shear radial bins (unchanged) ─────────────────────────────────────────────
def shape_noise_per_bin(R_bins_Mpc_over_h, DA_Mpc_over_h):
    dR                    = np.gradient(R_bins_Mpc_over_h)
    arcmin_per_Mpc_over_h = 180. * 60. / (np.pi * DA_Mpc_over_h)
    A_ann                 = 2. * np.pi * R_bins_Mpc_over_h * dR * arcmin_per_Mpc_over_h**2
    N_src                 = N_SOURCE * A_ann
    return SIGMA_SHAPE / np.sqrt(np.maximum(N_src, 1.))


# ── per-redshift slice loader (unchanged) ─────────────────────────────────────

def load_slice(z, hmf_type):
    if hmf_type[:8] == 'LCDM_hat':
        with open(f"{HALO_INPUT_FILE}/{z}_data_LCDM_full2h{FDM_suffix}.pkl", "rb") as f:
            data = pickle.load(f)
        axion_dic = None
        r_t, rho_crit, rho_char, concentration, r_s, Delta_vir, rho_0, rho_0_wh = \
            get_some_quantities(data, axion_dic=axion_dic)
        gamma_val        = float(hmf_type[8:11])
        if FDM_suffix == "_preferred":
            baryonic_profile = np.load(
                f"{FEEDBACK_INPUT_FILE}/{FEEDBACK_MODEL}/"
                f"utilde_results_gamma{gamma_val}_z{z}.npy",
                allow_pickle=True)[()]
        else:
            baryonic_profile = np.load(
                f"{FEEDBACK_INPUT_FILE}/{FEEDBACK_MODEL}/"
                f"utilde{FDM_suffix}_results_gamma{gamma_val}_z{z}.npy",
                allow_pickle=True)[()]
        Mtot = data['M']
        hmf  = baryonic_profile[-1]['hmf'] * Mtot
        rho  = baryonic_profile[-1]['rho_hat_unbiased']
        r_t  = baryonic_profile[-1]['r_t_unbiased']
        axion_dic = None

    elif hmf_type == 'LCDM':
        with open(f"{HALO_INPUT_FILE}/{z}_data_LCDM_full2h{FDM_suffix}.pkl", "rb") as f:
            data = pickle.load(f)
        Mtot, hmf_log10 = dn_dlog10Mtot(data, hmf_type)
        hmf  = hmf_log10 / np.log(10) # dn/dlnMvir in h units
        axion_dic = None
        r_t, rho_crit, rho_char, concentration, r_s, Delta_vir, rho_0, rho_0_wh = \
            get_some_quantities(data, axion_dic=axion_dic)
        rho  = get_realspace_profile(
            u_k=get_NFW_profile(r_v=r_t, k_arr=K_ARR, r_s=r_s, r_t=r_t),
            r_arr=R_ARR, k_arr=K_ARR, M_arr=Mtot, r_t=r_t)

    elif hmf_type[:3] == 'FDM':
        fname = f"{HALO_INPUT_FILE}/{z}_data_FDM_full2h{FDM_suffix}.pkl"
        with open(fname, "rb") as f:
            data = pickle.load(f)
        Mtot, hmf_log10 = dn_dlog10Mtot(data, hmf_type)
        hmf  = hmf_log10 / np.log(10)
        axion_dic = data['axion_param']
        r_t, rho_crit, rho_char, concentration, r_s, Delta_vir, rho_0, rho_0_wh = \
            get_some_quantities(data, axion_dic=axion_dic)
        rho_ax, rho_c = get_FDM_profile(
            data, r_arr=R_ARR, M_arr=data['M_CDM'], k_arr=K_ARR, full_output=True)
        rho = rho_ax + rho_c

    elif hmf_type in ('LCDM_COSMOCNC', 'LCDM_COSMOCNC_SR'):
        with open(f"{HALO_INPUT_FILE}/{z}_data_LCDM_full2h{FDM_suffix}.pkl", "rb") as f:
            data = pickle.load(f)
        Mtot = data['M']
        axion_dic = None
        r_t, rho_crit, rho_char, concentration, r_s, Delta_vir, rho_0, rho_0_wh = \
            get_some_quantities(data, axion_dic=axion_dic)
        rho  = get_realspace_profile(
            u_k=get_NFW_profile(r_v=r_t, k_arr=K_ARR, r_s=r_s, r_t=r_t),
            r_arr=R_ARR, k_arr=K_ARR, M_arr=Mtot, r_t=r_t)
        M200c, R200c = Mtot_to_MDelta(
            data, r_t=r_t, M_arr=Mtot, r_arr=R_ARR,
            Del=200, reference_density="critical",
            axion_dic=None, rho=rho, k_arr=K_ARR)

        hmf = np.array([
            build_cosmocnc_hmf_M200c(nc, z, np.log(m), data) for m in M200c
        ]) # dn/dlnM200c, in h-units

    else:
        raise ValueError(f"Unknown hmf_type: {hmf_type}")

    return data, Mtot, hmf, rho, r_t, axion_dic, concentration


if INCLUDE_SHEAR_PROFILES:
    _G_SI    = 6.674e-11
    _c_SI    = 2.998e8
    _Mpc_m   = 3.0857e22
    _Msol_kg = 1.989e30

    def compute_Sigma_crit_arr(omega_m, h_fid=h_fid,
                                _z_source_bins=_z_source_bins):
        """Unchanged from original. Returns Sigma_crit in Msol/h/(Mpc/h)²."""
        Om0   = omega_m / h_fid**2
        cosmo = FlatLambdaCDM(H0=h_fid * 100.0, Om0=Om0)
        Sigma_crit_arr = np.zeros(len(Z_ARR))

        for j, z_l in enumerate(Z_ARR):
            usable_sources = _z_source_bins[_z_source_bins > z_l]
            if len(usable_sources) == 0:
                Sigma_crit_arr[j] = np.inf
                continue
            DA_l = cosmo.angular_diameter_distance(z_l).value

            inv_scs = []
            for z_s in usable_sources:
                DA_s  = cosmo.angular_diameter_distance(z_s).value
                DA_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
                if DA_ls <= 0:
                    continue
                inv_sc = (
                    4.0 * np.pi * _G_SI / _c_SI**2
                    * (DA_l * _Mpc_m)
                    * (DA_ls * _Mpc_m)
                    / (DA_s * _Mpc_m)
                )
                inv_scs.append(inv_sc * _Msol_kg / _Mpc_m**2)

            Sigma_crit_arr[j] = (
                1.0 / np.mean(inv_scs) if len(inv_scs) > 0 else np.inf
            )

        # UNCHANGED: divide by h_fid, matching get_log_lik_externalShear convention
        return Sigma_crit_arr / h_fid   # Msol/h/(Mpc/h)²


# ── main catalog generator ────────────────────────────────────────────────────

def generate_mock_catalog(hmf_type, seed=42):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"  HMF type : {hmf_type}   seed : {seed}")
    print(f"{'='*60}")

    hmf_M_matrix      = np.zeros((len(Z_ARR), NUM_M_SAMPLE))
    q_of_M200c_slices = []
    M200c_slices      = []
    R200c_slices      = []
    Mvir_slices       = []
    Rvir_slices       = []
    shear_slices      = []
    Sigma_slices      = []
    DeltaSigma_slices = []
    R_proj_slices     = []

    # NEW: per-slice accumulators for wl_data block
    rho_crit_z_h_slices  = []   # rho_crit [Msol/h/(Mpc/h)³] per z slice
    r_s_slices           = []   # r_s [Mpc/h] per z slice, on M200c grid

    Sigma_crit_arr = None

    for i, z in tqdm(enumerate(Z_ARR), total=len(Z_ARR), desc="Building HMF grid"):

        data, Mtot, hmf, rho, r_t, axion_dic, concentration = load_slice(z, hmf_type)

        if INCLUDE_SHEAR_PROFILES:
            if Sigma_crit_arr is None:
                # UNCHANGED: same call as original
                Sigma_crit_arr = compute_Sigma_crit_arr(data['cosmos']['omega_m_0'])

        # NEW: extract rho_crit and r_s from get_some_quantities for wl_data.
        # get_some_quantities returns rho_crit in Msol/Mpc³ (verified).
        # Convert to Msol/h/(Mpc/h)³ = Msol/Mpc³ / h²
        _, rho_crit_Msol_Mpc3, _, _, r_s_full, _, _, _ = \
            get_some_quantities(data, axion_dic=axion_dic)
        rho_crit_z_h_slices.append(rho_crit_Msol_Mpc3 / h_fid**2)

        valid = Mtot > M_TRANSITION
        Mtot  = Mtot[valid]
        hmf   = hmf[valid]
        rho   = rho[valid]
        r_t   = r_t[valid]
        concentration = concentration[valid]
        r_s_valid = r_s_full[valid]   # Mpc/h, on same mass grid as Mtot

        M200c, R200c = Mtot_to_MDelta(
            data, r_t=r_t, M_arr=Mtot, r_arr=R_ARR,
            Del=M_DELTA, reference_density=M_REF_DENS,
            axion_dic=axion_dic, rho=rho, k_arr=K_ARR)

        # getting to dn/dlnM200c in h units
        if hmf_type in ('LCDM_COSMOCNC', 'LCDM_COSMOCNC_SR'):
            hmf_M200c = hmf
        else:
            hmf_M200c = hmf * np.gradient(np.log(Mtot)) / np.gradient(np.log(M200c)) # dn/dlnM200c

        if hmf_type == 'LCDM_COSMOCNC_SR':
            M500c, R500c = Mtot_to_MDelta(
                data, r_t=r_t, M_arr=Mtot, r_arr=R_ARR, Del=500,
                reference_density=M_REF_DENS, axion_dic=axion_dic, rho=rho, k_arr=K_ARR)
            if SR_params == "PlanckSZiFi":
                q_of_M200c = q_via_survey_sr(M500c, z, nc, A_sz=-4.3054, alpha_sz=1.1233,
                                               beta_sz=2.0, bias_sz=0.8, dof=0.0, h_fid=h_fid)
            elif SR_params == "bestfitSR":
                q_of_M200c = q_via_survey_sr(M500c, z, nc, A_sz=-4.352890598769685, alpha_sz=1.1991734303623067,
                                               beta_sz=1.9996943667059135, bias_sz=0.8, dof=0.0, h_fid=h_fid)
        else:
            if pressure_prescription == "AricoBCM":
                HM_sigma = func_sigma_M(Mtot, data['power_spec_dic']['k'], data['power_spec_dic']['power_total'], data['cosmos']['Omega_m_0'])
                HM_delta_c = func_delta_c(z, 
                                          data['cosmos']['Omega_ax_0'], data['cosmos']['Omega_m_0'], data['cosmos']['Omega_w_0'], 
                                          data['cosmos']['G_a'], data['cosmos']['version'])
                HM_nu = HM_delta_c / HM_sigma
                
                q_of_M200c = profile_to_q(
                    data, axion_dic=axion_dic, rho=rho, r_t=r_t,
                    M_arr=Mtot, k_arr=K_ARR, r_arr=R_ARR,
                    bias_sz=0.8, dof=0.0, 
                    nu=HM_nu,
                    theta_out=4.0,          # sets the polytropic index
                    theta_co=0.1,           # BCM core radius / r200c   (fixed in BCM)
                    theta_ej=4.0,           # BCM ejection radius / r200c
                    M_gas_c=1.2e14,         # beta_gas pivot mass, Msun/h
                    mu_gas=0.31,            # beta_gas slope
                    A_star=0.09,            # BCM stellar normalisation
                    M_star_0=2.5e11,        # BCM stellar pivot, Msun/h
                    eta_star=0.32,          # total stellar fraction slope
                    f_th_model=fth_prescription,
                    pressure_profile_prescription = "AricoBCM",
                    r_s = r_s_valid, conc = concentration)
            elif pressure_prescription == "baryonPasting":
                HM_sigma = func_sigma_M(Mtot, data['power_spec_dic']['k'], data['power_spec_dic']['power_total'], data['cosmos']['Omega_m_0'])
                HM_delta_c = func_delta_c(z, 
                                          data['cosmos']['Omega_ax_0'], data['cosmos']['Omega_m_0'], data['cosmos']['Omega_w_0'], 
                                          data['cosmos']['G_a'], data['cosmos']['version'])
                HM_nu = HM_delta_c / HM_sigma
                
                q_of_M200c = profile_to_q(
                    data, axion_dic=axion_dic, rho=rho, r_t=r_t,
                    M_arr=Mtot, k_arr=K_ARR, r_arr=R_ARR,
                    bias_sz=0.8, dof=0.0, 
                    nu=HM_nu,
                    Gamma=1.2,              # Shaw et al. (2010) default
                    B_param=None,           # None -> B = 1 (Arico-equivalent normalisation)
                    A_star=0.09,
                    M_star_0=2.5e11,
                    eta_star=0.32,
                    f_th_model=fth_prescription,
                    pressure_profile_prescription = "baryonPasting",
                    r_s = r_s_valid, conc = concentration)
            else:
                q_of_M200c = profile_to_q(
                    data, axion_dic=axion_dic, rho=rho, r_t=r_t,
                    M_arr=Mtot, k_arr=K_ARR, r_arr=R_ARR,
                    bias_sz=0.8, dof=0.0, pressure_profile_prescription = pressure_prescription)

        #q_min_mask       = q_of_M200c >= Q_MIN
        #hmf_M200c_masked = hmf_M200c * q_min_mask
        x = (q_of_M200c - Q_FLOOR) / (Q_MIN - Q_FLOOR)
        taper = np.clip(x, 0., 1.)
        taper = taper**2 * (3. - 2.*taper)   # smoothstep: C1-continuous, 0 at floor, 1 at Q_MIN_SOFT
        hmf_M200c_masked = hmf_M200c * taper

        hmf_M_matrix[i] = np.interp(
            ln_M_target,
            np.log(M200c),
            hmf_M200c_masked * dV_dz(data),
            left=0.0, right=0.0)

        q_of_M200c_slices.append(q_of_M200c)
        M200c_slices.append(M200c)
        R200c_slices.append(R200c)
        Mvir_slices.append(Mtot)
        Rvir_slices.append(r_t)

        # NEW: store r_s on the M200c mass grid for c200c computation
        # c200c = R200c / r_s (both Mpc/h) — stored per slice, interpolated per cluster
        c200c_slice   = R200c / r_s_valid   # dimensionless
        c200c_on_grid = np.interp(
            ln_M_target, np.log(M200c), c200c_slice,
            left=c200c_slice[0], right=c200c_slice[-1])
        r_s_slices.append(c200c_on_grid)   # reusing name: stores c200c on ln_M_target

        if INCLUDE_SHEAR_PROFILES:
            # UNCHANGED: identical call to original
            shear, deltasigma, sigma, R_proj = get_tangential_shear(
                rho, R_ARR, z, Mtot, M200c, r_t, Sigma_crit_arr[i],
                M_min_to_consider=M_TRANSITION, full_output=True)
            shear_slices.append(shear)
            Sigma_slices.append(sigma)
            DeltaSigma_slices.append(deltasigma)
            R_proj_slices.append(R_proj)

    # ── Poisson sampling (unchanged) ──────────────────────────────────────────
    hmf_M_fsky = hmf_M_matrix * FSKY
    dndz_M     = integrate.simpson(hmf_M_fsky, x=ln_M_target, axis=1)
    n_mean     = integrate.simpson(dndz_M, x=Z_ARR)
    n_tot      = int(rng.poisson(lam=n_mean))
    print(f"  Mean N (q_mean > Q_MIN={Q_MIN}) : {n_mean:.1f}")
    print(f"  Poisson draw N_tot            : {n_tot}")

    z_samp, lnM_samp = get_samples_pdf_2d(
        n_tot, Z_ARR, ln_M_target, hmf_M_fsky)
    M200c_samp = np.exp(lnM_samp)

    z_idx = np.argmin(np.abs(Z_ARR[:, None] - z_samp[None, :]), axis=0)
    q_mean = np.array([
        np.interp(M200c_samp[j], M200c_slices[iz], q_of_M200c_slices[iz])
        for j, iz in enumerate(z_idx)])
    Mvir_true = np.array([
        np.interp(M200c_samp[j], M200c_slices[iz], Mvir_slices[iz])
        for j, iz in enumerate(z_idx)])

    ln_q_true = np.log(q_mean) + rng.normal(0., SIGMA_LN_Q_INTRINSIC, size=n_tot)
    q_true    = np.exp(ln_q_true)
    q_obs     = q_true + rng.normal(0., SIGMA_Q_OBS, size=n_tot)

    sel   = q_obs > Q_TH
    n_sel = sel.sum()
    print(f"  Selected N (q_obs > {Q_TH}) : {n_sel}")

    all_halos_catalog = {
        'z'          : z_samp,        # (n_tot,)
        'M200c_true' : M200c_samp,    # (n_tot,) Msol/h
        'q_mean'     : q_mean,        # (n_tot,) noiseless q-M scaling relation
        'hmf_type'   : hmf_type,
        'seed'       : seed,
        'n_tot_mean' : float(n_mean),
        'n_tot_draw' : n_tot,
        'mass_units' : 'Msol/h',
        'h_fid'      : h_fid,
        'FSKY'       : FSKY,
        'Z_ARR'      : Z_ARR,
    }

    z_cat     = z_samp[sel]
    M200c_cat = M200c_samp[sel]
    iz_cat    = z_idx[sel]

    # ── Shear profiles (unchanged) ────────────────────────────────────────────
    N_R    = len(shear_slices[0][0])
    gt_list = []
    for idx in range(n_sel):
        iz       = iz_cat[idx]
        Mi       = M200c_cat[idx]
        gt_slice = shear_slices[iz]
        M200c_sl = M200c_slices[iz]

        valid_mask = np.array([g is not None for g in gt_slice])
        M_valid    = M200c_sl[valid_mask]
        gt_valid   = np.array([gt_slice[j]
                                for j in range(len(gt_slice)) if valid_mask[j]])

        gt_i = np.array([
            np.interp(np.log(Mi), np.log(M_valid), gt_valid[:, jr])
            for jr in range(N_R)])
        gt_list.append(gt_i)

    # ── NEW: build wl_data block for cosmocnc internal likelihood ─────────────
    # All arrays built from quantities already computed above.
    # Units:
    #   rho_crit_z_h : Msol/h/(Mpc/h)³  (rho_crit_Msol_Mpc3 / h_fid²)
    #   c_200c_grid  : dimensionless      (R200c/r_s on ln_M_target grid)
    #   R_proj       : Mpc/h              (directly from get_tangential_shear)
    #   gt_obs       : dimensionless      (same values as gt_list, padded)
    #   sigma_g      : dimensionless      (shape noise per bin)
    #   Sigma_crit   : Msol/h/(Mpc/h)²   (same as Sigma_crit_arr used for shear gen)

    # Build cosmology once for DA
    omega_m_0 = data['cosmos']['omega_m_0']
    cosmo_wl  = FlatLambdaCDM(H0=h_fid * 100., Om0=omega_m_0 / h_fid**2)

    gt_obs_pad  = np.full((n_sel, N_R_MAX), np.nan)
    sigma_g_pad = np.full((n_sel, N_R_MAX), 1e10)   # 1e10 → masked, zero chi² contribution
    R_proj_pad  = np.full((n_sel, N_R_MAX), np.nan)
    c200c_pad   = np.zeros((n_sel, NUM_M_SAMPLE))
    rho_crit_cl = np.zeros(n_sel)
    Sigma_cl    = np.zeros(n_sel)

    for idx in range(n_sel):
        iz = iz_cat[idx]
        zi = z_cat[idx]

        DA_h = cosmo_wl.angular_diameter_distance(zi).value * h_fid

        c200c_pad[idx]   = r_s_slices[iz]
        rho_crit_cl[idx] = rho_crit_z_h_slices[iz]
        Sigma_cl[idx]    = Sigma_crit_arr[iz]

        R_ref = R_proj_slices[iz]   # full reference grid from get_tangential_shear

        # Exact same R grid construction as external likelihood
        r_min  = 0.5
        r_max  = 3.2 / (1. + zi)
        if r_max <= r_min:
            r_max = r_min * 2.
        n_r   = max(int(np.round((r_max - r_min) / DELTA_R_BOCQUET)) + 1, 2)
        R_bins = np.linspace(r_min, r_max, n_r)

        R_proj_pad[idx, :n_r] = R_bins
        # padded bins keep np.nan

        # Re-interpolate gt onto R_bins (same as external likelihood's gt_new_list)
        gt_rebinned = np.interp(R_bins, R_ref, gt_list[idx])
        gt_obs_pad[idx, :n_r] = gt_rebinned
        # padded bins keep np.nan

        # Shape noise:
        dR_i    = np.gradient(R_bins)
        A_ann_i = (2. * np.pi * R_bins * dR_i
                   * (180. * 60. / np.pi / DA_h)**2)
        sigma_i = SIGMA_SHAPE / np.sqrt(np.maximum(N_SOURCE * A_ann_i, 1.)) # sigma_g,t
        noise = rng.normal(0., sigma_i, size=n_r)   # (n_r,) draw per bin
        gt_obs_pad[idx, :n_r] = gt_rebinned + noise
        sigma_g_pad[idx, :n_r] = sigma_i

    # ── Assemble catalog (all original keys preserved) ────────────────────────
    catalog = {
        # per-cluster observables
        'z'             : z_samp[sel],
        'q_obs'         : q_obs[sel],
        'gt_true'       : gt_list,
        'R_proj_arr'    : [R_proj_slices[iz_cat[idx]] for idx in range(n_sel)],
        # true values per cluster (diagnostics only)
        'Sigma_crit_cluster' : np.interp(z_samp[sel], Z_ARR, Sigma_crit_arr),
        'M200c_true'    : M200c_cat,         # Msol/h  (directly sampled)
        'Mvir_true'     : Mvir_true[sel],    # Msol/h
        'q_mean'        : q_mean[sel],       # mean q from scaling relation
        'q_true'        : q_true[sel],       # after intrinsic scatter, before noise
        # grids saved to avoid recomputation
        'shear_profiles'     : shear_slices,
        'Sigma_profiles'     : Sigma_slices,
        'DeltaSigma_profiles': DeltaSigma_slices,
        'Sigma_crit'         : Sigma_crit_arr,
        'R200c_slices'       : R200c_slices,
        'M200c_slices'       : M200c_slices,
        'Rvir_slices'        : Rvir_slices,
        'Mvir_slices'        : Mvir_slices,
        # metadata
        'hmf_type'      : hmf_type,
        'seed'          : seed,
        'n_tot_mean'    : float(n_mean),
        'n_tot_draw'    : n_tot,
        'n_selected'    : n_sel,
        # units / survey params
        'mass_units'    : 'Msol/h',
        'h_fid'         : h_fid,
        'Q_TH'          : Q_TH,
        'FSKY'          : FSKY,
        'SIGMA_LN_Q'    : SIGMA_LN_Q_INTRINSIC,
        'SIGMA_Q_OBS'   : SIGMA_Q_OBS,
        'Z_ARR'         : Z_ARR,
        # ── new: cosmocnc internal likelihood block ────────────────────────────
        'wl_data'  : {
            'rho_crit_z_h' : rho_crit_cl,     # (n_cl,)           Msol/h/(Mpc/h)³
            'c_200c_grid'  : c200c_pad,        # (n_cl, NUM_M_SAMPLE) dimensionless
            'R_proj'       : R_proj_pad,       # (n_cl, N_R_MAX)   Mpc/h; NaN=padded
            'gt_obs'       : gt_obs_pad,       # (n_cl, N_R_MAX)   shear; NaN=padded
            'sigma_g'      : sigma_g_pad,      # (n_cl, N_R_MAX)   noise; 1e10=masked
            'Sigma_crit'   : Sigma_cl,         # (n_cl,)           Msol/h/(Mpc/h)²
            'N_R_MAX'            : N_R_MAX,
            'lnM_grid_Msol_per_h': ln_M_target,  # (NUM_M_SAMPLE,) ln(M [Msol/h])
        }
    }

    return catalog, all_halos_catalog


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    for hmf_type in HMF_TYPES:
        SRparams_suffix = ""
        if hmf_type == 'LCDM_COSMOCNC_SR':
            if SR_params == "PlanckSZiFi":
                SRparams_suffix = "_PlanckSZiFi"
            elif SR_params == "bestfitSR":
                SRparams_suffix = ""
        for seed in SEEDS:
            catalog, all_halos_catalog   = generate_mock_catalog(hmf_type, seed=seed)
            safe_name = hmf_type.replace('.', 'p')
            out_path  = f"{OUTPUT_DIR}/{safe_name}{PRESSURE_SUFFIX}{wl_type_suffix}{SRparams_suffix}_mock_seed{seed}.pkl"
            all_halos_path = f"{OUTPUT_DIR}/{safe_name}{PRESSURE_SUFFIX}{wl_type_suffix}{SRparams_suffix}_mock_seed{seed}_noselection.pkl"

            with open(out_path, "wb") as f:
                pickle.dump(catalog, f)
            with open(all_halos_path, "wb") as f:
                pickle.dump(all_halos_catalog, f)

            '''
            print(f"  Saved -> {out_path}")
            print(f"  Clusters in catalog : {catalog['n_selected']}")
            print(f"  gt_true[0][0]       : {catalog['gt_true'][0][0]:.8f}")
            print(f"  wl_data gt_obs[0,0] : {catalog['wl_data']['gt_obs'][0,0]:.8f}  (should match)")
            print(f"  rho_crit_cl[0]      : {catalog['wl_data']['rho_crit_z_h'][0]:.4e}  Msol/h/(Mpc/h)³")
            print(f"  Sigma_crit_cl[0]    : {catalog['wl_data']['Sigma_crit'][0]:.4e}  Msol/h/(Mpc/h)²")
            '''

    print("\nAll mock catalogs generated.")