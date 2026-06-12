import os
import pickle
import numpy as np
from scipy import integrate, interpolate
from tqdm import tqdm
from astropy.cosmology import FlatLambdaCDM

# ── your existing imports ────────────────────────────────────────────────────
from feedback_mimic import (
    dn_dlog10Mtot, get_NFW_profile, get_realspace_profile,
    dV_dz, profile_to_q, Mtot_to_MDelta, get_some_quantities,
    get_FDM_profile, get_total_mass, get_tangential_shear
)

# ── paths ────────────────────────────────────────────────────────────────────
HALO_INPUT_FILE    = "/home/jange/scratch/FeedbackMimic/results/halo_model"
FEEDBACK_INPUT_FILE = "/home/jange/scratch/FeedbackMimic/results/feedback_models"
OUTPUT_DIR         = "/home/jange/scratch/FeedbackMimic/results/mock_catalogs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── configuration ────────────────────────────────────────────────────────────
HMF_TYPES = [
    'LCDM', 'FDM_preferred', 'LCDM_hat0.0', 'LCDM_hat0.5', 'LCDM_hat1.0'
]

INCLUDE_SHEAR_PROFILES = True
etaz            = 0.7
FEEDBACK_MODEL  = f"etaz{etaz}_preferred_cooled"

N_MOCKS_PER_TYPE = 1       # increase for ensemble runs
SEEDS            = [42]    # one seed per mock; extend list for ensemble

# ── survey / cosmology ───────────────────────────────────────────────────────
Q_TH    = 5.0              # SO detection threshold
Q_MIN   = 3.0              # pre-selection floor: halos with q_mean < Q_MIN
                           # are masked from the sampling distribution

FSKY    = 0.4              # SO sky fraction
h_fid   = 0.68             # h used throughout your halo model

# M-space sampling grid — uniform in ln M_200c
# Lower bound kept above M_TRANSITION so we never sample halos whose
# profiles are unreliable; upper bound generous to catch the rarest clusters.
M_MIN_SAMPLE = 10**12.5    # Msol/h  (matches M_TRANSITION)
M_MAX_SAMPLE = 1e16        # Msol/h
NUM_M_SAMPLE = 300
ln_M_target  = np.linspace(np.log(M_MIN_SAMPLE), np.log(M_MAX_SAMPLE), NUM_M_SAMPLE)
M_target     = np.exp(ln_M_target)

# SO-like two-layer scatter (Zubeldia & Bolliet 2024, Sec 5.1)
SIGMA_LN_Q_INTRINSIC = 0.173   # layer 1: lognormal intrinsic scatter on ln q
SIGMA_Q_OBS          = 1.0     # layer 2: unit Gaussian observational noise

# Weak-lensing shape noise (Bocquet+2024 / SO-like)
SIGMA_SHAPE = 0.37    # per-galaxy shape noise
N_SOURCE    = 5.5     # effective source density [arcmin^-2]

# Radial bins for shear profile [Mpc/h, physical at cluster redshift]
N_R_BINS = 10

# Virial Mass, redshift grid (must match what cluster_counts used)
M_MIN        = 1e11        # Msol/h
M_MAX        = 1e16        # Msol/h
NUM_M        = 300
M_DELTA      = 200
M_REF_DENS   = "critical"
M_TRANSITION = 10**12.5  # Mask out low-mass halos where feedback profiles are unreliable

Z_ARR        = np.linspace(0.0, 2.0, 81)
Z_ARR = Z_ARR[Z_ARR > 0]   # dV/dz is undefined/zero at z=0

K_ARR = np.logspace(-6, 6, 8192)
R_ARR = np.logspace(-4, np.log10(1 / K_ARR.min()), 1500)

# ── noise file (SO MMF) ──────────────────────────────────────────────────────
file_theta_500_vec, file_sigma_sz_vec = np.load(
    "/home/jange/scratch/FeedbackMimic/cosmocnc_axionHMcode_jax/cosmocnc_jax/data/so_sim_sz_mf_noise.npy")
file_theta_500_vec *= 180. * 60. / np.pi   # rad -> arcmin
sigma_sz_poly = np.polyfit(
    np.log(file_theta_500_vec), np.log(file_sigma_sz_vec), deg=3)


# ── CDF-inversion sampler ────────────────────────────────────────────────────

def _get_samples_pdf_1d(n_samples, x, cdf):
    """Draw n_samples from a 1-D CDF by linear interpolation."""
    cdf = cdf / np.max(cdf)
    u   = np.random.rand(n_samples)
    return np.interp(u, cdf, x)


def get_samples_pdf_2d(n_samples, x_arr, y_arr, pdf_xy):
    """
    Draw (x, y) pairs from the 2-D PDF pdf_xy[i_x, i_y].

    Uses CDF-inversion: first draw x from the marginal P(x),
    then draw y from the conditional P(y|x).

    Parameters
    ----------
    x_arr   : (nX,) – e.g. redshift
    y_arr   : (nY,) – e.g. ln M_200c
    pdf_xy  : (nX, nY)

    Returns
    -------
    x_samples, y_samples : each (n_samples,)
    """
    eps = 1e-12

    # marginal CDF over x (redshift)
    pdf_x   = np.sum(pdf_xy, axis=1) * (y_arr[1] - y_arr[0])
    cdf_x   = np.cumsum(pdf_x) * (x_arr[1] - x_arr[0])
    x_samp  = _get_samples_pdf_1d(n_samples, x_arr, cdf_x)

    # conditional CDF P(y | x) for each drawn x
    cdf_y_given_x = np.cumsum(pdf_xy, axis=1) * (y_arr[1] - y_arr[0])
    for ix in range(cdf_y_given_x.shape[0]):
        mx = cdf_y_given_x[ix, -1]
        if mx > eps:
            cdf_y_given_x[ix] /= mx

    interp_cdf = interpolate.RegularGridInterpolator(
        (x_arr, y_arr), cdf_y_given_x,
        method='linear', bounds_error=False, fill_value=None)

    u       = np.random.rand(n_samples)
    y_samp  = np.zeros(n_samples)
    for i, (xi, ui) in enumerate(zip(x_samp, u)):
        cdf_i  = interp_cdf(np.column_stack(
            [np.full_like(y_arr, xi), y_arr]))
        y_samp[i] = np.interp(ui, np.clip(cdf_i, 0, 1), y_arr)

    return x_samp, y_samp


# ── shear radial bins for a given cluster ────────────────────────────────────

def make_R_bins(z, DA_Mpc_over_h, n_bins=N_R_BINS):
    r_min = 0.5
    r_max = 3.2 / (1.0 + z)
    if r_max <= r_min:
        r_max = r_min * 2.
    return np.linspace(r_min, r_max, n_bins)


def shape_noise_per_bin(R_bins_Mpc_over_h, DA_Mpc_over_h):
    dR      = np.gradient(R_bins_Mpc_over_h)
    arcmin_per_Mpc_over_h = 180. * 60. / (np.pi * DA_Mpc_over_h)
    A_ann   = 2. * np.pi * R_bins_Mpc_over_h * dR * arcmin_per_Mpc_over_h**2
    N_src   = N_SOURCE * A_ann
    return SIGMA_SHAPE / np.sqrt(np.maximum(N_src, 1.))


# ── per-redshift slice loader ────────────────────────────────────────────────

def load_slice(z, hmf_type):
    """
    Load one redshift slice, return:
        data, Mtot, hmf (dn/dlnMtot), rho, r_t, axion_dic
    """
    if hmf_type[:8] == 'LCDM_hat':
        with open(f"{HALO_INPUT_FILE}/{z}_data_LCDM_full2h_preferred.pkl", "rb") as f:
            data = pickle.load(f)
        axion_dic = None
        r_t, rho_crit, rho_char, concentration, r_s, Delta_vir, rho_0, rho_0_wh = \
            get_some_quantities(data, axion_dic=axion_dic)
        gamma_val     = float(hmf_type[8:])
        baryonic_profile = np.load(
            f"{FEEDBACK_INPUT_FILE}/{FEEDBACK_MODEL}/"
            f"utilde_results_gamma{gamma_val}_z{z}.npy",
            allow_pickle=True)[()]
        Mtot = data['M']
        hmf  = baryonic_profile[-1]['hmf'] * Mtot   # dn/dlnMtot
        rho  = baryonic_profile[-1]['rho_hat_unbiased']
        r_t  = baryonic_profile[-1]['r_t_unbiased']
        axion_dic = None

    elif hmf_type == 'LCDM':
        with open(f"{HALO_INPUT_FILE}/{z}_data_LCDM_full2h_preferred.pkl", "rb") as f:
            data = pickle.load(f)
        Mtot, hmf_log10 = dn_dlog10Mtot(data, hmf_type)
        hmf  = hmf_log10 / np.log(10)
        axion_dic = None
        r_t, rho_crit, rho_char, concentration, r_s, Delta_vir, rho_0, rho_0_wh = \
            get_some_quantities(data, axion_dic=axion_dic)
        rho  = get_realspace_profile(
            u_k=get_NFW_profile(r_v=r_t, k_arr=K_ARR, r_s=r_s, r_t=r_t),
            r_arr=R_ARR, k_arr=K_ARR, M_arr=Mtot, r_t=r_t)

    elif hmf_type == 'FDM_preferred':
        fname = (f"{HALO_INPUT_FILE}/{z}_data_FDM_full2h_preferred.pkl")
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

    else:
        raise ValueError(f"Unknown hmf_type: {hmf_type}")

    return data, Mtot, hmf, rho, r_t, axion_dic

if INCLUDE_SHEAR_PROFILES:
    _G_SI    = 6.674e-11
    _c_SI    = 2.998e8
    _Mpc_m   = 3.0857e22
    _Msol_kg = 1.989e30
    def compute_Sigma_crit_arr(omega_m, h_fid=h_fid,
                              _z_source_bins = np.array([0.285, 0.476, 0.743, 0.942])):
        Om0 = omega_m / h_fid**2
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
    
        return Sigma_crit_arr


# ── main catalog generator ───────────────────────────────────────────────────

def generate_mock_catalog(hmf_type, seed=42):
    """
    Generate one mock cluster catalog for a given HMF type.

    Sampling is performed directly in (z, ln M_200c) space, i.e. from
    d²N / (dlnM_200c dz).  After sampling, q_mean(M,z) is recovered by
    interpolation and scatter / selection are applied in q-space as before.

    Returns a dict with per-cluster arrays.  Masses are in Msol/h throughout.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"  HMF type : {hmf_type}   seed : {seed}")
    print(f"{'='*60}")

    # ── Build the 2D sampling matrix in (z, lnM_200c) space ─────────────────
    # hmf_M_matrix[i_z, i_M] = dn/dlnM_200c * dV/dz * f_sky
    # interpolated onto the uniform ln_M_target grid.
    hmf_M_matrix       = np.zeros((len(Z_ARR), NUM_M_SAMPLE))

    # Per-slice ancillary data (needed later for shear and q interpolation)
    q_of_M200c_slices  = []
    M200c_slices       = []
    R200c_slices       = []
    Mvir_slices        = []
    Rvir_slices        = []
    shear_slices       = []
    Sigma_slices       = []
    DeltaSigma_slices  = []
    R_proj_slices      = []

    Sigma_crit_arr = None
    for i, z in tqdm(enumerate(Z_ARR), total=len(Z_ARR), desc="Building HMF grid"):

        data, Mtot, hmf, rho, r_t, axion_dic = load_slice(z, hmf_type)

        if INCLUDE_SHEAR_PROFILES:
            if Sigma_crit_arr is None:
                Sigma_crit_arr = compute_Sigma_crit_arr(data['cosmos']['omega_m_0']) / h_fid

        valid = Mtot > M_TRANSITION
        Mtot  = Mtot[valid]
        hmf   = hmf[valid]
        rho   = rho[valid]
        r_t   = r_t[valid]

        # Convert Mtot -> M200c (still Msol/h)
        M200c, R200c = Mtot_to_MDelta(
            data, r_t=r_t, M_arr=Mtot, r_arr=R_ARR,
            Del=M_DELTA, reference_density=M_REF_DENS,
            axion_dic=axion_dic, rho=rho, k_arr=K_ARR)

        # Jacobian: dn/dlnM_200c = dn/dlnMtot * |dlnMtot/dlnM_200c|
        hmf_M200c = hmf * np.gradient(np.log(Mtot)) / np.gradient(np.log(M200c))

        # q(M,z) — needed both for the q_min mask below and for post-sampling lookup
        q_of_M200c = profile_to_q(
            data, axion_dic=axion_dic, rho=rho, r_t=r_t,
            M_arr=Mtot, k_arr=K_ARR, r_arr=R_ARR,
            bias_sz=0.8, dof=0.0)

        # ── Interpolate onto the uniform lnM grid, masking M below q_min ────
        # Zero out halos whose mean q is so low they can never scatter above
        # Q_TH (using the same Q_MIN pre-selection as the original q-space
        # sampler).  This keeps n_mean at the correct ~26k level and avoids
        # sampling 88M clusters that would all be rejected anyway.
        q_min_mask = q_of_M200c >= Q_MIN   # Q_MIN defined at top of file
        hmf_M200c_masked = hmf_M200c * q_min_mask

        hmf_M_matrix[i] = np.interp(
            ln_M_target,
            np.log(M200c),
            hmf_M200c_masked * dV_dz(data),   # dn/dlnM_200c * dV/dz
            left=0.0, right=0.0
        )

        q_of_M200c_slices.append(q_of_M200c)
        M200c_slices.append(M200c)
        R200c_slices.append(R200c)
        Mvir_slices.append(Mtot)
        Rvir_slices.append(r_t)

        if INCLUDE_SHEAR_PROFILES:
            shear, deltasigma, sigma, R_proj = get_tangential_shear(
                rho, R_ARR, z, Mtot, M200c, r_t, Sigma_crit_arr[i],
                M_min_to_consider=M_TRANSITION, full_output=True)
            shear_slices.append(shear)
            Sigma_slices.append(sigma)
            DeltaSigma_slices.append(deltasigma)
            R_proj_slices.append(R_proj)

    # ── Poisson sampling ─────────────────────────────────────────────────────
    hmf_M_fsky = hmf_M_matrix * FSKY   # (nZ, nM_sample)
    dndz_M     = integrate.simpson(hmf_M_fsky, x=ln_M_target, axis=1)
    n_mean     = integrate.simpson(dndz_M, x=Z_ARR)
    n_tot      = int(rng.poisson(lam=n_mean))
    print(f"  Mean N (q_mean > Q_MIN={Q_MIN}) : {n_mean:.1f}")
    print(f"  Poisson draw N_tot            : {n_tot}")

    # ── Sample (z, lnM_200c) pairs from d²N/(dlnM dz) ───────────────────────
    z_samp, lnM_samp = get_samples_pdf_2d(
        n_tot, Z_ARR, ln_M_target, hmf_M_fsky)
    M200c_samp = np.exp(lnM_samp)   # sampled M_200c in Msol/h

    # ── Evaluate q_mean(M,z) by interpolation at each sampled point ──────────
    z_idx = np.argmin(np.abs(Z_ARR[:, None] - z_samp[None, :]), axis=0)
    q_mean = np.array([
        np.interp(M200c_samp[j], M200c_slices[iz], q_of_M200c_slices[iz])
        for j, iz in enumerate(z_idx)
    ])
    Mvir_true = np.array([
        np.interp(M200c_samp[j], M200c_slices[iz], Mvir_slices[iz])
        for j, iz in enumerate(z_idx)
    ])

    # ── Two-layer scatter (unchanged from original) ───────────────────────────
    # Layer 1 – intrinsic lognormal scatter on ln q
    ln_q_true = np.log(q_mean) + rng.normal(0., SIGMA_LN_Q_INTRINSIC, size=n_tot)
    q_true    = np.exp(ln_q_true)
    # Layer 2 – unit Gaussian observational noise
    q_obs     = q_true + rng.normal(0., SIGMA_Q_OBS, size=n_tot)

    # ── Selection cut ─────────────────────────────────────────────────────────
    sel   = q_obs > Q_TH
    n_sel = sel.sum()
    print(f"  Selected N (q_obs > {Q_TH}) : {n_sel}")

    # ── Shear profiles for selected clusters ──────────────────────────────────
    z_cat     = z_samp[sel]
    M200c_cat = M200c_samp[sel]
    iz_cat    = z_idx[sel]

    N_R = len(shear_slices[0][0])
    gt_list = []
    for idx in range(n_sel):
        iz  = iz_cat[idx]
        Mi  = M200c_cat[idx]
        gt_slice = shear_slices[iz]
        M200c_sl = M200c_slices[iz]

        valid_mask = np.array([g is not None for g in gt_slice])
        M_valid    = M200c_sl[valid_mask]
        gt_valid   = np.array([gt_slice[j] for j in range(len(gt_slice)) if valid_mask[j]])

        log_M_grid = np.log(M_valid)
        log_Mi     = np.log(Mi)
        gt_i = np.array([np.interp(log_Mi, log_M_grid, gt_valid[:, jr])
                         for jr in range(N_R)])
        gt_list.append(gt_i)

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
    }

    return catalog


# ── run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    for hmf_type in HMF_TYPES:
        for seed in SEEDS:

            catalog = generate_mock_catalog(hmf_type, seed=seed)

            safe_name = hmf_type.replace('.', 'p')
            out_path  = f"{OUTPUT_DIR}/{safe_name}_mock_seed{seed}.pkl"

            with open(out_path, "wb") as f:
                pickle.dump(catalog, f)

            print(f"  Saved -> {out_path}")
            print(f"  Clusters in catalog: {catalog['n_selected']}")

    print("\nAll mock catalogs generated.")