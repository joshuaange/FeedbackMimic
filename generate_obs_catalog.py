"""
generate_mock_catalogs.py

Generate mock cluster catalogs for comparing LCDM vs FDM/baryonic HMF bias.

Inputs (from your cluster_counts pipeline):
    - hmf_unmasked_matrix : (nZ, nM)  dn/dlnM * dV/dz  [no q cut, no fsky]
                            indexed at (Z_ARR, M200c_target) in Msol/h
    - q_vals_matrix       : (nZ, nM)  q(M, z) from profile_to_q
    - M200c_target        : (nM,)     M200c in Msol/h
    - Z_ARR               : (nZ,)     redshift array

Outputs:
    Per hmf_type, a pickle at OUTPUT_DIR/<hmf_type>_mock_seed<seed>.pkl
    containing a catalog dict with one entry per selected cluster.

Mass unit convention:
    - ALL internal masses are in Msol/h (matching your halo model)
    - Only divide by h when interfacing with cosmocnc machinery (in the MCMC)
    - The catalog stores masses in Msol/h; conversions are the MCMC's job
"""

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
    'LCDM_hat0.0', 'LCDM_hat0.5', 'LCDM_hat1.0', 'LCDM', 'FDM_preferred'
]

INCLUDE_SHEAR_PROFILES = True
etaz            = 0.7
FEEDBACK_MODEL  = f"etaz{etaz}_preferred_cooled"

N_MOCKS_PER_TYPE = 1       # increase for ensemble runs
SEEDS            = [42]    # one seed per mock; extend list for ensemble

# ── survey / cosmology ───────────────────────────────────────────────────────
Q_TH    = 5.0              # SO detection threshold
Q_MIN, Q_MAX = 0.1, 1000.0
NUM_Q        = 100
ln_q_target  = np.linspace(np.log(Q_MIN), np.log(Q_MAX), NUM_Q)
q_target     = np.exp(ln_q_target)

FSKY    = 0.4              # SO sky fraction (already baked into hmf_matrix)
h_fid   = 0.68             # h used throughout your halo model

# SO-like two-layer scatter (Zubeldia & Bolliet 2024, Sec 5.1)
SIGMA_LN_Q_INTRINSIC = 0.173   # layer 1: lognormal intrinsic scatter on ln q
SIGMA_Q_OBS          = 1.0     # layer 2: unit Gaussian observational noise

# Weak-lensing shape noise (Bocquet+2024 / SO-like)
SIGMA_SHAPE = 0.37    # per-galaxy shape noise
N_SOURCE    = 5.5     # effective source density [arcmin^-2]

# Radial bins for shear profile [Mpc/h, physical at cluster redshift]
# Following Bocquet: 500 h^-1 kpc to 3.2/(1+z) h^-1 Mpc
N_R_BINS = 10

# Virial Mass, redshift grid (must match what cluster_counts used)
M_MIN        = 1e11        # Msol/h
M_MAX        = 1e16        # Msol/h
NUM_M        = 300
M_DELTA      = 200
M_REF_DENS   = "critical"
M_TRANSITION = 10**12.5  # Mask out low-mass halos where feedback profiles are unreliable
    # this must actually be a bit lower because things scatter up to a higher q

Z_ARR        = np.linspace(0.0, 2.0, 81)
Z_ARR = Z_ARR[Z_ARR > 0]   # dV/dz is undefined/zero at z=0

K_ARR = np.logspace(-6, 6, 8192)
R_ARR = np.logspace(-4, np.log10(1 / K_ARR.min()), 1500)

# ── noise file (SO MMF) ──────────────────────────────────────────────────────
file_theta_500_vec, file_sigma_sz_vec = np.load(
    "/home/jange/my_cnc_files/data/so_sim_sz_mf_noise.npy")
file_theta_500_vec *= 180. * 60. / np.pi   # rad -> arcmin
sigma_sz_poly = np.polyfit(
    np.log(file_theta_500_vec), np.log(file_sigma_sz_vec), deg=3)


# ── CDF-inversion sampler (mirrors sim.py get_samples_pdf_2d) ───────────────

def _get_samples_pdf_1d(n_samples, x, cdf):
    """Draw n_samples from a 1-D CDF by linear interpolation."""
    cdf = cdf / np.max(cdf)
    u   = np.random.rand(n_samples)
    return np.interp(u, cdf, x)


def get_samples_pdf_2d(n_samples, x_arr, y_arr, pdf_xy):
    """
    Draw (x, y) pairs from the 2-D PDF pdf_xy[i_x, i_y].

    Uses CDF-inversion: first draw y from the marginal P(y),
    then draw x from the conditional P(x|y).

    Parameters
    ----------
    x_arr   : (nX,) – e.g. redshift
    y_arr   : (nY,) – e.g. ln M
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
    # build a (nX, nY) matrix of cumulative CDFs along y
    cdf_y_given_x = np.cumsum(pdf_xy, axis=1) * (y_arr[1] - y_arr[0])
    for ix in range(cdf_y_given_x.shape[0]):
        mx = cdf_y_given_x[ix, -1]
        if mx > eps:
            cdf_y_given_x[ix] /= mx

    # for each drawn x, interpolate the CDF to get the y sample
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
    """
    Return radial bin centres in Mpc/h (physical at cluster redshift).
    Matches Bocquet: r_min = 0.5 h^-1 Mpc, r_max = 3.2/(1+z) h^-1 Mpc.
    """
    r_min = 0.5                     # h^-1 Mpc
    r_max = 3.2 / (1.0 + z)        # h^-1 Mpc (outer cut following Bocquet)
    if r_max <= r_min:
        r_max = r_min * 2.
    return np.linspace(r_min, r_max, n_bins)   # Mpc/h


def shape_noise_per_bin(R_bins_Mpc_over_h, DA_Mpc_over_h):
    """
    σ_gt(r) = σ_shape / sqrt(N_source * A_annulus)
    R_bins and DA must be in consistent units (both Mpc/h here).
    """
    dR      = np.gradient(R_bins_Mpc_over_h)
    # angular diameter distance in arcmin/Mpc/h
    arcmin_per_Mpc_over_h = 180. * 60. / (np.pi * DA_Mpc_over_h)
    A_ann   = 2. * np.pi * R_bins_Mpc_over_h * dR * arcmin_per_Mpc_over_h**2
    N_src   = N_SOURCE * A_ann
    return SIGMA_SHAPE / np.sqrt(np.maximum(N_src, 1.))


# ── per-redshift slice loader (mirrors cluster_counts.py) ───────────────────

def load_slice(z, hmf_type):
    """
    Load one redshift slice, return:
        data, Mtot, hmf (dn/dlnMtot), rho, r_t, axion_dic
    hmf is dn/dlnMtot (NOT divided by ln10 here; we divide below).
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
        hmf  = hmf_log10 / np.log(10)              # dn/dlnMtot
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
        """
        Compute Sigma_crit(z_l) for all lens redshifts in Z_ARR
        directly from cosmology calculations, using Bocquet's source redshifts
        """
        Om0 = omega_m / h_fid**2
        cosmo = FlatLambdaCDM(H0=h_fid * 100.0, Om0=Om0)
        Sigma_crit_arr = np.zeros(len(Z_ARR))
    
        for j, z_l in enumerate(Z_ARR):
            usable_sources = _z_source_bins[_z_source_bins > z_l]
            if len(usable_sources) == 0:
                Sigma_crit_arr[j] = np.inf
                continue
            DA_l = cosmo.angular_diameter_distance(z_l).value  # Mpc
    
            inv_scs = []
    
            for z_s in usable_sources:
    
                DA_s = cosmo.angular_diameter_distance(z_s).value  # Mpc
                DA_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value  # Mpc
    
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

    Returns a dict with per-cluster arrays (all selected clusters).
    Masses are in Msol/h throughout.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"  HMF type : {hmf_type}   seed : {seed}")
    print(f"{'='*60}")

    # Build hmf matrix (nZ, nM) and convert to dn/dlnq dz
    hmf_q_matrix   = np.zeros((len(Z_ARR), NUM_Q))
    q_of_M200c_slices  = []   # list of (nM,) arrays
    M200c_slices       = []   # list of (nM,) arrays
    R200c_slices       = []   # list of (nM,) arrays
    # note: for these: nR is from np.linspace(0.5, 3.2/(1+z), 100)
    shear_slices       = []   # list of (nM,nR) arrays
    Sigma_slices       = []   # list of (nM,nR) arrays
    DeltaSigma_slices       = []   # list of (nM,nR) arrays

    Sigma_crit_arr = None
    for i, z in tqdm(enumerate(Z_ARR), total=len(Z_ARR), desc="Building HMF grid"):

        data, Mtot, hmf, rho, r_t, axion_dic = load_slice(z, hmf_type)
        # note: hmf here is dn/dlnMtot
        if INCLUDE_SHEAR_PROFILES:
            if Sigma_crit_arr is None:
                Sigma_crit_arr = compute_Sigma_crit_arr(data['cosmos']['omega_m_0'])
        valid = Mtot > M_TRANSITION
        Mtot  = Mtot[valid]
        hmf   = hmf[valid]
        rho   = rho[valid]   # (nM, nR) -> adjust axis if needed
        r_t   = r_t[valid]

        # Convert Mtot -> M200c (still Msol/h)
        M200c, R200c = Mtot_to_MDelta(
            data, r_t=r_t, M_arr=Mtot, r_arr=R_ARR,
            Del=M_DELTA, reference_density=M_REF_DENS,
            axion_dic=axion_dic, rho=rho, k_arr=K_ARR)

        # Re-index from Mtot to M200c_target via Jacobian
        hmf_M200c  = hmf * np.gradient(np.log(Mtot)) / np.gradient(np.log(M200c))

        # q values at each mass (no scatter yet)
        q_of_M200c = profile_to_q(
            data, axion_dic=axion_dic, rho=rho, r_t=r_t,
            M_arr=Mtot, k_arr=K_ARR, r_arr=R_ARR,
            bias_sz=0.8, dof=0.0)
        
        q_of_M200c_slices.append(q_of_M200c)
        M200c_slices.append(M200c)
        R200c_slices.append(R200c)

        # Change of variables: dn/dlnq = dn/dlnM * |dlnM/dlnq|, sorted for safety, but probably unnecessary
        sort_idx   = np.argsort(q_of_M200c)
        q_sorted   = q_of_M200c[sort_idx]
        hmf_sorted = hmf_M200c[sort_idx]
        lnM_sorted = np.log(M200c[sort_idx])
        # Jacobian |dlnM/dlnq| on the sorted q grid
        dlnM_dlnq = np.abs(np.gradient(lnM_sorted) / np.gradient(np.log(q_sorted + 1e-30)))
        
        hmf_in_q = hmf_sorted * dlnM_dlnq * dV_dz(data)
        hmf_q_matrix[i] = np.interp(ln_q_target, np.log(q_sorted), hmf_in_q,
                                  left=0.0, right=0.0) # on uniform q grid

        if INCLUDE_SHEAR_PROFILES:
            shear, deltasigma, sigma = get_tangential_shear(rho, R_ARR, z, Mtot, M200c, r_t, Sigma_crit_arr[i],
                                                            M_min_to_consider = M_TRANSITION, full_output=True)
            shear_slices.append(shear)
            Sigma_slices.append(sigma)
            DeltaSigma_slices.append(deltasigma)

    # Poisson sampling for total number of clusters
    hmf_q_fsky = hmf_q_matrix * FSKY   # (nZ, nQ), survey-volume weighted
    dndz_q     = integrate.simpson(hmf_q_fsky, x=ln_q_target, axis=1)
    n_mean     = integrate.simpson(dndz_q, x=Z_ARR)
    n_tot      = int(rng.poisson(lam=n_mean))
    print(f"  Mean N (above Q_MIN, all q) : {n_mean:.1f}")
    print(f"  Poisson draw N_tot          : {n_tot}")

    z_samp, lnq_samp = get_samples_pdf_2d(
        n_tot, Z_ARR, ln_q_target, hmf_q_fsky)
    q_mean = np.exp(lnq_samp)   # mean q from scaling relation, before any scatter

    #Let's also get the true M200c:
    z_idx = np.argmin(np.abs(Z_ARR[:, None] - z_samp[None, :]), axis=0)
    M200c_true = np.array([
        np.interp(q_mean[j], q_of_M200c_slices[iz], M200c_slices[iz])
        for j, iz in enumerate(z_idx)
    ])
    Mvir_true = np.array([
        np.interp(q_mean[j], q_of_M200c_slices[iz], Mtot)
        for j, iz in enumerate(z_idx)
    ])

    # Two-layer scatter (cosmocnc Sec 5.1)
    # Layer 1 – intrinsic lognormal scatter on ln q_mean -> ln q_true
    # Layer 2 – unit Gaussian observational noise on q_true -> q_obs
    ln_q_true = np.log(q_mean) + rng.normal(0., SIGMA_LN_Q_INTRINSIC, size=n_tot)
    q_true    = np.exp(ln_q_true)
    q_obs     = q_true + rng.normal(0., SIGMA_Q_OBS, size=n_tot)

    # Apply selection cut
    sel   = q_obs > Q_TH
    n_sel = sel.sum()
    print(f"  Selected N (q_obs > {Q_TH}) : {n_sel}")

    catalog = {
        # per-cluster observables
        'z'             : z_samp[sel],
        'q_obs'         : q_obs[sel],
        # true values per cluster (diagnostics only)
        'M200c_true'    : M200c_true[sel],   # Msol/h
        'Mvir_true'     : Mvir_true[sel],    # Msol/h
        'q_mean'        : q_mean[sel],       # mean q from scaling relation
        'q_true'        : q_true[sel],       # after intrinsic scatter, before noise
        # other values saved, just to not have to recompute
        'shear_profiles': shear_slices,      # nZ, nM, nR -> mean tangential shear
        'Sigma_profiles': Sigma_slices,      # nZ, nM, nR -> mean Sigma
        'DeltaSigma_profiles': DeltaSigma_slices, # nZ, nM, nR -> mean DeltaSigma
        'Sigma_crit'    : Sigma_crit_arr,    # in nZ -> used for shear profiles
        'R200c_slices'  : R200c_slices, # nZ, nM
        'M200c_slices'  : M200c_slices, # nZ, nM
        # metadata
        'hmf_type'      : hmf_type,
        'seed'          : seed,
        'n_tot_mean'    : float(n_mean),
        'n_tot_draw'    : n_tot,
        'n_selected'    : n_sel,
        # units / survey params
        'mass_units'        : 'Msol/h',
        'h_fid'             : h_fid,
        'Q_TH'              : Q_TH,
        'FSKY'              : FSKY,
        'SIGMA_LN_Q'        : SIGMA_LN_Q_INTRINSIC,
        'SIGMA_Q_OBS'       : SIGMA_Q_OBS,
        'Z_ARR'             : Z_ARR,
    }

    return catalog


# ── run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    for hmf_type in HMF_TYPES:
        for seed in SEEDS:

            catalog = generate_mock_catalog(hmf_type, seed=seed)

            # build output filename
            safe_name = hmf_type.replace('.', 'p')
            out_path  = f"{OUTPUT_DIR}/{safe_name}_mock_seed{seed}.pkl"

            with open(out_path, "wb") as f:
                pickle.dump(catalog, f)

            print(f"  Saved -> {out_path}")
            print(f"  Clusters in catalog: {catalog['n_selected']}")

    print("\nAll mock catalogs generated.")