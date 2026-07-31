import pickle
import numpy as np
import os
import cosmocnc_axionHMcode_jax.cosmocnc_jax as cosmocnc
from cosmocnc_axionHMcode_jax.cosmocnc_jax.hmf import (
    func_axionHMcode_D_z_unnorm_int,
    func_axionHMcode_D_z_unnorm)
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import simpson
import jax
import jax.numpy as jnp
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("jax").setLevel(logging.WARNING)
logging.disable(logging.DEBUG)


h_fid       = 0.68
omega_b_fid = 0.0226576
ns_fid      = 0.965

omega_m_fid = 0.143344
A_s_fid     = 2.1e-09

_SURVEY_SR  = "/home/user/FeedbackMimic/cosmocnc_axionHMcode_jax/cosmocnc_jax/surveys/survey_sr_so_sim.py"
_SURVEY_CAT = "/home/user/FeedbackMimic/cosmocnc_axionHMcode_jax/cosmocnc_jax/surveys/survey_cat_so_sim.py"

Z_ARR        = np.linspace(0.0, 2.0, 81)
Z_ARR = Z_ARR[Z_ARR > 0]

# Precompute a bunch of quantities so we dont have to do it on the fly, and can just interpolate
#################################################################################### COSMOLOGY CACHE
_astropy_cosmo_fid = FlatLambdaCDM(H0=h_fid*100., Om0=omega_m_fid/h_fid**2)
_omega_m_grid      = np.linspace(0.001, 0.299, 200)
_Om0_grid          = _omega_m_grid / h_fid**2
_z_table           = np.concatenate([np.linspace(1e-6, 1., 500), np.linspace(1., 100., 500)])

import os
#sTOMOGRAPHIC_BINS=DES python mcmc_likelihood.py
type_of_tomographic_bins = os.environ.get("TOMOGRAPHIC_BINS", "DES")
if type_of_tomographic_bins in ("DES", "Euclid"):
    # Euclid n_source/sigma_e still paired with DES Y3's tomographic bins
    # (the deliberate hybrid case) -- Sigma_crit depends only on the bins,
    # so this reuses the existing DES cache file untouched.
    _z_source_bins = np.array([0.285, 0.476, 0.743, 0.942])
    cache_suffix = ""
elif type_of_tomographic_bins == "deepEuclid":
    _z_source_bins = np.array([0.3195, 0.4941, 0.6206, 0.7338, 0.8440,
                                0.9581, 1.0841, 1.2336, 1.4331, 1.7875])
    cache_suffix = "_deepEuclid"
else:
    raise ValueError(f"Unknown type_of_tomographic_bins: {type_of_tomographic_bins!r}")
print(f"cache_suffix = {cache_suffix}")

from feedback_mimic import comoving_distance
_G_SI    = 6.674e-11
_c_SI    = 2.998e8
_Mpc_m   = 3.0857e22
_Msol_kg = 1.989e30

_cache_path = f"/home/user/FeedbackMimic/results/mcmc/MCMC_GRID_CACHE{cache_suffix}.pkl"
if os.path.exists(_cache_path):
    with open(_cache_path, "rb") as f:
        _G_a_table, _D_table, _norm_grid, _DA_table, _Sigma_crit_table = pickle.load(f)
    print("Loaded precomputed tables from cache")
else:
    print("Precomputing tables (one-time cost)...")
    def _G_a_of_z(Om0, Z=10000., n=200000):
        """G_a(z) reproducing axionHMcode's func_D_z_unnorm_int, vectorised over z.
           Same Z=10000 cutoff in both nested integrals as its dblquad uses."""
        Ow0 = 1. - Om0
        opz = np.logspace(0., np.log10(1.+Z), n)
        z   = opz - 1.
        E   = np.sqrt(np.abs(Om0*opz**3 + Ow0))
        def revcum(y, x):
            inc = (y[:-1]+y[1:])/2 * np.diff(x)
            return np.concatenate([np.cumsum(inc[::-1])[::-1], [0.]])
        inner = revcum(opz/E**3, z)
        outer = revcum(E/opz * inner, z)
        return z, 5.*Om0/2.*outer
    _G_a_table = np.zeros((len(_omega_m_grid), len(_z_table))) #(n_omega_m, _z_table)
    for i, Om0 in enumerate(_Om0_grid):
        zz, GG = _G_a_of_z(Om0)
        _G_a_table[i] = np.interp(_z_table, zz, GG)

        
    _norm_grid = np.array([
    func_axionHMcode_D_z_unnorm(
        0., Om0,
        lambda z, c=FlatLambdaCDM(H0=h_fid*100., Om0=Om0): c.H(z).value/(h_fid*100.))
    for Om0 in _Om0_grid])

    
    _D_table = np.zeros((len(_omega_m_grid), len(_z_table)))
    for i, Om0 in enumerate(_Om0_grid):
        cosmo_i = FlatLambdaCDM(H0=h_fid*100., Om0=Om0)
        E_z_i   = lambda z, c=cosmo_i: c.H(z).value / (h_fid * 100.)
        norm_i  = _norm_grid[i]
        _D_table[i] = [func_axionHMcode_D_z_unnorm(z, Om0, E_z_i) / norm_i
                       for z in _z_table]

        
    _DA_table = np.zeros((len(_omega_m_grid), len(Z_ARR)))
    for i, Om0 in enumerate(_Om0_grid):
        for j, z in enumerate(Z_ARR):
            _DA_table[i, j] = comoving_distance(
                {"cosmos": {"z": z, "h": h_fid,
                             "Omega_m_0": Om0, "Omega_w_0": 1.-Om0}}, z) / (1+z)

            
    _Sigma_crit_table = np.zeros((len(_omega_m_grid), len(Z_ARR)))
    for i, Om0 in enumerate(_Om0_grid):
        cosmo_i = FlatLambdaCDM(H0=h_fid*100., Om0=Om0)
        for j, z in enumerate(Z_ARR):
            usable = _z_source_bins[_z_source_bins > z]
            if len(usable) == 0:
                _Sigma_crit_table[i, j] = np.inf
                continue
            inv_scs = []
            DA_l = cosmo_i.angular_diameter_distance(z).value  # Mpc, astropy ← fix
            for z_s in usable:
                DA_s  = cosmo_i.angular_diameter_distance(z_s).value
                DA_ls = cosmo_i.angular_diameter_distance_z1z2(z, z_s).value
                if DA_ls <= 0:
                    continue
                inv_sc = (4.*np.pi*_G_SI/_c_SI**2
                          * DA_l*_Mpc_m * DA_ls*_Mpc_m / (DA_s*_Mpc_m))
                inv_scs.append(inv_sc * _Msol_kg / _Mpc_m**2)
            _Sigma_crit_table[i, j] = 1./np.mean(inv_scs) if inv_scs else np.inf    
    with open(_cache_path, "wb") as f:
        pickle.dump((_G_a_table, _D_table, _norm_grid, _DA_table, _Sigma_crit_table), f)
    print("Saved tables")


def _interp_omega_m(omega_m, table):
    """Linear interpolation over the omega_m grid."""
    w    = np.interp(omega_m/h_fid**2, _Om0_grid, np.arange(len(_Om0_grid), dtype=float))
    i_lo = int(np.clip(np.floor(w), 0, len(_Om0_grid)-2))
    frac = w - i_lo
    return (1.-frac)*table[i_lo] + frac*table[i_lo+1]

def get_G_a_interp(omega_m):        return _interp_omega_m(omega_m, _G_a_table)   # now returns an array over Z_ARR
def get_norm_interp(omega_m):       return np.interp(omega_m, _omega_m_grid, _norm_grid)
def get_D_grid_interp(omega_m):     return _z_table.copy(), _interp_omega_m(omega_m, _D_table)
def get_DA_grid_interp(omega_m):    return Z_ARR.copy(),    _interp_omega_m(omega_m, _DA_table)
def get_Sigma_crit_arr(omega_m):    return _interp_omega_m(omega_m, _Sigma_crit_table)

#################################################################################### PROFILE CACHE
import os
import pickle
import numpy as np
from tqdm import tqdm
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d

from feedback_mimic import (
    get_NFW_profile, get_realspace_profile, get_some_quantities,
    dn_dlog10Mtot, Mtot_to_MDelta, get_tangential_shear
)

CACHE_PATH = f"/home/user/FeedbackMimic/results/mcmc/MCMC_SHEAR_LCDM_CACHE{cache_suffix}.npz"
if os.path.exists(CACHE_PATH):
    print("Shear cache already made")
else:
    HALO_INPUT_FILE = "/home/user/FeedbackMimic/results/halo_model"
    
    # ── grids ─────────────────────────────────────────────────────────────────────
    h_fid         = 0.68
    omega_m_0_fid = 0.143344
    
    Z_ARR = np.linspace(0.0, 2.0, 81)
    Z_ARR = Z_ARR[Z_ARR > 0]
    
    K_ARR = np.logspace(-6, 6, 8192)
    R_ARR = np.logspace(-4, np.log10(1 / K_ARR.min()), 1500)
    
    M_TRANSITION     = 10**12.5
    M_DELTA          = 200
    M_REF_DENS       = "critical"
    R_PROJ_ARR_NUM   = 300
    R_FOR_SIGMA_NUM  = 100
    
    # ── KEY FIX: single common R grid for ALL redshifts ──────────────────────────
    # Use the smallest R_max across all z (= 3.2/(1+z_max)) so every z slice
    # covers the full grid. Profiles beyond a given z's r_vir will naturally
    # drop to zero; we never extrapolate outward.
    R_MAX_GLOBAL  = 3.2 / (1. + Z_ARR.min())   # ~1.6/(1+2) ≈ 0.533 Mpc/h for z_max=2
    R_proj_common = np.linspace(0.5, R_MAX_GLOBAL, R_PROJ_ARR_NUM)   # (nR,) fixed
    
    # ── Sigma_crit ────────────────────────────────────────────────────────────────
    _G_SI    = 6.674e-11
    _c_SI    = 2.998e8
    _Mpc_m   = 3.0857e22
    _Msol_kg = 1.989e30

    def compute_Sigma_crit_arr(omega_m, h_fid=h_fid,
                                _z_source_bins=np.array([0.285, 0.476, 0.743, 0.942])):
        Om0   = omega_m / h_fid**2
        cosmo = FlatLambdaCDM(H0=h_fid * 100.0, Om0=Om0)
        Sigma_crit_arr = np.zeros(len(Z_ARR))
        for j, z_l in enumerate(Z_ARR):
            usable = _z_source_bins[_z_source_bins > z_l]
            if len(usable) == 0:
                Sigma_crit_arr[j] = np.inf
                continue
            DA_l    = cosmo.angular_diameter_distance(z_l).value
            inv_scs = []
            for z_s in usable:
                DA_s  = cosmo.angular_diameter_distance(z_s).value
                DA_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
                if DA_ls <= 0:
                    continue
                inv_sc = (4.*np.pi*_G_SI/_c_SI**2
                          * DA_l*_Mpc_m * DA_ls*_Mpc_m / (DA_s*_Mpc_m))
                inv_scs.append(inv_sc * _Msol_kg / _Mpc_m**2)
            Sigma_crit_arr[j] = 1./np.mean(inv_scs) if inv_scs else np.inf
        return Sigma_crit_arr
    
    # Sigma_crit in Msol/h/(Mpc/h)^2
    Sigma_crit_arr = compute_Sigma_crit_arr(omega_m_0_fid) / h_fid
    
    # ── mass grid ─────────────────────────────────────────────────────────────────
    lnM_min    = np.log(10**13.5)
    lnM_max    = np.log(1e16)
    N_M        = 600
    lnM_common = np.linspace(lnM_min, lnM_max, N_M)   # log(M200c [Msol/h])
    M_common   = np.exp(lnM_common)
    
    # ── allocate output arrays ────────────────────────────────────────────────────
    nZ = len(Z_ARR)
    nM = N_M
    nR = R_PROJ_ARR_NUM
    
    GT_cache = np.zeros((nZ, nM, nR))
    DS_cache = np.zeros((nZ, nM, nR))
    S_cache  = np.zeros((nZ, nM, nR))
    
    # ── main loop ─────────────────────────────────────────────────────────────────
    print("Computing shear cache...")
    for iz, z in enumerate(tqdm(Z_ARR)):
    
        with open(f"{HALO_INPUT_FILE}/{z}_data_LCDM_full2h_preferred.pkl", "rb") as f:
            data = pickle.load(f)
    
        Mtot, _ = dn_dlog10Mtot(data, 'LCDM')
        r_t, _, _, _, r_s, _, _, _ = get_some_quantities(data, axion_dic=None)
    
        valid  = Mtot > M_TRANSITION
        Mtot_v = Mtot[valid]
        r_t_v  = r_t[valid]
        r_s_v  = r_s[valid]
    
        rho_v = get_realspace_profile(
            u_k=get_NFW_profile(r_v=r_t_v, k_arr=K_ARR, r_s=r_s_v, r_t=r_t_v),
            r_arr=R_ARR, k_arr=K_ARR, M_arr=Mtot_v, r_t=r_t_v)
    
        M200c, _ = Mtot_to_MDelta(
            data, r_t=r_t_v, M_arr=Mtot_v, r_arr=R_ARR,
            Del=M_DELTA, reference_density=M_REF_DENS,
            axion_dic=None, rho=rho_v, k_arr=K_ARR)
    
        # ── KEY FIX: pass R_proj_common so every z uses the same R grid ──────────
        GT_raw, DS_raw, S_raw, R_proj_out = get_tangential_shear(
            rho_v, R_ARR, z, Mtot_v, M200c, r_t_v,
            Sigma_crit_arr[iz],
            R_min=R_proj_common[0],
            R_max=R_proj_common[-1],          # same R_max for every z
            R_proj_arr_num=R_PROJ_ARR_NUM,
            R_for_Sigma_num=R_FOR_SIGMA_NUM,
            M_min_to_consider=M_TRANSITION,
            full_output=True)
    
        # verify R_proj_out matches (it should by construction)
        assert np.allclose(R_proj_out, R_proj_common, rtol=1e-6), \
            f"R grid mismatch at z={z}"
    
        # filter valid entries and sort by M200c
        valid2    = M200c > 0
        lnM_nat   = np.log(M200c[valid2])
        GT_valid  = np.array([GT_raw[i] for i in range(len(M200c))
                              if valid2[i] and GT_raw[i] is not None])
        DS_valid  = np.array([DS_raw[i] for i in range(len(M200c))
                              if valid2[i] and DS_raw[i] is not None])
    
        # Sigma is on R_for_Sigma (log grid); resample onto R_proj_common
        R_for_Sigma = np.logspace(np.log10(1e-4), np.log10(R_proj_common[-1]),
                                  R_FOR_SIGMA_NUM)
        S_valid = np.array([
            np.interp(R_proj_common, R_for_Sigma, np.array(S_raw[i]))
            for i in range(len(M200c))
            if valid2[i] and S_raw[i] is not None])
    
        lnM_valid = lnM_nat[[GT_raw[i] is not None
                              for i in range(len(lnM_nat))]]
    
        order     = np.argsort(lnM_valid)
        lnM_valid = lnM_valid[order]
        GT_valid  = GT_valid[order]
        DS_valid  = DS_valid[order]
        S_valid   = S_valid[order]
    
        # interpolate onto common lnM grid for each R bin
        for ir in range(nR):
            GT_cache[iz, :, ir] = np.interp(lnM_common, lnM_valid, GT_valid[:, ir],
                                             left=0., right=0.)
            DS_cache[iz, :, ir] = np.interp(lnM_common, lnM_valid, DS_valid[:, ir],
                                             left=0., right=0.)
            S_cache[iz,  :, ir] = np.interp(lnM_common, lnM_valid, S_valid[:, ir],
                                             left=0., right=0.)
    
    np.savez(CACHE_PATH,
             GT_cache   = GT_cache,       # (nZ, nM, nR)
             DS_cache   = DS_cache,       # (nZ, nM, nR)
             S_cache    = S_cache,        # (nZ, nM, nR)
             R_proj     = R_proj_common,  # (nR,)  SINGLE common grid — key fix
             Z_ARR      = Z_ARR,          # (nZ,)
             lnM_common = lnM_common)     # (nM,)  log(M200c Msol/h)
    
    print(f"Saved to {CACHE_PATH}")
    print(f"Grid: nZ={nZ}, nM={nM}, nR={nR}")
    print(f"R grid: {R_proj_common[0]:.3f} to {R_proj_common[-1]:.3f} Mpc/h (FIXED for all z)")
    print(f"lnM:    {lnM_min:.2f} to {lnM_max:.2f}  "
          f"({np.exp(lnM_min):.2e} to {np.exp(lnM_max):.2e} Msol/h)")

def get_log_likelihood(
    catalogue_name,
    # ── free parameters ──
    omega_m  = omega_m_fid,
    A_s    = A_s_fid,
    A_sz     = -4.352890598769685,
    alpha_sz = 1.1991734303623067,
    beta_sz  = 1.9996943667059135,
    b_WL_0    = 0.0, 
    b_WL_z    = 0.0,
    b_WL_M    = 1.0,
    # ── fixed ────────────
    bias_sz       = 0.8,
    sigma_lnq     = 0.173,
    dof           = 0.0,
    q_cutoff      = 0.4,
    obs_select_min = 5.0,
    obs_select_max = 300.0,
    M_WL_pivot    = 2e14, # Msol/h
    z_WL_pivot    = 0.4,
    return_nc   = False,
    observables = [["q_so_sim"],["gt"]],
    pad_abundance = False,
    n_points = 16384,
    n_mass_points_coarse = 300,
    M_min = 10**12.5 / h_fid,
    M_max = 1e16 / h_fid,
    z_min = None,
    z_max = None,
    n_z = None,
    data_lik_from_abundance = False,
    n_points_data_lik = 320,
    bc_chunk_size = 2000
):
    z_table, D_grid = get_D_grid_interp(omega_m)
    nc = cosmocnc.cluster_number_counts()
    nc.cnc_params.update({
        "survey_sr"                : _SURVEY_SR,
        "survey_cat"               : _SURVEY_CAT,
        "obs_select"               : "q_so_sim",
        "observables"              : observables,
        "obs_select_min"           : obs_select_min,
        "obs_select_max"           : obs_select_max,
        "z_min"                    : Z_ARR[0] if z_min is None else z_min,
        "z_max"                    : Z_ARR[-1] if z_max is None else z_max,
        "M_min"                    : M_min,
        "M_max"                    : M_max,
        "cluster_catalogue"        : catalogue_name,
        "likelihood_type"          : "unbinned",
        "data_lik_from_abundance"  : data_lik_from_abundance,
        "scalrel_type_deriv"       : "analytical",
        "n_points"                 : n_points,
        "n_z"                      : len(Z_ARR) if n_z is None else n_z,
        "cosmology_tool"           : "classy_sz_jax",
        "power_spectrum_type"      : "cosmopower",
        "cosmo_amplitude_parameter": "A_s",
        "hmf_calc"                 : "cnc",
        "hmf_type"                 : "ST_axionHMcode",
        "mass_definition"          : "500c",
        "number_cores_hmf"         : 1,
        "number_cores_abundance"   : 1,
        "number_cores_data"        : 1,
        "cosmocnc_verbose"         : "none",
        "return_profile_params"    : True,
        "stacked_likelihood"       : False,
        "non_validated_clusters"   : False,
        "compute_abundance_matrix" : True,
        "cosmo_model"              : 'mnu',
        "apply_obs_cutoff"         : {str(["q_so_sim"]): True},
        "pad_abundance"            : pad_abundance,
        "n_points_data_lik"        : n_points_data_lik,
        "bc_chunk_size"            : bc_chunk_size
    })
    nc.cosmo_params.update({
        "Om0"     : omega_m / h_fid**2,
        "Ob0"     : omega_b_fid / h_fid**2,
        "h"       : h_fid,
        "H0"      : h_fid * 100.,
        "A_s"     : A_s,
        "n_s"     : ns_fid,
        "m_nu"    : 0.0,
        "Onu0": 0.0,
        "tau_reio": 0.06,
        "n_mass_points_coarse": n_mass_points_coarse,
        "Pk_scaling_for_sigmaM" : 1.,
        'm_ncdm': 0.0,    
        'N_ncdm': 0,      
        'N_ur': 3.046,
        # ── These are necessary to make ST_axionHMcode fast ──
        "G_a_precomputed":           get_G_a_interp(omega_m),
        "D_grid_z_precomputed":      z_table,
        "D_grid_precomputed":        D_grid,
        "normalisation_precomputed": get_norm_interp(omega_m),
        "Sigma_crit_vec": get_Sigma_crit_arr(omega_m)
    })
    nc.scal_rel_params.update({
        "A_szifi"        : A_sz,
        "alpha_szifi"    : alpha_sz,
        "beta_szifi"     : beta_sz,
        "bias_sz"        : bias_sz,
        "sigma_lnq_szifi": sigma_lnq,
        "dof"            : dof,
        "q_cutoff"       : q_cutoff,
        "b_WL_0":      b_WL_0,
        "b_WL_M":      b_WL_M,
        "M_WL_pivot":  M_WL_pivot / h_fid,   # in Msol
        "h_fid":      h_fid,        # ← required by get_profile_sr_params
        "b_WL_z":      b_WL_z,
        "z_WL_pivot":  z_WL_pivot,
    })
    nc.initialise()

    try:
        nc.get_hmf(return_profile_params=True)
        log_lik_sz_cosmocnc = nc.get_log_lik()
    except MemoryError:
        raise                      # never retry an OOM
    except Exception as e:
        print(f"cosmocnc_jax failed: {e}")
        return -np.inf

    if return_nc:
        return float(log_lik_sz_cosmocnc), nc
    return float(log_lik_sz_cosmocnc)