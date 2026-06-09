"""
likelihood.py  –  get_log_likelihood(omega_m, logAs, A_sz, alpha_sz, beta_sz)

Catalogue inputs (per halo):
    q_so_sim : SZ SNR, the selection observable
    z        : redshift (assumed precise)

Free parameters:
    omega_m  = Omega_m * h^2          (physical matter density)
    logAs    = ln(10^10 A_s)          (amplitude of scalar perturbations)
    A_sz     -> A_szifi               (SZ amplitude, log10 units)
    alpha_sz -> alpha_szifi           (SZ mass slope)
    beta_sz  -> beta_szifi            (SZ E(z) slope)

Fixed:
    h=0.68, omega_b=0.0226576, n_s=0.965
    bias_sz=0.8, sigma_lnq_szifi=0.173, dof=0, q_cutoff=0

Likelihood (Zubeldia & Bolliet 2024, eq. 3.18):

    ln L = -N_bar  +  sum_i ln P(q_obs,i, z_i | params)

    N_bar  = integral dM dz dOmega  dn/dM  P(q > q_th | M, z)
    P(D_i) = integral dM  dn/dM  P(q_obs,i | M, z_i)

    SZ scaling relation: two-layer Gaussian hierarchy (section 3.2.2):
        Layer 0:  ln q_mean(M, z)  ->  ln q   [intrinsic scatter sigma_lnq]
        Layer 1:  exp(ln q)        ->  q_obs  [unit Gaussian noise]
"""

import numpy as np
import os
import cosmocnc_axionHMcode_jax.cosmocnc_jax as cosmocnc

h_fid       = 0.68
omega_b_fid = 0.0226576
ns_fid      = 0.965
omega_m_fid = 0.143344
logAs_fid   = np.log(2.1e-09 * 1e10)

_SURVEY_SR  = "/home/jange/scratch/FeedbackMimic/cosmocnc_axionHMcode_jax/cosmocnc_jax/surveys/survey_sr_so_sim.py"
_SURVEY_CAT = "/home/jange/scratch/FeedbackMimic/cosmocnc_axionHMcode_jax/cosmocnc_jax/surveys/survey_cat_so_sim.py"

# one nc object per catalogue, initialised lazily
_nc_cache = {}

def _get_nc(catalogue_name):
    if catalogue_name not in _nc_cache:
        nc = cosmocnc.cluster_number_counts()
        nc.cnc_params.update({
            "survey_sr"                : _SURVEY_SR,
            "survey_cat"               : _SURVEY_CAT,
            "obs_select"               : "q_so_sim",
            "observables"              : [["q_so_sim"]],
            "obs_select_min"           : 5.0,
            "obs_select_max"           : 200.0,
            "z_min"                    : 0.05,
            "z_max"                    : 1.5,
            "M_min"                    : 1e13 / h_fid,
            "M_max"                    : 1e16 / h_fid,
            "cluster_catalogue"        : catalogue_name,
            "likelihood_type"          : "unbinned",
            "data_lik_from_abundance"  : False,
            "scalrel_type_deriv"       : "analytical",
            "n_points"                 : 4096,
            "n_points_data_lik"        : 128,
            "n_z"                      : 50,
            "cosmology_tool"           : "classy_sz_jax",
            "power_spectrum_type"      : "cosmopower",
            "cosmo_amplitude_parameter": "A_s",
            "hmf_calc"                 : "cnc",
            "hmf_type"                 : "Tinker08",
            "mass_definition"          : "500c",
            "number_cores_hmf"         : 1,
            "number_cores_abundance"   : 1,
            "number_cores_data"        : 1,
            "cosmocnc_verbose"         : "none",
            "include_wl_bias"          : False,
        })
        nc.initialise()
        _nc_cache[catalogue_name] = nc
    return _nc_cache[catalogue_name]


def get_log_likelihood(
    catalogue_name,
    # ── free parameters ──
    omega_m  = omega_m_fid,
    logAs    = logAs_fid,
    A_sz     = -4.3054,
    alpha_sz = 1.1233,
    beta_sz  = 2.0,
    # ── fixed ────────────
    bias_sz       = 0.8,
    sigma_lnq     = 0.173,
    dof           = 0.0,
    q_cutoff      = 0.0,
):
    """
    Returns  ln L = -N_bar + sum_i ln P(q_obs,i, z_i | params).
    First call per catalogue_name runs initialise() (slow).
    Subsequent calls only run reinitialise() (fast).
    """
    nc = _get_nc(catalogue_name)

    nc.cosmo_params.update({
        "Om0"     : omega_m / h_fid**2,
        "Ob0"     : omega_b_fid / h_fid**2,
        "h"       : h_fid,
        "A_s"     : np.exp(logAs) * 1e-10,
        "n_s"     : ns_fid,
        "m_nu"    : 0.06,
        "tau_reio": 0.0544,
    })
    nc.scal_rel_params.update({
        "A_szifi"        : A_sz,
        "alpha_szifi"    : alpha_sz,
        "beta_szifi"     : beta_sz,
        "bias_sz"        : bias_sz,
        "sigma_lnq_szifi": sigma_lnq,
        "dof"            : dof,
        "q_cutoff"       : q_cutoff,
    })

    try:
        nc.reinitialise()
        return nc.get_log_lik()
    except Exception as e:
        print(f"cosmocnc_jax failed: {e}")
        return -np.inf