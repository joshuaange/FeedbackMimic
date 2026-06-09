import jax
import jax.numpy as jnp

def _f_nfw_jax(x):
    safe_lt = jnp.sqrt(jnp.maximum(1. - x**2, 1e-30))
    safe_gt = jnp.sqrt(jnp.maximum(x**2 - 1., 1e-30))
    return jnp.where(
        x < 1., jax.lax.atanh(safe_lt) / safe_lt,
        jnp.where(x > 1., jnp.arctan(safe_gt) / safe_gt, 1.))


def _g_nfw_jax(x):
    safe_lt = jnp.sqrt(jnp.maximum(1. - x**2, 1e-30))
    safe_gt = jnp.sqrt(jnp.maximum(x**2 - 1., 1e-30))
    return jnp.where(
        x < 1., jnp.log(x / 2.) + jax.lax.atanh(safe_lt) / safe_lt,
        jnp.where(x > 1., jnp.log(x / 2.) + jnp.arctan(safe_gt) / safe_gt,
                  1. + jnp.log(0.5)))


@jax.jit
def gt_nfw_from_profile(r_bins_Mpc, r_s, delta_char, rho_m, Sigma_crit):
    """
    g_t(r) ≈ ΔΣ(r) / Σ_crit using precomputed NFW profile parameters.

    profile_params convention:
        rho(r) = delta_char * rho_m / (x (1+x)²),   x = r/r_s
    so rho_s = delta_char * rho_m.
    """
    rho_s = delta_char * rho_m
    x = r_bins_Mpc / r_s
    Sigma       = 2. * rho_s * r_s / (x**2 - 1.) * (1. - _f_nfw_jax(x))
    mean_Sigma  = 4. * rho_s * r_s / x**2 * _g_nfw_jax(x)
    return (mean_Sigma - Sigma) / Sigma_crit


@jax.jit
def log_lik_wl_cluster(gt_obs, sigma_gt, r_bins,
                        r_s_arr, delta_char_arr, rho_m, Sigma_crit,
                        b_WL_0, b_WL_M,
                        M_arr, M_pivot,
                        mass_pdf):           # P(M | q_obs, z) from bc_cpdf
    """
    Correct Bocquet-style WL term: marginalise over the SZ mass posterior.
    
    M_arr:      (n_M,) mass grid from bc_lnM_array
    mass_pdf:   (n_M,) normalised P(M | q, z) from bc_cpdf_array
    r_s_arr:    (n_M,) r_s interpolated at each mass in M_arr
    delta_char_arr: (n_M,) delta_char at each mass in M_arr
    """
    lnM_arr = jnp.log(M_arr)
    lnM_pivot_val = jnp.log(M_pivot)
    
    def _one_mass(r_s, delta_char, M):
        lnM_ratio = jnp.log(M) - lnM_pivot_val
        # Mean M_WL from b_WL relation
        lnMWL_ratio = b_WL_0 + b_WL_M * lnM_ratio
        r_s_wl = r_s * jnp.exp((lnMWL_ratio - lnM_ratio) / 3.)
        gt_pred = gt_nfw_from_profile(r_bins, r_s_wl, delta_char, rho_m, Sigma_crit)
        return -0.5 * jnp.sum(((gt_obs - gt_pred) / sigma_gt)**2)
    
    # Evaluate log P(g_t | M) at each mass grid point
    log_liks = jax.vmap(_one_mass)(r_s_arr, delta_char_arr, M_arr)
    
    # Marginalise: integral of P(g_t|M) * P(M|q,z) dM
    # = trapz(exp(log_liks) * mass_pdf, lnM_arr)
    log_liks_shifted = log_liks - jnp.max(log_liks)
    integral = jnp.trapezoid(jnp.exp(log_liks_shifted) * mass_pdf, lnM_arr)
    return jnp.log(jnp.maximum(integral, 1e-300)) + jnp.max(log_liks)


log_lik_wl_all = jax.jit(jax.vmap(
    log_lik_wl_cluster,
    in_axes=(0, 0, 0,        # gt_obs, sigma_gt, r_bins
             0, 0, 0, 0,     # r_s, delta_char, rho_m, Sigma_crit
             None, None,     # b_WL_0, b_WL_M,
             0, None, 0)     # lnM_halo, lnM_pivot, n_MWL
))