import numpy as np
import jax
import jax.numpy as jnp

def _f_nfw(x):
    x = jnp.asarray(x)

    def x_lt_1(x):
        return 1.0 / jnp.sqrt(1 - x**2) * jax.lax.atanh(jnp.sqrt(1 - x**2))

    def x_gt_1(x):
        return 1.0 / jnp.sqrt(x**2 - 1) * jnp.arctan(jnp.sqrt(x**2 - 1))

    return jnp.where(x < 1, x_lt_1(x),
           jnp.where(x > 1, x_gt_1(x), 1.0))

def _g_nfw(x):
    x = jnp.asarray(x)

    def x_lt_1(x):
        return jnp.log(x/2) + 1/jnp.sqrt(1 - x**2) * jax.lax.atanh(jnp.sqrt(1 - x**2))

    def x_gt_1(x):
        return jnp.log(x/2) + 1/jnp.sqrt(x**2 - 1) * jnp.arctan(jnp.sqrt(x**2 - 1))

    return jnp.where(x < 1, x_lt_1(x),
           jnp.where(x > 1, x_gt_1(x), 1 + jnp.log(0.5)))
    
def func_NFW_DeltaSigma(R, M, rho_crit, c_nfw):

    r_200 = (3.0 * M / (4.0 * jnp.pi * 200.0 * rho_crit))**(1.0 / 3.0)
    r_s = r_200 / c_nfw

    g_c = jnp.log(1 + c_nfw) - c_nfw / (1 + c_nfw)
    rho_s = M / (4.0 * jnp.pi * r_s**3 * g_c)

    x = R / r_s

    Sigma = 2.0 * rho_s * r_s / (x**2 - 1.0) * (1.0 - _f_nfw(x))
    Sigma_bar = 4.0 * rho_s * r_s / x**2 * _g_nfw(x)

    return Sigma_bar - Sigma


@jax.jit
def gt_nfw_from_profile(r_bins, M, rho_crit_z, c_200c, Sigma_crit):
    """
    Reduced tangential shear g_t = DeltaSigma / (Sigma_crit - Sigma).
    
    Args:
        r_bins:     projected radii, h^-1 Mpc
        M:          M_200c in Msol/h
        rho_crit_z: critical density at cluster redshift, Msol/(h^-1 Mpc)^3
        c_200c:     concentration c_200c (dimensionless)
        Sigma_crit: critical surface density, Msol/(h^-1 Mpc)^2
    """
    r_200   = (3.0 * M / (4.0 * jnp.pi * 200.0 * rho_crit_z))**(1.0 / 3.0)
    r_s     = r_200 / c_200c
    g_c     = jnp.log(1.0 + c_200c) - c_200c / (1.0 + c_200c)
    rho_s   = M / (4.0 * jnp.pi * r_s**3 * g_c)

    x         = r_bins / r_s
    
    Sigma     = 2.0 * rho_s * r_s / (x**2 - 1.0) * (1.0 - _f_nfw(x))
    Sigma_bar = 4.0 * rho_s * r_s / x**2 * _g_nfw(x)
    DeltaSigma = Sigma_bar - Sigma
    return DeltaSigma / (Sigma_crit - Sigma)

def build_wl_kernel():
    def wl_kernel(gt_obs, sigma_gt, r_bins,
                  M_arr, rho_crit_z, c_200c_arr, Sigma_crit,
                  b_WL_0, b_WL_M, M_pivot):
        lnM_pivot = jnp.log(M_pivot)

        def _one_mass(M):
            ln_MWL = b_WL_0 + b_WL_M*jnp.log(M) - b_WL_M*lnM_pivot + lnM_pivot
            M_wl  = jnp.exp(ln_MWL)
            c_200c_WL = jnp.interp(
                jnp.log(M_wl),
                jnp.log(M_arr),
                c_200c_arr
            )
            gt_pred = gt_nfw_from_profile(r_bins, M_wl, rho_crit_z, c_200c_WL, Sigma_crit)
            chi2 = jnp.sum(
                ((gt_obs - gt_pred) / sigma_gt) ** 2
            )
            return -0.5 * chi2
            
        logP = jax.vmap(_one_mass)(M_arr)
        logP += (
            -jnp.sum(jnp.log(sigma_gt))
            -0.5 * gt_obs.size * jnp.log(2.0 * jnp.pi)
        )
        return logP

    return wl_kernel

_wl_kernel = build_wl_kernel()

logP_gt_all = jax.jit(
    jax.vmap(
        _wl_kernel,
        in_axes=(
            0, 0, 0,
            0, 0, 0, 0,
            None, None,
            None
        )
    )
)

######################################## Numerical cache approach instead:

# ── Load shear cache ──────────────────────────────────────────────────────────
_CACHE_PATH = "/home/jange/scratch/FeedbackMimic/results/mcmc/MCMC_SHEAR_LCDM_CACHE.npz"
_cache      = np.load(_CACHE_PATH)

DS_GRID  = jnp.array(_cache['DS_cache'])    # (nZ, nM, nR)
S_GRID   = jnp.array(_cache['S_cache'])     # (nZ, nM, nR)
Z_GRID   = jnp.array(_cache['Z_ARR'])       # (nZ,)
LNM_GRID = jnp.array(_cache['lnM_common'])  # (nM,)  log(M200c Msol/h)
R_GRID   = jnp.array(_cache['R_proj'])      # (nR,)  Mpc/h

# ── Grid-based gt ─────────────────────────────────────────────────────────────
def _gt_from_grid(r_bins, lnM_wl, z_cl, Sigma_crit,
                  DS_grid, S_grid, Z_arr, lnM_arr, R_arr):
    """Bilinear interpolation in (z, lnM), then interp in R."""
    iz = jnp.clip(jnp.searchsorted(Z_arr,   z_cl)   - 1, 0, Z_arr.shape[0]   - 2)
    iM = jnp.clip(jnp.searchsorted(lnM_arr, lnM_wl) - 1, 0, lnM_arr.shape[0] - 2)
    tz = (z_cl   - Z_arr[iz])   / (Z_arr[iz+1]   - Z_arr[iz])
    tM = (lnM_wl - lnM_arr[iM]) / (lnM_arr[iM+1] - lnM_arr[iM])

    def bilinear(grid):
        return (grid[iz,   iM  ] * (1-tz) * (1-tM)
              + grid[iz,   iM+1] * (1-tz) * tM
              + grid[iz+1, iM  ] * tz     * (1-tM)
              + grid[iz+1, iM+1] * tz     * tM)

    DS = jnp.interp(r_bins, R_arr, bilinear(DS_grid))
    S  = jnp.interp(r_bins, R_arr, bilinear(S_grid))
    return DS / (Sigma_crit - S)


def build_wl_kernel_grid():
    def wl_kernel(gt_obs, sigma_gt, r_bins,
                  M_arr, rho_crit_z, c_200c_arr, Sigma_crit,
                  b_WL_0, b_WL_M, M_pivot,
                  z_cl):
        lnM_pivot = jnp.log(M_pivot)
        def _one_mass(M):
            ln_MWL = b_WL_0 + b_WL_M*jnp.log(M) - b_WL_M*lnM_pivot + lnM_pivot
            gt_pred = _gt_from_grid(r_bins, ln_MWL, z_cl, Sigma_crit,
                                    DS_GRID, S_GRID, Z_GRID, LNM_GRID, R_GRID)
            chi2 = jnp.sum(((gt_obs - gt_pred) / sigma_gt)**2)
            return -0.5 * chi2
        logP = jax.vmap(_one_mass)(M_arr)
        logP += (-jnp.sum(jnp.log(sigma_gt))
                 -0.5 * gt_obs.size * jnp.log(2.*jnp.pi))
        return logP
    return wl_kernel

_wl_kernel_grid = build_wl_kernel_grid()

logP_gt_all_grid = jax.jit(
    jax.vmap(
        _wl_kernel_grid,
        in_axes=(0, 0, 0,
                 0, 0, 0, 0,
                 None, None, None,
                 0)   # z_cl per cluster
    )
)