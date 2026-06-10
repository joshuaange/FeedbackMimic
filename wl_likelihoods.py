"""
wl_likelihoods.py  — weak-lensing likelihood kernels

Contains:
  - Analytic NFW kernel (gt_nfw_from_profile, logP_gt_all)
  - Numerical cache kernel (build_wl_kernel_numerical, logP_gt_numerical)

Key fix in numerical kernel: R_proj is a SINGLE static array loaded from the
cache, not per-z. This is required because jnp.interp needs a static xp grid.
"""

import numpy as np
import jax
import jax.numpy as jnp

# ─────────────────────────────────────────────────────────────────────────────
# ANALYTIC NFW KERNEL
# ─────────────────────────────────────────────────────────────────────────────

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

@jax.jit
def gt_nfw_from_profile(r_bins, M, rho_crit_z, c_200c, Sigma_crit):
    """
    Analytic NFW reduced tangential shear.

    Args:
        r_bins:     (nR,) projected radii          [Mpc/h]
        M:          scalar M_200c                  [Msol/h]
        rho_crit_z: scalar rho_crit(z)             [Msol/h/(Mpc/h)^3]
        c_200c:     scalar concentration           [dimensionless]
        Sigma_crit: scalar critical surface dens   [Msol/h/(Mpc/h)^2]
    """
    r_200      = (3.0 * M / (4.0 * jnp.pi * 200.0 * rho_crit_z))**(1.0/3.0)
    r_s        = r_200 / c_200c
    g_c        = jnp.log(1.0 + c_200c) - c_200c / (1.0 + c_200c)
    rho_s      = M / (4.0 * jnp.pi * r_s**3 * g_c)
    x          = r_bins / r_s
    Sigma      = 2.0 * rho_s * r_s / (x**2 - 1.0) * (1.0 - _f_nfw(x))
    Sigma_bar  = 4.0 * rho_s * r_s / x**2 * _g_nfw(x)
    DeltaSigma = Sigma_bar - Sigma
    return DeltaSigma / (Sigma_crit - Sigma)

def build_wl_kernel():
    def wl_kernel(gt_obs, sigma_gt, r_bins,
                  M_arr, rho_crit_z, c_200c_arr, Sigma_crit,
                  b_WL_0, b_WL_M, M_pivot):
        lnM_pivot = jnp.log(M_pivot)
        def _one_mass(M):
            ln_MWL    = b_WL_0 + b_WL_M*jnp.log(M) - b_WL_M*lnM_pivot + lnM_pivot
            M_wl      = jnp.exp(ln_MWL)
            c_200c_WL = jnp.interp(jnp.log(M_wl), jnp.log(M_arr), c_200c_arr)
            gt_pred   = gt_nfw_from_profile(r_bins, M_wl, rho_crit_z, c_200c_WL, Sigma_crit)
            chi2      = jnp.sum(((gt_obs - gt_pred) / sigma_gt)**2)
            return -0.5 * chi2
        logP  = jax.vmap(_one_mass)(M_arr)
        logP += (-jnp.sum(jnp.log(sigma_gt))
                 -0.5 * gt_obs.size * jnp.log(2.0 * jnp.pi))
        return logP
    return wl_kernel

_wl_kernel = build_wl_kernel()

logP_gt_all = jax.jit(
    jax.vmap(_wl_kernel,
             in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None)))


# ─────────────────────────────────────────────────────────────────────────────
# NUMERICAL CACHE KERNEL
# ─────────────────────────────────────────────────────────────────────────────

_CACHE_PATH = "/home/jange/scratch/FeedbackMimic/results/mcmc/MCMC_SHEAR_LCDM_CACHE.npz"
_cache      = np.load(_CACHE_PATH)

# ── KEY FIX: R_proj is a SINGLE static 1-D array, same for every z ───────────
GT_GRID  = jnp.array(_cache['GT_cache'])    # (nZ, nM, nR)
DS_GRID  = jnp.array(_cache['DS_cache'])    # (nZ, nM, nR)
S_GRID   = jnp.array(_cache['S_cache'])     # (nZ, nM, nR)
Z_GRID   = jnp.array(_cache['Z_ARR'])       # (nZ,)
LNM_GRID = jnp.array(_cache['lnM_common'])  # (nM,)  log(M200c Msol/h)
R_COMMON = jnp.array(_cache['R_proj'])      # (nR,)  STATIC — same for all z


def build_wl_kernel_numerical():
    """
    Drop-in replacement for build_wl_kernel() that interpolates gt from the
    precomputed numerical cache instead of using analytic NFW formulas.

    The kernel signature matches the analytic version (plus z_cluster) so the
    call site only needs to add z_wl as the last per-cluster argument.

    KEY: R_COMMON is a module-level constant so jnp.interp always sees a
    static xp array — no per-z R grid that would confuse JAX tracing.
    """

    def _gt_from_cache(r_bins, lnM_wl, iz, iz_frac):
        """
        Bilinear interp in (z, lnM), then interp in R.

        iz, iz_frac: pre-computed integer index and fractional weight for z.
        lnM_wl:     log(M_wl) in Msol/h.
        r_bins:     query projected radii in Mpc/h (must lie within R_COMMON).
        """
        iM     = jnp.clip(jnp.searchsorted(LNM_GRID, lnM_wl) - 1,
                          0, LNM_GRID.shape[0] - 2)
        tM     = (lnM_wl - LNM_GRID[iM]) / (LNM_GRID[iM+1] - LNM_GRID[iM])

        def bilinear_R(grid):
            # (nR,) profile at the four surrounding (z, M) corners
            v00 = grid[iz,   iM  ]
            v01 = grid[iz,   iM+1]
            v10 = grid[iz+1, iM  ]
            v11 = grid[iz+1, iM+1]
            return (v00*(1-iz_frac)*(1-tM) + v01*(1-iz_frac)*tM
                  + v10*iz_frac*(1-tM)     + v11*iz_frac*tM)

        gt_profile = bilinear_R(GT_GRID)   # (nR,) on R_COMMON
        # R_COMMON is static → jnp.interp is safe inside jit
        return jnp.interp(r_bins, R_COMMON, gt_profile, left=0., right=0.)

    def wl_kernel(gt_obs, sigma_gt, r_bins,
                  M_arr, rho_crit_z, c_200c_arr, Sigma_crit,
                  b_WL_0, b_WL_M, M_pivot,
                  z_cl):
        lnM_pivot = jnp.log(M_pivot)

        # Pre-compute z interpolation weights once per cluster (not per mass)
        iz     = jnp.clip(jnp.searchsorted(Z_GRID, z_cl) - 1,
                          0, Z_GRID.shape[0] - 2)
        iz_frac = (z_cl - Z_GRID[iz]) / (Z_GRID[iz+1] - Z_GRID[iz])

        def _one_mass(M):
            ln_MWL = b_WL_0 + b_WL_M*jnp.log(M) - b_WL_M*lnM_pivot + lnM_pivot
            gt_pred = _gt_from_cache(r_bins, ln_MWL, iz, iz_frac)
            chi2    = jnp.sum(((gt_obs - gt_pred) / sigma_gt)**2)
            return -0.5 * chi2

        logP  = jax.vmap(_one_mass)(M_arr)
        logP += (-jnp.sum(jnp.log(sigma_gt))
                 -0.5 * gt_obs.size * jnp.log(2.0 * jnp.pi))
        return logP

    return wl_kernel

_wl_kernel_numerical = build_wl_kernel_numerical()

logP_gt_numerical = jax.jit(
    jax.vmap(
        _wl_kernel_numerical,
        in_axes=(
            0, 0, 0,          # gt_obs, sigma_gt, r_bins
            0, 0, 0, 0,       # M_arr, rho_crit_z, c_200c_arr, Sigma_crit
            None, None, None, # b_WL_0, b_WL_M, M_pivot
            0,                # z_cl — per cluster
        )
    )
)