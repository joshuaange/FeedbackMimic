from axionHMcode.halo_model import PS_nonlin_cold
import numpy as np
from tqdm import tqdm
from scipy import special
from scipy.integrate import simpson, trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import brentq, root_scalar
from mcfit import SphericalBessel
from axionHMcode.halo_model.axion_density_profile import func_dens_profile_ax
from axionHMcode.halo_model import PS_nonlin_cold
from axionHMcode.cosmology.overdensities import func_r_vir, func_Delta_vir
from axionHMcode.cosmology.basic_cosmology import func_H_z as H_of_z
from axionHMcode.cosmology.basic_cosmology import func_rho_comp_0
from axionHMcode.halo_model.cold_density_profile import func_delta_char, func_conc_param


def smooth_kmax_filter(k_arr, k_max, width=0.5):
    return 1.0 / (1.0 + np.exp(-(np.log10(k_arr) - np.log10(k_max)) / width))


def get_some_quantities(data, M_arr=None, axion_dic=None):
    if M_arr is None:
        try:
            M_arr = data['M_CDM']
        except:
            M_arr = data['M']
    r_v = np.atleast_1d(func_r_vir(data['cosmos']['z'], M_arr, data['cosmos']['Omega_ax_0'], data['cosmos']['Omega_m_0'],
                                    data['cosmos']['Omega_m_0'], data['cosmos']['Omega_w_0'],
                                    data['cosmos']['G_a'], data['cosmos']['version']))
    rho_crit = 3. * H_of_z(data['cosmos']['z'],
                            data['cosmos']['h'],
                            data['cosmos']['Omega_m_0'],
                            data['cosmos']['Omega_w_0'])**2 \
               / (8. * np.pi * 4.300917270069977e-09)
    rho_char = func_delta_char(M_arr, data['power_spec_dic']['k'], data['power_spec_dic']['power_cold'],
                               data['cosmos'], data['hmcode_params'], data['cosmos']['Omega_db_0'], data['c'], eta_given=False, axion_dic=axion_dic)
    concentration = np.atleast_1d(func_conc_param(M_arr, data['power_spec_dic']['k'], data['power_spec_dic']['power_cold'],
                                                   data['cosmos'], data['cosmos']['Omega_m_0'], data['c'], axion_dic=axion_dic))
    r_s = r_v / concentration
    Delta_vir = func_Delta_vir(data['cosmos']['z'], data['cosmos']['Omega_ax_0'], data['cosmos']['Omega_m_0'],
                               data['cosmos']['Omega_w_0'], data['cosmos']['G_a'], data['cosmos']['version'])
    rho_0 = func_rho_comp_0(data['cosmos']['Omega_m_0'])
    rho_0_wh = rho_0 * data['cosmos']['h']**2
    return r_v, rho_crit, rho_char, concentration, r_s, Delta_vir, rho_0, rho_0_wh


def get_enclosed_mass(rho, r_t, r_arr, just_one_M=False):
    if just_one_M:
        rt = r_t
        mask = r_arr < rt
        r_int = r_arr[mask]
        rho_int = rho[mask]
        rho_rt = np.interp(rt, r_arr, rho)
        r_int = np.append(r_int, rt)
        rho_int = np.append(rho_int, rho_rt)
        return trapezoid(4 * np.pi * r_int**2 * rho_int, x=r_int)

    rho = np.asarray(rho)
    r_arr = np.asarray(r_arr)
    r_t = np.asarray(r_t)
    M = np.empty(np.size(r_t))
    for i in range(np.size(r_t)):
        rt = r_t[i]
        mask = r_arr < rt
        r_int = np.append(r_arr[mask], rt)
        rho_int = np.append(rho[i, mask], np.interp(rt, r_arr, rho[i]))
        M[i] = trapezoid(4 * np.pi * r_int**2 * rho_int, x=r_int)
    return M


def get_NFW_profile(r_t, r_v, k_arr, r_s, just_one_M=False):
    def f(x):
        return -x / (1 + x) + np.log(1 + x)

    if just_one_M:
        b = k_arr * r_s
        Si_bkrt, Ci_bkrt = special.sici(b + k_arr * r_t)
        Si_b,    Ci_b    = special.sici(b)
        return (np.cos(b) * (Ci_bkrt - Ci_b)
                + np.sin(b) * (Si_bkrt - Si_b)
                - np.sin(k_arr * r_t) / (b + k_arr * r_t)) / f(r_v / r_s)

    b = np.einsum('k,M->Mk', k_arr, r_s)
    Si_bkrt, Ci_bkrt = special.sici(b + np.einsum('k,M->Mk', k_arr, r_t))
    Si_b,    Ci_b    = special.sici(b)
    return (np.cos(b) * (Ci_bkrt - Ci_b)
            + np.sin(b) * (Si_bkrt - Si_b)
            - np.sin(np.einsum('k,M->Mk', k_arr, r_t))
              / (b + np.einsum('k,M->Mk', k_arr, r_t))) \
           * (1. / f(r_v / r_s))[:, None]


def inverse_spherical_ft(u_k, r_arr, k_arr, fudge_factor=0.7939074305962273):
    transform = SphericalBessel(k_arr, nu=0, q=1., lowring=True)
    r_fftlog, rho_fft = transform(u_k, extrap=False)
    rho_interp = interp1d(r_fftlog, rho_fft, bounds_error=False, fill_value=0)(r_arr)
    rho_interp /= (2 * np.pi**2)
    rho_interp /= fudge_factor
    return rho_interp


def get_realspace_profile(u_k, r_t, r_arr, k_arr, M_arr, just_one_M=False):
    if just_one_M:
        return M_arr * inverse_spherical_ft(u_k, r_arr, k_arr)
    rho = np.zeros((np.size(M_arr), np.size(r_arr)))
    for i, M_val in enumerate(M_arr):
        rho[i] = M_arr[i] * inverse_spherical_ft(u_k[i], r_arr, k_arr)
        rho[i][r_arr > r_t[i]] = 0.0
    return rho


def get_spectra(data_LCDM, utilde=None, full_2h=True, full_output=False,
                input_squared_NFW_FT=False, alpha=False, eta_given=False,
                one_halo_damping=True, two_halo_damping=False, concentration_param=True):
    result = PS_nonlin_cold.func_non_lin_PS_matter(
        data_LCDM['M'], data_LCDM['power_spec_dic']['k'], data_LCDM['power_spec_dic']['power_total'],
        data_LCDM['cosmos'], data_LCDM['hmcode_params'], data_LCDM['cosmos']['Omega_m_0'],
        alpha=alpha, eta_given=eta_given,
        one_halo_damping=one_halo_damping, two_halo_damping=two_halo_damping,
        concentration_param=concentration_param, full_2h=full_2h,
        custom_NFW_FT=utilde, input_squared_NFW_FT=input_squared_NFW_FT,
        full_output=full_output,
    )
    if full_output:
        return result
    return result[0]


def get_Delta_naive(PFDM, PLCDM1h_tosubtract, PLCDM2h_tosubtract,
                    PLCDM1h, PLCDM2h, all_outputs=False):
    delta_highk = np.sqrt(((PFDM - PLCDM2h_tosubtract) / PLCDM1h).astype(complex)).real
    delta_lowk  = np.sqrt(((PFDM - PLCDM1h_tosubtract) / PLCDM2h).astype(complex)).real
    R = (PLCDM1h_tosubtract / (PLCDM1h_tosubtract + PLCDM2h_tosubtract))**0.5
    F = R * delta_highk + (1 - R) * delta_lowk
    if all_outputs:
        return F.real, delta_highk.real, delta_lowk.real
    return F.real


def get_Delta(low_k_value, high_k_value, k0_of_M, k_arr, M_arr, s=0.1):
    def blend_two(k_arr, delta_naive, oneminusuhatconst, k0, s):
        wU = 1. / (1 + np.exp((k_arr - k0) / s))
        wD = 1. / (1 + np.exp((k0 - k_arr) / s))
        return (delta_naive * wD + oneminusuhatconst * wU) / (wU + wD)

    Delta = np.ones((np.size(M_arr), np.size(k_arr)))
    for i, M in enumerate(M_arr):
        Delta[i] = blend_two(k_arr, high_k_value, low_k_value[i], k0_of_M[i], s)
    return Delta


def uniform_density_ft(mass_fraction, M_arr, r_t, k_arr):
    kr = np.einsum('k,M->Mk', k_arr, r_t)
    sinc_term = (np.sin(kr) - kr * np.cos(kr)) / kr**3
    return 3 * (mass_fraction[:, None] / M_arr[:, None]) * sinc_term


def solve_r_t(target_mass_per_halo, Delta_per_halo, r_v, r_s, M_arr, r_arr, k_arr,
              min_r, max_r, u_k_base=None, fcga=None):
    if fcga is None:
        fcga = np.zeros(np.size(M_arr))
    max_r_trunc = max(r_arr)
    r_t = r_v.copy()

    for i, M_val in tqdm(enumerate(M_arr)):
        if target_mass_per_halo[i] <= 0:
            continue
        Delta_i = Delta_per_halo[i] if np.ndim(Delta_per_halo) == 2 else Delta_per_halo
        base_i  = u_k_base[i] if u_k_base is not None else \
                  get_NFW_profile(r_v=r_v[i], k_arr=k_arr, r_s=r_s[i], just_one_M=True, r_t=max_r_trunc)

        def mass_residual(r_t_test):
            rho = get_realspace_profile(u_k=base_i * Delta_i, r_arr=r_arr, k_arr=k_arr,
                                        M_arr=M_arr[i], just_one_M=True, r_t=max_r_trunc)
            return target_mass_per_halo[i] - get_enclosed_mass(rho=rho, r_t=r_t_test,
                                                                r_arr=r_arr, just_one_M=True)

        f_min, f_max = mass_residual(min_r), mass_residual(max_r)
        if f_min * f_max > 0:
            r_t[i] = r_v[i]
            if fcga[i] >= 1.:
                print(f"Warning for log10(M_arr) = {np.log10(M_arr[i])}")
            continue
        r_t[i] = root_scalar(mass_residual, bracket=[min_r, max_r], method='brentq').root

    return r_t


def Mtot_to_MDelta(data, r_t, M_arr, k_arr, r_arr, Del=200, axion_dic=None, rho=None,
                   initial_r_index_to_try=5, reference_density="mean", bias=1.0):
    def f(x):
        return -x / (1 + x) + np.log(1 + x)

    r_v, rho_crit, rho_char, concentration, r_s, Delta_vir, rho_0, rho_0_wh = \
        get_some_quantities(data, axion_dic=axion_dic)

    rho_ref = (rho_0 if reference_density == "mean"
               else rho_crit / data['cosmos']['h']**2)

    if Del == "vir":
        Del = Delta_vir
        if reference_density == "mean":
            return M_arr, r_v

    def Mvir_to_MDelta_NFW():
        M_out, r_out = [], []
        for j, M_def in enumerate(M_arr):
            def func(R):
                return M_def * f(R/r_s[j]) / f(concentration[j]) \
                       - (4./3)*np.pi*R**3 * Del * rho_ref * bias
            try:
                r_d = brentq(func, r_arr[initial_r_index_to_try], r_t[j])
                M_d = M_def * f(r_d/r_s[j]) / f(concentration[j])
            except:
                r_d, M_d = 0.0, 0.0
            r_out.append(r_d)
            M_out.append(M_d)
        return np.asarray(M_out), np.asarray(r_out)

    def Mvir_to_MDelta_numerical(rho):
        M_out, r_out = [], []
        for j, M_def in enumerate(M_arr):
            def func(R):
                return get_enclosed_mass(rho=rho[j], r_t=R, r_arr=r_arr, just_one_M=True) \
                       - (4./3)*np.pi*R**3 * Del * rho_ref * bias
            try:
                r_d = brentq(func, r_arr[initial_r_index_to_try], r_t[j])
                M_d = get_enclosed_mass(rho=rho[j], r_t=r_d, r_arr=r_arr, just_one_M=True)
            except:
                r_d, M_d = 0.0, 0.0
            r_out.append(r_d)
            M_out.append(M_d)
        return np.asarray(M_out), np.asarray(r_out)

    if rho is None:
        return Mvir_to_MDelta_NFW()
    return Mvir_to_MDelta_numerical(rho)


def get_rho_cga(r_arr, M, M200c, r200c, A_star=0.09, M_star0=2.5e11, eta_cga=0.6, R_cga_h_frac=0.015):
    R_h   = R_cga_h_frac * r200c
    f_cga = np.clip(A_star * (M_star0 / M200c)**eta_cga, 0.0, 1.0)
    return (M * f_cga / (4*np.pi**(3./2) * R_h * r_arr**2)) * np.exp(-(r_arr / (2.0 * R_h))**2)

def get_spectra(data_LCDM, utilde=None, full_2h=True,
                full_output=False, input_squared_NFW_FT=False,
                alpha = False, eta_given = False, 
                one_halo_damping = True, two_halo_damping = False,
                concentration_param = True):
    """
    Compute the non-linear matter power spectrum with a custom halo profile.

    Parameters
    ----------
    utilde : array_like, optional
        Custom Fourier-space halo profile (or its square if ``input_squared_NFW_FT=True``). If None, just use NFW
    data_LCDM : dict
        ΛCDM data dictionary with keys ``'M'``, ``'power_spec_dic'``,
        ``'cosmos'``, and ``'hmcode_params'``.
    full_2h : bool, optional
        Whether to use the full two-halo term.
    full_output : bool, optional
        If ``True``, return the full tuple ``(PS, 1h, 2h, hmf, M, u, c_min)``.
    input_squared_NFW_FT : bool, optional
        If ``True``, ``utilde`` is |u(k,M)|² rather than u(k,M). Only allowed if full_2h is ``False``

    Returns
    -------
    PS : ndarray or tuple
        Power spectrum array, or full output tuple when ``full_output=True``.
    """
    result = PS_nonlin_cold.func_non_lin_PS_matter(data_LCDM['M'], data_LCDM['power_spec_dic']['k'], data_LCDM['power_spec_dic']['power_total'], 
                                                  data_LCDM['cosmos'], data_LCDM['hmcode_params'], data_LCDM['cosmos']['Omega_m_0'],
                                                  alpha = alpha, 
                                                  eta_given = eta_given, 
                                                  one_halo_damping = one_halo_damping, 
                                                  two_halo_damping = two_halo_damping,
                                                  concentration_param=concentration_param,
                                                  full_2h = full_2h,
                                                  custom_NFW_FT = utilde,
                                                  input_squared_NFW_FT = input_squared_NFW_FT,
                                                  full_output = full_output)
    if full_output:
        return result # PS, 1h, 2h, hmf, M, u, c_min
    return result[0]