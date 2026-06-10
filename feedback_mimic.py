from axionHMcode.halo_model import PS_nonlin_cold
import numpy as np
from tqdm import tqdm
from scipy import special
from scipy.integrate import simpson, trapezoid, quad
from scipy.interpolate import interp1d
from scipy.optimize import brentq, root_scalar
from mcfit import SphericalBessel
from axionHMcode.halo_model.axion_density_profile import func_dens_profile_ax, MaxofMc
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

def get_FDM_profile(data, r_arr, M_arr, k_arr, full_output=False):
    """
    Real-space cold + fuzzy dark matter (FDM) profile = axion + NFW components.

    Parameters
    ----------
    data : dict
        Must contain keys ``'cosmos'``, ``'power_spec_dic'``,
        ``'hmcode_params'``, and ``'axion_param'``.
    r_arr : array_like
        Radial grid.
    M_arr : array_like
        Halo mass array (cold).
    k_arr : array_like
        Wavenumber grid.
    full_output : bool, optional
        If ``True``, return ``(rho_ax, rho_NFW)`` separately.

    Returns
    -------
    rho_total : ndarray or tuple of ndarray
        Shape ``(num_M, N_r)``. If ``full_output`` is ``True``, returns
        ``(rho_ax, rho_NFW)`` instead.
    """
    cosmo_dic = data['cosmos']
    r_v, rho_crit, rho_char, concentration, r_s, Delta_vir, rho_0, rho_0_wh = get_some_quantities(data, axion_dic=data['axion_param'])
    
    rho_ax_ex = func_dens_profile_ax(r_arr=r_arr, M=M_arr, cosmo_dic=data['cosmos'], power_spec_dic=data['power_spec_dic'], 
                                  rho_central_param = data['axion_param']['central_dens'], hmcode_dic=data['hmcode_params'], 
                                  concentration_param=True, eta_given=False, axion_dic=data['axion_param'])
    rho_ax = np.zeros((np.size(M_arr),np.size(r_arr)))
    for i, M_val in enumerate(M_arr):
        rho_ax[i] = rho_ax_ex[i]
        rho_ax[i][r_arr > r_v[i]] = 0.0
    
    rho_NFW = get_realspace_profile(get_NFW_profile(r_t=max(r_arr) * r_v/r_v, r_v=r_v, k_arr=k_arr, r_s=r_s, just_one_M=False), r_t=r_v, r_arr=r_arr, k_arr=k_arr, M_arr=M_arr, just_one_M=False)

    if full_output:
        return rho_ax,rho_NFW
    
    return rho_ax + rho_NFW


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

def get_total_mass(data):
    c_frac = 1 - data['cosmos']['omega_ax_0']/data['cosmos']['omega_m_0']
    Ma_func = MaxofMc(data['M_CDM'], beta1=data['axion_param']['beta1'], beta2=data['axion_param']['beta2'], z=data['cosmos']['z'], OMEGA_M=data['cosmos']['Omega_m_0'], 
                       c_frac=c_frac, little_h=data['cosmos']['h'], m_a=data['cosmos']['m_ax'], 
                       version=data['cosmos']['version'], M_cut=data['axion_param']['M_cut'], no_cut = True)
    Mtot = data['M_CDM'] + Ma_func
    return Mtot

def dn_dlog10Mtot(data, cosmology_type):
    if cosmology_type == "LCDM":
        dnLCDM_dlnMc = data['hmf'] # (1/M dn/dlnM) in units (h^4/MsolMpc^3)
        dnLCDM_dlnMc *= data['M'] # (dn/dlnM) in units (h^3/Mpc^3)
        dnLCDM_dlogMc = dnLCDM_dlnMc * np.log(10)  # (dn/dlogM)
        
        return data['M'], dnLCDM_dlogMc
    elif cosmology_type == "FDM_ST":
        # This is Sheth-Tormen FDM, just uses the linear cold power spectrum
        dnST_dlnMc = data['hmf'] * data['M_CDM']
        dnST_dlogMc = dnST_dlnMc * np.log(10)
        Mtot = get_total_mass(data)
        dnST_dlogMtot = dnST_dlnMc * np.gradient(np.log(data['M_CDM']))/np.gradient(np.log10(Mtot))

        return Mtot, dnST_dlogMtot
    if cosmology_type == "FDM" or cosmology_type == "FDM_preferred":
        # This is the Dome fix for Sheth-Tormen FDM 
        dnST_dlnMc = data['hmf'] * data['M_CDM']
        dnST_dlogMc = dnST_dlnMc * np.log(10)
        Mtot = get_total_mass(data)
        dnST_dlogMtot = dnST_dlnMc * np.gradient(np.log(data['M_CDM']))/np.gradient(np.log10(Mtot))

        # This is the Dome fix
        alpha1 = 1.1
        alpha2 = 2.2
        beta = data['axion_param']['beta2']
        f_ax = data['cosmos']['omega_ax_0'] / (data['cosmos']['omega_ax_0']+data['cosmos']['omega_d_0']+data['cosmos']['omega_b_0'])
        M0 = 1.6e10 * (data['cosmos']['m_ax']/1e-22)**(-4/3) * data['cosmos']['h']
        factor = 1 - beta*f_ax + beta*f_ax*(1 + (Mtot/M0)**(-alpha1))**(-alpha2)
        
        dn_dlnMtot = dnST_dlnMc * factor
        dn_dlogMtot = dn_dlnMtot * np.log(10)
        
        return Mtot, dn_dlogMtot

def comoving_distance(data, z=None):
    if z is None:
        z = data['cosmos']['z']
    c = 299792.458
    integral, _ = quad(lambda z_prime: 1 / H_of_z(z_prime,
                                                  data['cosmos']['h'],
                                                  data['cosmos']['Omega_m_0'],
                                                  data['cosmos']['Omega_w_0']),
                       0, z)
    return c * integral

def dV_dz(data, z=None):
    if z is None:
        z = data['cosmos']['z']
    c = 299792.458
    Dc = comoving_distance(data, z)
    Hz = H_of_z(z, data['cosmos']['h'],
                data['cosmos']['Omega_m_0'],
                data['cosmos']['Omega_w_0'])
    return 4*np.pi * Dc**2 * (c/Hz) # Mpc^3 / dz

def universal_pressure_profile(data, axion_dic, \
                              rho, r_t, M_arr, k_arr, r_arr,
                              alpha = 1.0510, beta = 5.4905, gamma = 0.3081, c500 = 1.177, P0h70m32 = 8.403, alpha_P = 0.12, include_alpha_P_prime=True,
                              bias_sz = 1.0, full_output=False):    
    # P500
    r_v, rho_crit, rho_char, concentration, r_s, Delta_vir, rho_0, rho_0_wh = get_some_quantities(data)
    M500, r500 = Mtot_to_MDelta(
                data, r_t, M_arr, k_arr, r_arr, Del=500, axion_dic=axion_dic, rho=rho,\
                initial_r_index_to_try = 5, reference_density="critical", bias=bias_sz
            )
    f_B = data['cosmos']['Omega_b_0']/data['cosmos']['Omega_m_0']
    G = 4.300917270069977e-09 # Mpc Msun^-1 (km/s)^2
    mu = 0.59
    mu_e = 1.14
    P500 = (500.*f_B*G*mu*rho_crit/(2.*mu_e)) * (M500/r500) # km^2 * M_sun / (Mpc^3 * s^2)
    P500 *= 1.989e33 # km^2 * g / (Mpc^3 * s^2)
    P500 /= 3.086e24**3 # km^2 * g / (cm^3 * s^2)
    P500 *= 1e5**2 # cm^2 * g / (cm^3 * s^2) = erg / cm^3
    P500 *= 6.242e8 # keV / cm^3

    # p(x) - dimensionless pressure shape
    x = np.einsum('r,M->Mr',r_arr,1./r500) # M,r
    P_0 = P0h70m32 * (data['cosmos']['h']/0.7)**(3./2)
    p_x = P_0 * (c500*x)**-gamma * (1 + (c500*x)**alpha)**((gamma-beta)/alpha)
    # vals from Arnaud:
    # P0, c500, γ, α, β = 8.403 (h_70)**(3./2), 1.177, 0.3081, 1.0510, 5.4905

    # other mass dependence, which deviates from self-similarity
    # from Hasselfield (cites Arnaud), an approx: alpha_p_x = 0.22 / (1 + 8*x**3)
    # from Arnaud :
        #alpha_P = 0.12
        #alpha_P_prime = 0.10 - (alpha_P + 0.10) * (x/0.5)**3 / (1 + (x/0.5)**3)
        #alpha_p_x = alpha_P + alpha_P_prime 
    if include_alpha_P_prime:
        alpha_P_prime = 0.10 - (alpha_P + 0.10) * (x/0.5)**3 / (1 + (x/0.5)**3)
    else:
        alpha_P_prime = 0.
    alpha_p_x = alpha_P + alpha_P_prime 
        
    m = M500 / (3e14 * 0.7)

    P = P500[:,None] * p_x * m[:,None]**alpha_p_x # M,r
    P[r_arr[None,:] > r_t[:,None]] = 0.0

    if full_output:
        return P, r500
    return P

def compute_y0(P, r_arr, r500, h, scale_on_r500_for_int_bounds = 5):
    sigma_T_over_mec2 = 6.6524587321e-29 * 1e4 / 511 # cm^2 / keV.
    Mpc_to_cm = 3.086e24 / h  # cm per Mpc/h

    y0 = np.zeros(len(r500))
    
    for i in range(len(r500)):
        mask = r_arr <= scale_on_r500_for_int_bounds * r500[i] # Arnaud suggestion for r_max
        r_int = r_arr[mask] * Mpc_to_cm # cm
        P_int = P[i, mask] # keV cm^-3
        
        y0_numerical = 2 * sigma_T_over_mec2 * np.trapezoid(P_int, r_int)

        y0[i] = y0_numerical

    # this now roughly matches cosmocnc, at least for the mass range and redshifts that are most relevant
    return y0

def profile_to_q(data, axion_dic, \
                              rho, r_t, M_arr, k_arr, r_arr,
                              bias_sz = 0.8, dof = 0.0,
                              so_sim_sz_mf_noise_file_path="/home/jange/scratch/FeedbackMimic/cosmocnc_axionHMcode_jax/cosmocnc_jax/data/so_sim_sz_mf_noise.npy",
                              full_output = False,
                              alpha = 1.0510, beta = 5.4905, gamma = 0.3081, c500 = 1.177, P0h70m32 = 8.403, alpha_P = 0.12, include_alpha_P_prime=True,
                              profile_type = "Arnaud"):
    # y0
    if profile_type == "Arnaud": 
        P, r500_biased = universal_pressure_profile(data, axion_dic, \
                                  rho, r_t, M_arr, k_arr, r_arr,
                                  alpha, beta, gamma, c500, P0h70m32, alpha_P, include_alpha_P_prime,
                                  bias_sz = bias_sz, full_output=True)
    y0 = compute_y0(P, r_arr, r500_biased, data['cosmos']['h'])
    log_y0 = np.log(y0)

    # theta_500
    D_A = comoving_distance(data) / (1 + data['cosmos']['z'])
    theta_500 = r500_biased / D_A * (180 * 60 / np.pi)
    log_theta_500 = np.log(theta_500)

    # MMF noise
    file_theta_500_vec, file_sigma_sz_vec = np.load(so_sim_sz_mf_noise_file_path)
    file_theta_500_vec *= 180. * 60. / np.pi  # rad to arcmin
    sigma_sz_poly = np.polyfit(np.log(file_theta_500_vec), np.log(file_sigma_sz_vec), deg=3)
    log_sigma_sz = np.polyval(sigma_sz_poly, log_theta_500)

    # layer 0
    log_q_true = log_y0 - log_sigma_sz
    # layer 1
    q = np.sqrt(np.exp(log_q_true)**2 + dof)

    if full_output:
        return q, y0, np.exp(log_sigma_sz), theta_500

    return q

def get_tangential_shear(rho, r_arr, z, Mtot, M200c_arr, r_t, Sigma_crit_arr,
            z_max = 20.0, R_min = 0.5, R_max = None, r_min = 1e-4, r_max = None, full_output=False,
           R_for_Sigma_num = 100, R_proj_arr_num=100, M_min_to_consider=10**12.5):
    """
    Parameters
    ----------
    rho : array_like, shape ``(num_M, N_r)``
        Real-space density profiles.
    r_arr : array_like, shape ``(N_r,)``
        Radial grid (same units as ``R`` in ``NFW_DeltaSigma``).
    z : float
        Halo redshift.
    M200c_arr : array_like, shape ``(num_M,)``
        M_200c values used as initial guesses for the fit. dmo 
    r_t : array_like, shape ``(num_M,)``
        Virial radii, used to set the upper limit of the log radial grid.
    """
    if R_max is None:
        R_max = 3.2 / (1 + z) # these is set by Bocquet
    if r_max is None:
        r_max = R_max
    R_proj_arr = np.linspace(R_min, R_max, R_proj_arr_num)
    R_for_Sigma = np.logspace(np.log10(r_min), np.log10(r_max), R_for_Sigma_num)

    DeltaSigmas = []
    Sigmas = []
    tangential_shear_prof = []
    for i in range(len(M200c_arr)):
        if Mtot[i] <= M_min_to_consider:  # skip low masses
            MWL_arr[i] = M200c_arr[i]  # b_WL = 1 assumption
            DeltaSigmas.append(None)
            Sigmas.append(None)
            continue
        r_log_i = np.logspace(np.log10(r_min), np.log10(r_t[i]), 3000)
        rho_interp_i = interp1d(r_log_i, np.interp(r_log_i, r_arr, rho[i]),
                                kind='linear', bounds_error=False, fill_value=0.)

        # Sigma(r)
        Sigma_wide = np.zeros(len(R_for_Sigma))
        for j, R in enumerate(R_for_Sigma):
            #z_los = np.concatenate([np.linspace(0, R, 500), np.linspace(R, z_max, 200)[1:]])
            z_inner = np.logspace(-8, np.log10(R), 1000)
            z_outer = np.logspace(np.log10(R), np.log10(z_max), 500)
            z_los = np.concatenate([[0], z_inner, z_outer[1:]])
            Sigma_wide[j] = 2 * np.trapezoid(rho_interp_i(np.sqrt(R**2 + z_los**2)), z_los)
        if full_output:
            Sigmas.append(Sigma_wide.copy())
        Sigma_arr = np.interp(R_proj_arr, R_for_Sigma, Sigma_wide)

        # DeltaSigma(r)
        Sigma_arr = np.interp(R_proj_arr, R_for_Sigma, Sigma_wide)
        DeltaSigma_arr = np.zeros(len(R_proj_arr))
        for j, R in enumerate(R_proj_arr):
            R_inner = np.logspace(np.log10(r_min), np.log10(R), 500)
            mean_Sigma = 2.0 / R**2 * np.trapezoid(np.interp(R_inner, R_for_Sigma, Sigma_wide) * R_inner, R_inner)
            DeltaSigma_arr[j] = mean_Sigma - Sigma_arr[j]
        if full_output:
            DeltaSigmas.append(DeltaSigma_arr.copy())


        # tangential shear
        tangential_shear_prof.append(DeltaSigma_arr / (Sigma_crit_arr - Sigma_arr))

    if full_output:
        return tangential_shear_prof, DeltaSigmas, Sigmas, R_proj_arr
    return tangential_shear_prof, R_proj_arr