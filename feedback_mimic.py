from axionHMcode.halo_model import PS_nonlin_cold

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