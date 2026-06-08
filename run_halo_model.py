"""
Compute and save non-linear matter power spectra for a mixed dark matter (FDM) cosmology and its ΛCDM counterpart across a redshift grid.
"""

import pickle
import os
import numpy as np
from axionHMcode.axionCAMB_and_lin_PS import axionCAMB_wrapper
from axionHMcode.axionCAMB_and_lin_PS import load_cosmology
from axionHMcode.axionCAMB_and_lin_PS import lin_power_spectrum
from axionHMcode.halo_model import HMcode_params
from axionHMcode.halo_model import PS_nonlin_cold
from axionHMcode.halo_model import PS_nonlin_axion
from axionHMcode.axion_functions import axion_params
from feedback_mimic import get_spectra
from tqdm import tqdm

CAMB_EXE     = "/home/jange/scratch/FeedbackMimic/axionCAMB"
CAMB_WORKDIR = "/home/jange/scratch/FeedbackMimic/axionHMcode_work_directory"
OUTPUT_DIR   = "/home/jange/scratch/FeedbackMimic/results/halo_model"
 
NUM_M      = 250
Z_ARR      = np.linspace(0.0, 2.0, 67)
DO_FULL_2H = True
C_MIN      = 5.196
 
HMCODE_FLAGS = dict(
    alpha             = False,
    eta_given         = False,
    one_halo_damping  = True,
    two_halo_damping  = False,
    concentration_param = True,
)

FDM_mode = "preferred"
if FDM_mode == "preferred":
    INPUT_FILE   = "/home/jange/scratch/FeedbackMimic/axionHMcode_input/axionHMcode_input_preferred.txt"
    OUTPUT_APPENDIX = "_preferred"

##############################################
if os.path.exists(CAMB_EXE+'/./camb') == False:
    print("executable axionCAMB is not in the given directory, pleas check the path")
    
cosmos = load_cosmology.load_cosmology_input(INPUT_FILE) 
cosmos_LCDM = load_cosmology.load_cosmology_input(INPUT_FILE, LCDM = True)
print("computing the non-linear total matter power spectrum in the folowing MDM cosmology \n")
print("omega_m = {0} \nomega_cdm = {1} \nomega_ax = {2} \nomega_b = {3} \nm_ax = {4}eV \nz = {5} \nh = {6} \n".format(cosmos['omega_m_0'], cosmos['omega_d_0'], cosmos['omega_ax_0'], cosmos['omega_b_0'], 
                                                                                                                          cosmos['m_ax'], cosmos['z'], cosmos['h']))

print(f"working on Z_ARR = {Z_ARR}")

for z in tqdm(Z_ARR):
    cosmos['z'] = z
    cosmos_LCDM['z'] = z
    print("z = " + str(z))

    print("axionCAMB is running. Computes transfer function for cosmology with a axion fraction of {}"
          .format(cosmos['Omega_ax_0']/(cosmos['Omega_ax_0']+cosmos['Omega_d_0'])))
    
    axionCAMB_wrapper.axioncamb_params(f'{CAMB_WORKDIR}/{z}_paramfile_axionCAMB.txt', 
                                        cosmos, output_root=f'{CAMB_WORKDIR}/{z}_cosmos', print_info = False,
                                        transfer_k_per_logint = 60)
    axionCAMB_wrapper.run_axioncamb(f'{CAMB_WORKDIR}/{z}_paramfile_axionCAMB.txt', 
                                    CAMB_EXE, 
                                    cosmos, print_info = False)    
    
    print("axionCAMB is running. Computes transfer function for a LCDM cosmology")
    axionCAMB_wrapper.axioncamb_params(f'{CAMB_WORKDIR}/{z}_paramfile_axionCAMB_LCDM.txt', 
                                       cosmos_LCDM, output_root=f'{CAMB_WORKDIR}/{z}_cosmos_LCDM', print_info = False,
                                        transfer_k_per_logint = 60)
    axionCAMB_wrapper.run_axioncamb(f'{CAMB_WORKDIR}/{z}_paramfile_axionCAMB_LCDM.txt', 
                                    CAMB_EXE, 
                                    cosmos_LCDM, print_info = False)
    
    power_spec_dic_ax = lin_power_spectrum.func_power_spec_dic(f'{CAMB_WORKDIR}/{z}_cosmos_transfer_out.dat', cosmos)
    power_spec_dic_LCDM = lin_power_spectrum.func_power_spec_dic(f'{CAMB_WORKDIR}/{z}_cosmos_LCDM_transfer_out.dat', cosmos_LCDM)

    print("Create dictionary with parameters of HMCode2020")
    M_arr = np.logspace(cosmos['M_min'], cosmos['M_max'], NUM_M)
    hmcode_params = HMcode_params.HMCode_param_dic(cosmos, power_spec_dic_ax['k'], power_spec_dic_ax['power_cold'])
    hmcode_params_LCDM = HMcode_params.HMCode_param_dic(cosmos_LCDM, power_spec_dic_LCDM['k'], power_spec_dic_LCDM['power_cold'])
    if C_MIN is not None:
        hmcode_params['c_min'] = C_MIN
        hmcode_params_LCDM['c_min'] = C_MIN
    
    print("Calculate axion quantities; cut-off mass, central density scale of axion density profile and axion halo mass.")
    # make sure that concentration_param here and in the calculation of the non-linear power spectrum is the same
    axion_param = axion_params.func_axion_param_dic(M_arr, cosmos, power_spec_dic_ax, hmcode_params, concentration_param=HMCODE_FLAGS['concentration_param'])
    
    print('Calculate non-linear power spectrum in mixed DM cosmology with the halo model, full2h')
    PS_matter_nonlin_full2h = PS_nonlin_axion.func_full_halo_model_ax(
        M_arr, power_spec_dic_ax, cosmos, hmcode_params, axion_param,
        alpha=HMCODE_FLAGS['alpha'], eta_given=HMCODE_FLAGS['eta_given'],
        one_halo_damping=HMCODE_FLAGS['one_halo_damping'], two_halo_damping=HMCODE_FLAGS['two_halo_damping'],
        concentration_param=HMCODE_FLAGS['concentration_param'], full_2h=DO_FULL_2H, full_output=True)
 
    print('Calculate non-linear power spectrum in LCDM cosmology with the halo model, full2h')
    data_LCDM = dict(
        M              = M_arr,
        power_spec_dic = power_spec_dic_LCDM,
        cosmos         = cosmos_LCDM,
        hmcode_params  = hmcode_params_LCDM,
    )
    PS_LCDM_matter_nonlin_full2h = get_spectra(
        data_LCDM, utilde=None,
        full_2h=DO_FULL_2H, full_output=True,
        alpha=HMCODE_FLAGS['alpha'], eta_given=HMCODE_FLAGS['eta_given'],
        one_halo_damping=HMCODE_FLAGS['one_halo_damping'], two_halo_damping=HMCODE_FLAGS['two_halo_damping'],
        concentration_param=HMCODE_FLAGS['concentration_param'])

    P, P_1h, P_2h, hmf, M, utilde, c = PS_LCDM_matter_nonlin_full2h

    Pstar, Pstar_CDM, Pstar_1h_CDM, Pstar_2h_CDM,\
     Pstar_total_cross, Pstar_total_ax,\
     cstar, hmf_CDM, M_CDM, utilde_CDM,\
     halo_mass_func_arr, halo_mass_func_arr2,\
     dens_profile_cold_arr_2, dens_profile_ax_arr_2 = PS_matter_nonlin_full2h
    
    data_LCDM_full2h = {
        'P': P, 'P_1h': P_1h, 'P_2h': P_2h,
        'hmf': hmf, 'M': M, 'utilde': utilde, 'c': c,
        'power_spec_dic': power_spec_dic_LCDM,
        'cosmos': cosmos_LCDM,
        'hmcode_params': hmcode_params_LCDM,
    }
 
    data_FDM_full2h = {
        'P': Pstar, 'P_CDM': Pstar_CDM,
        'P_1h_CDM': Pstar_1h_CDM, 'P_2h_CDM': Pstar_2h_CDM,
        'P_total_cross': Pstar_total_cross, 'P_total_ax': Pstar_total_ax,
        'c': cstar, 'hmf_CDM': hmf_CDM, 'M_CDM': M_CDM, 'utilde_CDM': utilde_CDM,
        'hmf': halo_mass_func_arr, 'hmf2': halo_mass_func_arr2,
        'dens_profile_cold_arr_2': dens_profile_cold_arr_2,
        'dens_profile_ax_arr_2': dens_profile_ax_arr_2,
        'power_spec_dic': power_spec_dic_ax,
        'cosmos': cosmos,
        'hmcode_params': hmcode_params,
        'axion_param': axion_param,
    }

    suffix = "full2h" if DO_FULL_2H else ""
    with open(OUTPUT_DIR + f"/{z}_data_LCDM_{suffix}{OUTPUT_APPENDIX}.pkl", "wb") as f:
        pickle.dump(data_LCDM_full2h, f)
    with open(OUTPUT_DIR + f"/{z}_data_FDM_{suffix}{OUTPUT_APPENDIX}.pkl", "wb") as f:
        pickle.dump(data_FDM_full2h, f)
 