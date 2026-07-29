"""
Iteratively construct a halo profile u(k,M) so the power spectrum matches
P_FDM(k), then save the final iteration results.
"""

import pickle
from scipy.interpolate import make_smoothing_spline
import numpy as np
from scipy.interpolate import interp1d
from feedback_mimic import (
    get_some_quantities,
    get_enclosed_mass,
    get_spectra,
    get_NFW_profile,
    get_realspace_profile,
    get_Delta_naive,
    get_Delta,
    uniform_density_ft,
    solve_r_t,
    smooth_kmax_filter,
    Mtot_to_MDelta,
    get_rho_cga
)

OUTPUT_DIR   = "/home/jange/scratch/FeedbackMimic/results/profile_matching"
etaz = 0.7
if etaz == 0.0:
    MODEL = f"no_decay_preferred_cooled"
else:
    MODEL = f"etaz{etaz}_preferred_cooled"

verbose = False

INPUT_DIR   = "/home/jange/scratch/FeedbackMimic/results/halo_model"
INPUT_SUFFIX = "full2h"   # set to '' to load files without the full2h suffix
Z                  = 0.0
NUM_ITERATIONS     = 25
GAMMA_SCALINGS     = [0.0]
K_ARR  = np.logspace(-6, 6, 8192)
R_ARR  = np.logspace(-4, np.log10(1 / K_ARR.min()), 1500)
MIN_R      = 1e-4        # lower bracket for root_scalar [Mpc/h]
MAX_R      = 30         # upper bracket for root_scalar [Mpc/h]

suffix = f"_{INPUT_SUFFIX}" if INPUT_SUFFIX else ""

with open(INPUT_DIR + f"/{Z}_data_LCDM{suffix}_preferred.pkl", "rb") as f:
    data_LCDM = pickle.load(f)
with open(INPUT_DIR + f"/{Z}_data_FDM{suffix}_preferred.pkl", "rb") as f:
    data_FDM = pickle.load(f)
r_v, rho_crit, rho_char, concentration, r_s, Delta_vir, rho_0, rho_0_wh = \
    get_some_quantities(data_LCDM)
M_arr        = data_LCDM['M']                   # we assume these are consistent
k_arr_HMCode = data_LCDM['power_spec_dic']['k'] # for both LCDM and FDM (M_c and k)

    # Just to stop ringing:
window_max = np.exp(-(K_ARR / (K_ARR[-1] / 10))**2)[None,:]

# modify low masses to just equal the baseline - note: we are being a little wasteful by doing the whole proc. above
# this is cheap and dirty and should be replaced with something more efficient for release
M_transition = 2.5e11

###############################################
print(MODEL)

S_cooling = 1.
Pk_base = data_FDM['P']
cooling_baseline = True

if cooling_baseline:
    M200c_arr, r200c_arr = Mtot_to_MDelta(
        data_LCDM, r_v, M_arr, K_ARR, R_ARR, Del=200,
        reference_density="critical"
    )
    A_star  = 0.09
    M_0star = 2.5e11
    eta_cga = 0.6
    f_cga = A_star * (M_0star / M200c_arr)**eta_cga
    R_h     = 0.015 * r200c_arr
    
    rho_cga = np.zeros((np.size(M_arr), len(R_ARR)))
    for j in range(np.size(M_arr)):
        if M_arr[j] >= M_transition:
            rho_cga[j] = get_rho_cga(R_ARR, M_arr[j], M200c_arr[j], r200c_arr[j])
    rho_cga[R_ARR[None,:] > r_v[:,None]] = 0.0 
            
    u_cga = f_cga[:, None] * np.exp(-(K_ARR[None, :] * R_h[:, None])**2)  # (num_M, N_k)
    u_cga_HMCodek = f_cga[:, None] * np.exp(-(k_arr_HMCode[None, :] * R_h[:, None])**2)  # (num_M, N_k)
    M_cga = f_cga * M_arr

    u_cooled = (u_cga_HMCodek + (1 - f_cga[:,None]) * data_LCDM['utilde']) # baseline template
    u_cooled[M_arr < M_transition] = data_LCDM['utilde'][M_arr < M_transition]
    P_cooled = get_spectra(data_LCDM, u_cooled, full_output=False)
    S_cooling = P_cooled / data_LCDM['P']
else:
    u_cga = 0.0
    rho_cga = 0.0
    M_cga = 0.0

if MODEL == "no_decay_preferred_cooled":
    print("Using base FDM + cooling S(k) from BCM cga profile")
    Pk_to_match = Pk_base
elif MODEL[:4] == 'etaz': # assumed cooling here too
    print(f"Using etaz={etaz} times z={Z}")
    with open(INPUT_DIR + f"/{0.0}_data_FDM{suffix}_preferred.pkl", "rb") as f:
        data_FDM_z0 = pickle.load(f)
    with open(INPUT_DIR + f"/{0.0}_data_LCDM{suffix}_preferred.pkl", "rb") as f:
        data_LCDM_z0 = pickle.load(f)
        
    Pk_zindependent = data_LCDM['P'] * data_FDM_z0['P'] / data_LCDM_z0['P']
    Pk_fdm_zdep     = Pk_zindependent + (1 - (Pk_zindependent / data_LCDM['P']) ** (etaz * Z)) * data_LCDM['P']
    Pk_to_match = Pk_fdm_zdep # not matching cooling, JUST suppression
    if (Pk_fdm_zdep/data_LCDM['P'])[-1] > 1:
        print("doing LCDM as thing to match to (+cooling), not Pkfdmindep (+cooling)")
        Pk_to_match = data_LCDM['P']
    if verbose:
        print("f_cga:", f_cga.min(), f_cga.max())
else:
    print("Using base FDM")
    Pk_to_match = Pk_base

if verbose:
    print(f"Pk_to_match / P_LCDM at low k: {(Pk_to_match/data_LCDM['P'])[:5]}")
    print(f"k values: {k_arr_HMCode[:5]}")
    
for gamma_scaling in GAMMA_SCALINGS:
    print(f"\n{'='*60}\ngamma_scaling = {gamma_scaling}")
 
    gamma    = gamma_scaling * np.ones(np.size(M_arr))
    k0_of_M  = 1e-2 * np.ones(np.size(M_arr))
 
    v_ratio = np.ones(np.size(M_arr))
    w_ratio = np.ones(np.size(k_arr_HMCode))
    
    my_results = []
    
    for i in np.arange(NUM_ITERATIONS +1):
        print(f"  iteration {i}")
        if i > 0:
            w_ratio *= np.sqrt(Pk_to_match*S_cooling/my_results[i-1]['P'])
            v_ratio *= np.sqrt(1./my_results[i-1]['u_k_hat_unbiased'][:,0])
            if verbose:
                print(f"  w_ratio low-k: {w_ratio[:5]}")
                print(f"  v_ratio high-k: {v_ratio[M_arr > 1e14][:5]}")

        u_k_NFW_to_rv = get_NFW_profile(r_t=r_v, r_v=r_v, k_arr=K_ARR, r_s=r_s) # data_LCDM['utilde']
        u_k_NFW_to_max = get_NFW_profile(r_t=max(R_ARR)*np.ones_like(r_v), r_v=r_v, k_arr=K_ARR, r_s=r_s) # data_LCDM['utilde']
        u_k_NFW_to_max *= window_max
        rho_NFW_to_rv = get_realspace_profile(u_k=u_k_NFW_to_max,
                                          r_arr=R_ARR, k_arr=K_ARR, M_arr=M_arr, r_t=r_v)
        
        u_k_NFW_to_rv_interptoHMCode = np.array([interp1d(K_ARR, row, kind='linear')(k_arr_HMCode) for row in u_k_NFW_to_rv])

        Delta_naive, H, L = get_Delta_naive(Pk_to_match,data_LCDM['P_1h'],data_LCDM['P_2h'],data_LCDM['P_1h'],data_LCDM['P_2h'],
                                                           all_outputs = True)
        P_1h_highk_modified = get_spectra(data_LCDM, 
                                          (1 - f_cga[:,None]) * u_k_NFW_to_rv_interptoHMCode * (Delta_naive * w_ratio)[None,:], 
                                          full_output = True)[1]
        Delta_naive, H, L = get_Delta_naive(Pk_to_match,P_1h_highk_modified,data_LCDM['P_2h'],data_LCDM['P_1h'],data_LCDM['P_2h'],
                                                           all_outputs = True)
        
        # Now we need to compute the enclosed mass in r_v
        #Delta_naive_interp = np.interp(K_ARR, k_arr_HMCode, Delta_naive * w_ratio)
        Delta_naive_interp = interp1d(k_arr_HMCode, Delta_naive * w_ratio, 
                               kind='linear', fill_value='extrapolate')(K_ARR)
    
        u_k_naive_to_rv = u_k_NFW_to_rv * Delta_naive_interp[None,:]
        u_k_naive_to_max = u_k_NFW_to_max * Delta_naive_interp[None,:]
        rho_naive_to_rv_NFWmod = get_realspace_profile(u_k=u_k_NFW_to_max * Delta_naive_interp[None,:],
                                          r_arr=R_ARR, k_arr=K_ARR, M_arr=M_arr, r_t=r_v)
            
        mhat = get_enclosed_mass(rho=rho_naive_to_rv_NFWmod,
                                            r_t=r_v, r_arr=R_ARR)
        mass_to_have_from_rt = mhat + gamma * (M_arr - mhat)
        
        # Solve for the r_t
        r_t = solve_r_t(mass_to_have_from_rt, Delta_naive_interp,
                        r_v, r_s, M_arr, R_ARR, K_ARR, MIN_R, MAX_R, u_k_base = u_k_NFW_to_max, fcga=f_cga)

        u_k_naive_to_rt = get_NFW_profile(r_t=r_t, r_v=r_v, k_arr=K_ARR, r_s=r_s) * Delta_naive_interp[None,:]
        u_k_naive_to_max = u_k_NFW_to_max * Delta_naive_interp[None,:]
        rho_naive_to_rt = get_realspace_profile(u_k=u_k_NFW_to_max * Delta_naive_interp[None,:],
                                          r_arr=R_ARR, k_arr=K_ARR, M_arr=M_arr, r_t=r_t)
        
        mass_to_add_from_constant = M_arr - mass_to_have_from_rt
        
        u_k_constant = uniform_density_ft(mass_to_add_from_constant, M_arr, r_t, K_ARR)
        
        u_k_hat = u_k_naive_to_rt + u_k_constant
        rho_hat = get_realspace_profile(u_k=get_NFW_profile(r_t=max(R_ARR)*np.ones_like(r_v), r_v=r_v, k_arr=K_ARR, r_s=r_s) * Delta_naive_interp[None,:] * window_max,
                                          r_arr=R_ARR, k_arr=K_ARR, M_arr=M_arr, r_t=r_t) \
                                            + get_realspace_profile(u_k=u_k_constant,
                                          r_arr=R_ARR, k_arr=K_ARR, M_arr=M_arr, r_t=r_t) 
        
        # Now unbiasing matter
        
        # First we need the new unbiased delta
        Delta = get_Delta(
            low_k_value  = v_ratio * (1.0 - u_k_constant.T[0]),
            high_k_value = Delta_naive_interp,
            k0_of_M      = k0_of_M,
            k_arr=K_ARR, M_arr=M_arr,
        )
        
        if verbose:
            print(f"  Delta_naive low-k: {Delta_naive[:5]}")
            print(f"  Delta interp low-k: {np.interp(K_ARR[:5], k_arr_HMCode, Delta_naive * w_ratio)}")
        
        # Now we need to compute the enclosed mass in r_v (unbiased)
        u_k_unbiased_to_rv = u_k_NFW_to_rv * Delta
        u_k_unbiased_to_max = u_k_NFW_to_max * Delta
        rho_unbiased_to_rv_NFWmod = get_realspace_profile(u_k=u_k_NFW_to_max * Delta,
                                            r_arr=R_ARR, k_arr=K_ARR, M_arr=M_arr, r_t=r_v)
            
        mhat_unbiased = get_enclosed_mass(rho=rho_unbiased_to_rv_NFWmod,
                                                    r_t=r_v, r_arr=R_ARR)
        mass_unbiased_to_have_from_rt = mhat_unbiased + gamma * (M_arr - mhat_unbiased)
        
        # Solve for the r_t
        r_t_unbiased = solve_r_t(mass_unbiased_to_have_from_rt, Delta,
                                 r_v, r_s, M_arr, R_ARR, K_ARR, MIN_R, MAX_R, u_k_base = u_k_NFW_to_max, fcga=f_cga)

        u_k_unbiased_to_rt = get_NFW_profile(r_t=r_t_unbiased, r_v=r_v, k_arr=K_ARR, r_s=r_s) * Delta
        u_k_unbiased_to_max = u_k_NFW_to_max * Delta
        rho_unbiased_to_rt = get_realspace_profile(u_k=get_NFW_profile(r_t=max(R_ARR)*np.ones_like(r_v), r_v=r_v, k_arr=K_ARR, r_s=r_s) * Delta * window_max,
                                          r_arr=R_ARR, k_arr=K_ARR, M_arr=M_arr, r_t=r_t_unbiased)
        
        mass_unbiased_to_add_from_constant = M_arr - mass_unbiased_to_have_from_rt
        
        u_k_constant_unbiased = uniform_density_ft(
            mass_unbiased_to_add_from_constant, M_arr, r_t_unbiased, K_ARR)

        M_NFW_enclosed = get_enclosed_mass(
            rho=get_realspace_profile(
                u_k=get_NFW_profile(r_t=max(R_ARR)*np.ones_like(r_v), r_v=r_v, k_arr=K_ARR, r_s=r_s) * Delta * window_max,
                r_arr=R_ARR, k_arr=K_ARR, M_arr=M_arr, r_t=r_t_unbiased),
            r_t=r_t_unbiased, r_arr=R_ARR)
        #M_constant_enclosed = get_enclosed_mass(
        #    rho=get_realspace_profile(u_k=u_k_constant_unbiased,
        #        r_arr=R_ARR, k_arr=K_ARR, M_arr=M_arr, r_t=r_t_unbiased),
        #    r_t=r_t_unbiased, r_arr=R_ARR)
        M_constant_enclosed = mass_unbiased_to_add_from_constant
        M_cga_enclosed = f_cga * M_arr
        M_total_enclosed = (1 - f_cga) * M_NFW_enclosed + (1 - f_cga) * M_constant_enclosed + M_cga_enclosed
        rho_NFW = get_realspace_profile(u_k=get_NFW_profile(r_t=max(R_ARR)*np.ones_like(r_v), r_v=r_v, k_arr=K_ARR, r_s=r_s) * Delta * window_max,
                                          r_arr=R_ARR, k_arr=K_ARR, M_arr=M_arr, r_t=r_t_unbiased)
        #rho_constant = get_realspace_profile(u_k=u_k_constant_unbiased,
        #                                  r_arr=R_ARR, k_arr=K_ARR, M_arr=M_arr, r_t=r_t_unbiased)
        rho_constant_value = mass_unbiased_to_add_from_constant / (4./3 * np.pi * r_t_unbiased**3)
        rho_constant = np.zeros_like(rho_NFW)
        rho_constant[R_ARR[None,:] <= r_t_unbiased[:,None]] = np.repeat(rho_constant_value[:,None], len(R_ARR), axis=1)[R_ARR[None,:] <= r_t_unbiased[:,None]]

        u_k_hat_unbiased = u_cga + (1 - f_cga[:,None]) * (u_k_unbiased_to_rt + u_k_constant_unbiased)
        u_k_hat_unbiased[M_arr < M_transition] = u_k_NFW_to_rv[M_arr < M_transition]
        u_k_NFW = get_NFW_profile(r_t=r_t_unbiased, r_v=r_v, k_arr=K_ARR, r_s=r_s) * Delta
        u_k_constant = u_k_constant_unbiased
        
        rho_hat_unbiased = rho_cga + (1 - f_cga[:,None]) * (rho_NFW + rho_constant)
        
        if verbose:
            print(f"  u_k_hat_unbiased[:,0] (should→1): min={u_k_hat_unbiased[:,0].min():.4f}, max={u_k_hat_unbiased[:,0].max():.4f}")
            print("u_k_unbiased_to_rt")
            print(u_k_unbiased_to_rt[:,0].min(), u_k_unbiased_to_rt[:,0].max())
            print("u_k_constant_unbiased")
            print(u_k_constant_unbiased[:,0].min(), u_k_constant_unbiased[:,0].max())
            print("sum of u_k_unbiased_to_rt + u_k_constant_unbiased")
            print((u_k_unbiased_to_rt[:,0] + u_k_constant_unbiased[:,0]).min(), 
                  (u_k_unbiased_to_rt[:,0] + u_k_constant_unbiased[:,0]).max())
            sum_lowk = u_k_unbiased_to_rt[:,0] + u_k_constant_unbiased[:,0]
            offending = sum_lowk > 1.01  # or whatever threshold
            print("Offending M_arr bins:", M_arr[offending])
            #print("f_cga at offending bins:", f_cga[offending])
            #print("sum at offending bins:", sum_lowk[offending])
            print("r_v at high-mass offending bins:", r_v[M_arr > 1e14])
            print("r_s at high-mass offending bins:", r_s[M_arr > 1e14])
            print("concentration at high-mass offending bins:", concentration[M_arr > 1e14])
            print("mhat_unbiased at high-mass offending bins:", mhat_unbiased[M_arr > 1e14])
            
            
        # PS with modified profile
        u_k_hat_unbiased_HMCode = np.array([interp1d(K_ARR, row, kind='linear')(k_arr_HMCode) for row in u_k_hat_unbiased])
        P_tot, P_1h, P_2h, hmf, M, u_k_axionHMCode, _ = get_spectra(
            data_LCDM, utilde=u_k_hat_unbiased_HMCode, full_output=True)

        if verbose:
            print(f"  P_tot/P_LCDM low-k: {(P_tot/data_LCDM['P'])[:5]}")
            print(f"  P_tot/Pk_to_match low-k: {(P_tot/Pk_to_match)[:5]}")
            print(f"  P_1h low-k: {P_1h[:5]}")
            print(f"  P_2h low-k: {P_2h[:5]}")
            print(f"  P_LCDM_1h low-k: {data_LCDM['P_1h'][:5]}")
    
        ##################################
        '''
        iteration_results = {
            "Delta_naive": Delta_naive, # original blending between H and L
            "H": H, # high-k fraction from power spectra
            "L": L, # low-k fraction from power spectra
            "u_k_naive_to_rv": u_k_naive_to_rv, # profile using Delta naive
            "rho_naive_to_rv": rho_naive_to_rv, # profile using Delta naive
            "mhat": mhat, # enclosed virial mass from profile using Delta naive
            "mass_to_have_from_rt": mass_to_have_from_rt, # target truncation mass
            "r_t": r_t, # truncation radius
            "u_k_naive_to_rt": u_k_naive_to_rt, # profile using Delta naive to r_t
            "rho_naive_to_rt": rho_naive_to_rt, # profile using Delta naive to r_t
            "mass_to_add_from_constant": mass_to_add_from_constant, # remaining mass
            "u_k_constant": u_k_constant, # profile to add that includes remaining mass
            "u_k_hat": u_k_hat, # overall profile 
            "rho_hat": rho_hat, # overall profile
            "Delta": Delta, # new Delta, which blends from Delta_naive to unbiased low-k behavior
            "k0_of_M": k0_of_M, # cutoff point for Delta_naive and unbiased low-k behavior
            "u_k_unbiased_to_rv": u_k_unbiased_to_rv,                               # all the following are the same as above,
            "rho_unbiased_to_rv": rho_unbiased_to_rv,                               # but unbiased and using Delta rather than
            "mhat_unbiased": mhat_unbiased,                                         # Delta_naive
            "mass_unbiased_to_have_from_rt": mass_unbiased_to_have_from_rt,
            "r_t_unbiased": r_t_unbiased,
            "u_k_unbiased_to_rt": u_k_unbiased_to_rt,
            "rho_unbiased_to_rt": rho_unbiased_to_rt,
            'mass_unbiased_to_add_from_constant' : mass_unbiased_to_add_from_constant,
            'u_k_constant_unbiased': u_k_constant_unbiased,
            "u_k_hat_unbiased": u_k_hat_unbiased, # overall profile 
            "rho_hat_unbiased": rho_hat_unbiased, # overall profile
            "P": P_tot, # overall power spectrum
            "P_1h": P_1h, # 1h term
            "P_2h": P_2h, # 2h term
            "hmf": hmf, # hmf
            "u_k_axionHMCode": u_k_axionHMCode, # profile from axionHMCode
        }
        '''
        iteration_results = {
            "Delta": Delta, # new Delta, which blends from Delta_naive to unbiased low-k behavior
            "r_t_unbiased": r_t_unbiased, # truncation for NFW and constant, cga is truncated at r_v
            "u_k_hat_unbiased": u_k_hat_unbiased, # overall profile 
            "rho_hat_unbiased": rho_hat_unbiased, # overall profile
            "P": P_tot, # overall power spectrum
            "P_1h": P_1h, # 1h term
            "P_2h": P_2h, # 2h term
            "hmf": hmf, # hmf
            "u_k_axionHMCode": u_k_axionHMCode, # profile from axionHMCode
            "M_total_enclosed": M_total_enclosed,
            "M_NFW_enclosed": (1 - f_cga) * M_NFW_enclosed,
            "M_constant_enclosed": (1 - f_cga) * M_constant_enclosed,
            "M_cga_enclosed": M_cga_enclosed,
            "rho_NFW": (1 - f_cga[:,None]) * rho_NFW,
            "rho_constant": (1 - f_cga[:,None]) * rho_constant,
            "rho_cga": rho_cga,
            "u_k_NFW": (1 - f_cga[:,None]) * u_k_NFW,
            "u_k_constant": (1 - f_cga[:,None]) * u_k_constant,
            "u_k_cga": u_cga
        }
        my_results.append(iteration_results)
    
    np.save(OUTPUT_DIR + f"/{MODEL}/utilde_results_gamma{gamma_scaling}_{NUM_ITERATIONS}it_z{Z}.npy", [my_results[-1]])