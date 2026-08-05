# FeedbackMimic
Repository for constructing halo density profiles that mimic the effects of ULA physics, generating mock data catalogs, and running MCMC

## Running FDM and LCDM physics:
- ``make camb`` in ``axionCAMB/`` to make sure the camb executable exists.
- ``run_halo_model.py`` to run FDM and LCDM across a range of halo masses and redshifts in ``results/halo_model/``
## Making LCDM profiles that match FDM:
- ``match_profile_for_Pk.py`` to find profiles such that LCDM P(k) matches target power at every half integer, in ``results/profile_matching/``
- ``match_profile_to_hydro_sims.py`` to find profiles such that LCDM P(k) matches target power for hydrodynamical simulations
- ``interpolate_profile_across_z.py`` to interpolate between the pivot z, in ``results/feedback_models/``
## Generating mock cluster catalogs
The parts relevant to making a mock catalog (for LCDM, FDM, and LCDM+feedback):
- ``generate_obs_catalog.py`` to do Poisson sampling with the hmf and get a list of halos by q_obs, z, and g_t(r), in ``results/mock_catalogs/``
- Note that we save more in ``results/mock_catalogs/``, such as the ``noselection`` files which are built along the way in generating catalogues and don't include the selection cut, ``MWL_data`` which comes from the notebook and houses M_WL for each catalogue and the quantities needed to fit b_WL parameters across sims, and `NelderMeadeOptimized_???WL` which comes from the notebook and houses the Nelder-Meade optimal cosmology and SZ values, given a certain b_WL prior center (this is useful for getting a sense of how big the realization difference is without needing to run more mcmcs)
## Running MCMC
- Note that we build on the so survey files, so continue to name them as "so_sim" even with the changes.
- The plots notebook includes individual snippets for the log likelihood.
- ``run_mcmc.py``
