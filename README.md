# FeedbackMimic
Repository for constructing halo density profiles that mimic the effects of ULA physics, generating mock data catalogs, and running MCMC

## Making a feedback profile that matches ULA/FDM:
- ``make camb`` in ``axionCAMB/`` to make sure the camb executable exists.
- ``run_halo_model.py`` to run FDM and LCDM across a range of halo masses and redshifts in ``results/halo_model/``
- ``match_profile_for_Pk.py`` to find profiles such that LCDM P(k) matches target power at every half integer, in ``results/profile_matching/``
- ``interpolate_profile_across_z.py`` to interpolate between the pivot z, in ``results/feedback_models/``
## Generating mock cluster catalogs
The parts relevant to making a mock catalog (for LCDM, FDM, and LCDM+feedback):
- ``generate_obs_catalog.py`` to do Poisson sampling with the hmf and get a list of halos by q and z, in ``results/mock_catalogs/``