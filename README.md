# FeedbackMimic
Repository for constructing halo density profiles that mimic the effects of ULA physics, generating mock data catalogs, and running MCMC

- First use ``make camb`` in ``axionCAMB/`` to make sure the camb executable exists.
- Then use ``run_halo_model.py`` to run FDM and LCDM across a range of halo masses and redshifts in ``results/halo_model/``