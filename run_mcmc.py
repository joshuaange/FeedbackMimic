"""
MCMC using cobaya: fit cluster counts + shear to mock catalogue.
"""
import os

_NT = os.environ.get("NT", "1")
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = _NT
if _NT == "1":
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
os.environ["OMP_NUM_THREADS"]      = _NT
os.environ["MKL_NUM_THREADS"]      = _NT
os.environ["OPENBLAS_NUM_THREADS"] = _NT

os.environ["JAX_PLATFORMS"]          = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"]         = "1"
os.environ.pop("DISPLAY", None)

# NT=2 mpirun -n 16 python run_mcmc.py

_RAM_PER_CHAIN_GB = 8.5
N_CORES_PER_CHAIN = int(_NT)

try:
    mpi_rank    = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", 0)))
    core_offset = int(os.environ.get("CORE_OFFSET", "0"))          # NEW
    core_start  = core_offset + mpi_rank * N_CORES_PER_CHAIN
    cores = set(range(core_start, core_start + N_CORES_PER_CHAIN))
    os.sched_setaffinity(0, cores)
    print(f"Rank {mpi_rank} pinned to cores {cores}")
except AttributeError:
    pass

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("cobaya").setLevel(logging.WARNING)
logging.getLogger("jax").setLevel(logging.WARNING)
logging.disable(logging.DEBUG)

import pickle, numpy as np, time
from cobaya.likelihood import Likelihood
from cobaya.run import run
from threadpoolctl import threadpool_limits

from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

print(f"MPI rank {RANK}/{SIZE} starting", flush=True)


mock_obs_type = "LCDM_DES_mock_seed50" 
import os

os.environ["TOMOGRAPHIC_BINS"] = "DES"
from mcmc_likelihood import get_log_likelihood, get_G_a_interp, get_D_grid_interp, get_norm_interp, get_Sigma_crit_arr

# ── Paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR = "/home/user/FeedbackMimic/results/"

catalogues = {
    "LCDM_DES_mock_seed50"                        : f"{OUTPUT_DIR}mock_catalogs/LCDM_DES_mock_seed50.pkl",
    "LCDM_DES_mock_seed60"                        : f"{OUTPUT_DIR}mock_catalogs/LCDM_DES_mock_seed60.pkl",
    "LCDM_DES_mock_seed70"                        : f"{OUTPUT_DIR}mock_catalogs/LCDM_DES_mock_seed70.pkl",
}
fiducials = {
    # using the "fiducial" SZ parameters from 25 combined sims
    "LCDM_DES_mock_seed50"                        : (0.143344, 2.1e-09, -4.352890598769685, 1.1991734303623067, 1.9996943667059135, 0.0, 0.0, 1.0),
    "LCDM_DES_mock_seed60"                        : (0.143344, 2.1e-09, -4.352890598769685, 1.1991734303623067, 1.9996943667059135, 0.0, 0.0, 1.0),
    "LCDM_DES_mock_seed70"                        : (0.143344, 2.1e-09, -4.352890598769685, 1.1991734303623067, 1.9996943667059135, 0.0, 0.0, 1.0),
}
output_names = {
    "LCDM_DES_mock_seed50"                        : f"{OUTPUT_DIR}mcmc/LCDM_DES_50_cobaya",
    "LCDM_DES_mock_seed60"                        : f"{OUTPUT_DIR}mcmc/LCDM_DES_60_cobaya",
    "LCDM_DES_mock_seed70"                        : f"{OUTPUT_DIR}mcmc/LCDM_DES_70_cobaya",
}
initial_covmats = {
    "LCDM_DES_mock_seed50"                        : None,
    "LCDM_DES_mock_seed60"                        : None,
    "LCDM_DES_mock_seed70"                        : None,
}
CATALOGUE_PATH = catalogues[mock_obs_type]
(omega_m_fid, A_s_fid, A_sz_fid, alpha_sz_fid, beta_sz_fid, b_WL0_fid, b_WLz_fid, b_WLM_fid) = fiducials[mock_obs_type]

# ── Run config ─────────────────────────────────────────────────────────
resume_run     = True
max_samples    = 5000
proposal_scale = 2.38
# mpirun -n 8 python run_mcmc.py
#CORE_OFFSET=0 TOMOGRAPHIC_BINS=DES        mpirun -n 8 python run_mcmc.py &
#CORE_OFFSET=8 TOMOGRAPHIC_BINS=deepEuclid mpirun -n 8 python run_mcmc.py &
# pkill -TERM -f 'mpirun -n 8 python run_mcmc.py'
#CORE_OFFSET=8 TOMOGRAPHIC_BINS=DES mpirun -n 8 python run_mcmc.py &

output_name = output_names[mock_obs_type]

# ── Fiducial / fixed params ────────────────────────────────────────────
h_fid        = 0.68
omega_b_fid  = 0.0226576
ns_fid       = 0.965

# ── likelihood wrapper ──────────────────────────────────────────
ln_L0, nc = get_log_likelihood(
                catalogue_name=CATALOGUE_PATH,
                omega_m=omega_m_fid,
                A_s=A_s_fid,
                A_sz=A_sz_fid,
                alpha_sz=alpha_sz_fid,
                beta_sz=beta_sz_fid,
                b_WL_0=b_WL0_fid,
                b_WL_z=b_WLz_fid,
                b_WL_M=b_WLM_fid,
                return_nc=True,
            )
def build_cosmo_params(omega_m, A_s):
    z_table, D_grid = get_D_grid_interp(omega_m)
    return {
        "Om0": omega_m / h_fid**2, "Ob0": omega_b_fid / h_fid**2,
        "h": h_fid, "H0": h_fid * 100., "A_s": A_s, "n_s": ns_fid,
        "m_nu": 0.0, "Onu0": 0.0, "tau_reio": 0.06, "Pk_scaling_for_sigmaM": 1.,
        'm_ncdm': 0.0, 'N_ncdm': 0, 'N_ur': 3.046, 
        "G_a_precomputed": get_G_a_interp(omega_m),
        "D_grid_z_precomputed": z_table, "D_grid_precomputed": D_grid,
        "normalisation_precomputed": get_norm_interp(omega_m),
        "Sigma_crit_vec": get_Sigma_crit_arr(omega_m),
        "n_mass_points_coarse": 300,
    }
def build_scal_rel_params(A_sz, alpha_sz, beta_sz, b_WL_0, b_WL_M, b_WL_z):
    return {
        "A_szifi": A_sz, "alpha_szifi": alpha_sz, "beta_szifi": beta_sz,
        "bias_sz": 0.8, "sigma_lnq_szifi": 0.173, "dof": 0.0, "q_cutoff": 0.4,
        "b_WL_0": b_WL_0, "b_WL_M": b_WL_M,  "b_WL_z": b_WL_z,
        "M_WL_pivot": 2e14 / h_fid, "z_WL_pivot": 0.4, "h_fid": h_fid,
    }
def fast_log_lik(omega_m, A_s, A_sz, alpha_sz, beta_sz, b_WL_0, b_WL_M, b_WL_z):
    nc.update_params(
        build_cosmo_params(omega_m, A_s),
        build_scal_rel_params(A_sz, alpha_sz, beta_sz, b_WL_0, b_WL_M, b_WL_z))
    nc._static_lnM_bounds = None
    nc.get_hmf(return_profile_params=True)
    return float(nc.get_log_lik())

# ── Cobaya likelihood wrapper ──────────────────────────────────────────
class ShearCNCLikelihood(Likelihood):

    def initialize(self):
            
        with open(CATALOGUE_PATH, "rb") as f:
            self.wl_cat = pickle.load(f)
        self.cat_name = CATALOGUE_PATH
        
        # Warm-up JIT + cost estimate
        n_time = 3
        t0 = time.time()
        for _ in range(n_time):
            fast_log_lik(
                omega_m         = omega_m_fid,
                A_s             = A_s_fid,
                A_sz            = A_sz_fid,
                alpha_sz        = alpha_sz_fid,
                beta_sz         = beta_sz_fid,
                b_WL_0          = b_WL0_fid,
                b_WL_M          = b_WLM_fid,
                b_WL_z          = b_WLz_fid,
            )
        t_per = (time.time() - t0) / n_time
        t_est = t_per / 0.05 * max_samples
        print(f"Single theory call : {t_per:.2f}s")
        print(f"Estimated for {max_samples} samples "
              f"(5% acceptance): {t_est/60:.1f} min  ({t_est/3600:.2f} hr)")
        print("Likelihood initialised and JIT warmed up.")
        
    def get_requirements(self):
        return {}  # standalone — no Boltzmann solver needed

    def logp(self, omega_m, A_s, A_sz, alpha_sz, beta_sz, b_WL_0, b_WL_M, b_WL_z, **kwargs):
        omega_m  = self.provider.get_param("omega_m")
        A_s      = self.provider.get_param("A_s")
        A_sz     = self.provider.get_param("A_sz")
        alpha_sz = self.provider.get_param("alpha_sz")
        beta_sz  = self.provider.get_param("beta_sz")
        b_WL_0   = self.provider.get_param("b_WL_0")
        b_WL_M   = self.provider.get_param("b_WL_M")
        b_WL_z   = self.provider.get_param("b_WL_z")
        try:
            return fast_log_lik(
                omega_m         = omega_m,
                A_s             = A_s,
                A_sz            = A_sz,
                alpha_sz        = alpha_sz,
                beta_sz         = beta_sz,
                b_WL_0          = b_WL_0,
                b_WL_M          = b_WL_M,
                b_WL_z          = b_WL_z,
            )
        except MemoryError:
            raise                      # never retry an OOM
        except Exception as e:
            msg = str(e).lower()
            if isinstance(e, MemoryError) or "out of memory" in msg \
                    or "resource_exhausted" in msg or "bad_alloc" in msg:
                print(f"[fatal] OOM in logp: {e}", flush=True)
                raise
            self._nfail = getattr(self, "_nfail", 0) + 1
            if self._nfail <= 5 or self._nfail % 200 == 0:
                print(f"[fail {self._nfail}] {type(e).__name__}: {e}  "
                      f"om={omega_m:.5f} As={A_s:.4e} Asz={A_sz:.4f} "
                      f"al={alpha_sz:.4f} be={beta_sz:.4f}", flush=True)
            return -np.inf


# ── Cobaya info dict ───────────────────────────────────────────────────
info = {
    "verbose": False,
    "debug":   False,
    "likelihood": {
        "shear_cnc": {
            "external": ShearCNCLikelihood,
            "input_params": [
                "omega_m",
                "A_s",
                "A_sz",
                "alpha_sz",
                "beta_sz",
                "b_WL_0",
                "b_WL_M",
                "b_WL_z",
            ],
        }
    },
    "params": {
        "omega_m": {
            "prior":    {"dist": "uniform", "min": 0.05, "max": 0.25},
            "ref":      {"dist": "norm", "loc": omega_m_fid,  "scale": 0.008},
            "proposal": 0.008 / 10,
            "latex":    r"\omega_m",
        },
        "A_s": {
            "prior":    {"dist": "uniform", "min": 0.2e-9, "max": 4.0e-9},
            "ref":      {"dist": "norm", "loc": A_s_fid,  "scale": 0.02e-9},
            "proposal": 0.02e-9 / 4,
            "latex":    r"A_\mathrm{s}",
        },
        "A_sz": {
            "prior":    {"dist": "uniform", "min": -7.0, "max": -3.0},
            "ref":      {"dist": "norm", "loc": A_sz_fid,     "scale": 0.32},
            "proposal": 0.32 / 10,
            "latex":    r"A_\mathrm{SZ}",
        },
        "alpha_sz": {
            "prior":    {"dist": "uniform", "min": 0.8, "max": 2.0},
            "ref":      {"dist": "norm", "loc": alpha_sz_fid, "scale": 0.12},
            "proposal": 0.12 / 10,
            "latex":    r"\alpha_\mathrm{SZ}",
        },
        "beta_sz": {
            "prior":    {"dist": "uniform", "min": 0.0, "max": 5.0},
            "ref":      {"dist": "norm", "loc": beta_sz_fid,  "scale": 0.16},
            "proposal": 0.16 / 10,
            "latex":    r"\beta_\mathrm{SZ}",
        },
        "b_WL_0": {
            "prior":    {"dist": "norm", "loc": b_WL0_fid, "scale": 0.02},
            "ref":      {"dist": "norm", "loc": b_WL0_fid, "scale": 0.01},
            "proposal": 0.01 / 10,
            "latex":    r"b_{\mathrm{WL},0}",
        },
        "b_WL_M": {
            "prior":    {"dist": "norm", "loc": b_WLM_fid, "scale": 0.02},
            "ref":      {"dist": "norm", "loc": b_WLM_fid, "scale": 0.01 },
            "proposal": 0.01 / 10,
            "latex":    r"b_{\mathrm{WL},M}",
        },
        "b_WL_z": {
            "prior":    {"dist": "norm", "loc": b_WLz_fid, "scale": 0.02},
            "ref":      {"dist": "norm", "loc": b_WLz_fid, "scale": 0.01},
            "proposal": 0.01 / 10,
            "latex":    r"b_{\mathrm{WL},z}",
        },
    },
    "sampler": {"mcmc": {
        "burn_in":   0,
        "max_samples": max_samples,
        "learn_proposal": True,
        "learn_every": "3d",
        "max_tries": 5000,
        "learn_proposal_Rminus1_max":       30.0,
        "learn_proposal_Rminus1_max_early": 1000.0,
        "proposal_scale": proposal_scale,
        "Rminus1_stop":    0.05,
        "Rminus1_cl_stop": 0.2,
        "covmat": initial_covmats[mock_obs_type],
        "measure_speeds": True,
    }},
    "output": output_name,
    "resume": resume_run,
}


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":    
    import resource
    _RAM_BYTES = int(_RAM_PER_CHAIN_GB * 1024**3)
    resource.setrlimit(resource.RLIMIT_DATA, (_RAM_BYTES, _RAM_BYTES))
    resource.setrlimit(resource.RLIMIT_AS, (_RAM_BYTES, _RAM_BYTES))
    print(f"RAM limit set to {_RAM_PER_CHAIN_GB} GB per process")

    with threadpool_limits(limits=int(_NT)):
        t0 = time.time()
        updated_info, sampler = run(info)
        print(f"Done in {(time.time()-t0)/60:.1f} min")