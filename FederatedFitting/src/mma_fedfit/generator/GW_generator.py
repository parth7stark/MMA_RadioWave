import torch
from typing import Tuple, Dict, Optional, Any
from omegaconf import DictConfig
import emcee
import afterglowpy as grb
import numpy as np
from multiprocessing import Pool

class LocalGenerator():
    """
    LocalGenerator:
        LocalGenerator for FL clients, which computes/generates the local posterior samples using the given data
    """  
    def __init__(
        self,
        model: torch.nn.Module=None,
        generator_configs: DictConfig = DictConfig({}),
        logger: Optional[Any]=None,
        **kwargs
    ):
        
        self.generator_configs = generator_configs
        self.logger = logger
        self.__dict__.update(kwargs)

        if not hasattr(self.generator_configs, "device"):
            self.generator_configs.device = "cpu"

        self.model.to(self.generator_configs.device)

                                

    def get_parameters(self) -> Dict:
        if self.generator_configs.use_approach==2:
            return
            # return {
            #     'train_embedding': self.train_embedding.detach().clone().cpu(),
            #     'val_embedding': self.val_embedding.cpu(),
            # }
        else:
            
            return {
                'chain': self.chain,
                'min_time': self.min_time,
                'max_time': self.max_time,
                'unique_frequencies': self.freqs
            }


    def compute_embeddings(self, inference_data):
        # 1 batch of windows
        with torch.no_grad():
            self.model.eval()
            # Move the data to appropriate device
            inference_tensor = inference_data.to(self.generator_configs.device)
            self.inference_embedding = self.model(inference_tensor)

    #-------------------------------------------------
    # GRB light-curve likelihood, prior, probability
    #-------------------------------------------------
    def log_likelihood(theta, nu, x, y, yerr):
        E0, thetaObs, thetaCore, n0, epsilon_e, epsilon_B, p = theta

        # Model parameters (from Makhathini 2021)
        Z = {
            "jetType": grb.jet.Gaussian,
            "specType": 0,
            "thetaObs": thetaObs,
            "E0": E0,
            "thetaCore": thetaCore,
            "thetaWing": 0.6,
            "n0": n0,
            "p": p,
            "epsilon_e": epsilon_e,
            "epsilon_B": epsilon_B,
            "xi_N": 1.0,
            "d_L": 1.2344e26,
            "z": 0.00897,
        }
        model = grb.fluxDensity(x, nu, **Z)
        sigma2 = yerr**2
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log10(sigma2))

    def log_prior(theta):
        E0, thetaObs, thetaCore, n0, epsilon_e, epsilon_B, p = theta
        if (0 <= thetaObs < np.pi * 0.5 and
            0.01 < thetaCore < np.pi * 0.5 and
            2 < p < 3 and
            0 < epsilon_e <= 1 and
            0 < epsilon_B <= 1 and
            0 < n0):
            return 0.0
        return -np.inf

    def log_probability(theta, nu, x, y, yerr):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, nu, x, y, yerr)
    
    #-------------------------------------------------
    # LOCAL MCMC FUNCTION FOR EACH SITE
    #-------------------------------------------------
    def run_local_mcmc(self, local_data, pos, niter, nwalkers, ndim):
        # Extract arrays from this site's data subset
        t = np.array(local_data["t"])
        nu = np.array(local_data["frequency"])
        fnu = np.array(local_data["flux"])
        err = np.array(local_data["err"])
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(nu, t, fnu, err), pool=pool)
            sampler.run_mcmc(pos, niter, progress=True)
        
        # Discard burn-in and thin the chain
        self.chain = sampler.get_chain(discard=10, thin=2, flat=True)
        
        # Compute summary statistics to send (without sending raw data).
        self.min_time = local_data["t"].min()
        self.max_time = local_data["t"].max()
        self.freqs = local_data["frequency"].unique()

    
    def compute_log_likelihood(self, theta, fitting_data):
        """
        Function that takes proposed theta from server and computes local log-likelihood
        """

        # Extract site data arrays.
        t = np.array(data_site["t"])
        nu = np.array(data_site["frequency"])
        fnu = np.array(data_site["flux"])
        err = np.array(data_site["err"])
        return self.log_likelihood(theta, nu, t, fnu, err)