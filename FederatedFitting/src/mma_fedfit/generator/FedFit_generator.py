import torch
from typing import Tuple, Dict, Optional, Any
from omegaconf import DictConfig
import emcee
import afterglowpy as grb
import numpy as np
from multiprocessing import Pool
from emcee import moves as mvs
import gc
import os
import pickle
# from mma_fedfit.generator.inference_utils import *
from .inference_utils import *
from astropy.cosmology import Planck18 as cosmo
import math



class LocalGenerator():
    """
    LocalGenerator:
        LocalGenerator for FL clients, which computes/generates the local posterior samples using the given data
    """  
    def __init__(
        self,
        client_agent_config: DictConfig = DictConfig({}),
        logger: Optional[Any]=None,
        **kwargs
    ):
        
        self.client_agent_config = client_agent_config
        self.logger = logger
        self.__dict__.update(kwargs)
                                

    def get_parameters(self) -> Dict:
        if self.client_agent_config.fitting_configs.use_approach==2:
            return
            # return {
            #     'train_embedding': self.train_embedding.detach().clone().cpu(),
            #     'val_embedding': self.val_embedding.cpu(),
            # }
        else:
            
            # return {
            #     'chain': self.chain,
            #     'min_time': self.min_time,
            #     'max_time': self.max_time,
            #     'unique_frequencies': self.freqs
            # }
            return {
                'chain': self.chain
            }


    #-------------------------------------------------
    # GRB light-curve likelihood, prior, probability
    #-------------------------------------------------
    def log_likelihood(self, theta, nu, x, y, yerr):
        z_known = bool(self.client_agent_config.mcmc_configs.Z_known)
        z_fixed = int(self.client_agent_config.mcmc_configs.Z_fixed)

        if z_known == True:
            logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing = theta
            z = z_fixed
        else:
            logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, z = theta

        DL = cosmo.luminosity_distance(z).to(u.cm)
        DL = DL.value


        E0 = 10 ** logE0
        n0 = 10 ** logn0
        epsilon_e = 10 ** logepsilon_e
        epsilon_B = 10 ** logepsilon_B


        Z = {
        "jetType": grb.jet.Gaussian,  # Jet type
        "specType": 0,  # Basic Synchrotron Emission Spectrum
        "thetaObs": thetaObs, #35.2/180.0 * math.pi,  # Viewing angle in radians
        "E0": E0, #1.5e53*9.8e-3,  # Isotropic-equivalent energy in erg
        "thetaCore": thetaCore, #0.25,  # Half-opening angle in radians
        "thetaWing": thetaWing,  # "wing" truncation angle of the jet, in radians
        "n0": n0, #9.8e-3,  # circumburst density in cm^-3
        "p": p, #2.168,  # electron energy distribution index
        #"b": 6.0,  # power law structure index
        "epsilon_e": epsilon_e, #7.8e-3,  # epsilon_e
        "epsilon_B": epsilon_B, #9.9e-4,  # epsilon_B
        "xi_N": 1.0,  # Fraction of electrons accelerated
        "d_L": DL,  # Luminosity distance in cm of 40Mpc
        "z": z, #40Mpc
        }
        
        try:    

            model = grb.fluxDensity(x, nu, **Z)
            sigma2 = yerr**2
            return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log10(sigma2))
        
        except:

            #Just put in different parameters to make it run.
            print('enters except block')

            Z = {
            "jetType": grb.jet.Gaussian,  # Jet type
            "specType": 0,  # Basic Synchrotron Emission Spectrum
            "thetaObs": math.pi/4.0, #35.2/180.0 * math.pi,  # Viewing angle in radians
            "E0": 1e52, #1.5e53*9.8e-3,  # Isotropic-equivalent energy in erg
            "thetaCore": math.pi/10.0, #0.25,  # Half-opening angle in radians
            "thetaWing": math.pi/3.0,  # "wing" truncation angle of the jet, in radians
            "n0": 5e-3, #9.8e-3,  # circumburst density in cm^-3
            "p": 2.5, #2.168,  # electron energy distribution index
            #"b": 6.0,  # power law structure index
            "epsilon_e": .1, #7.8e-3,  # epsilon_e
            "epsilon_B": .01, #9.9e-4,  # epsilon_B
            "xi_N": 1.0,  # Fraction of electrons accelerated
            "d_L": 1.2344e26,  # Luminosity distance in cm of 40Mpc
            "z": 0.00897, #40Mpc
            }

            model = grb.fluxDensity(x, nu, **Z)
            sigma2 = yerr**2
            return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log10(sigma2))

    
    def ensure_float_list(self, lst):
        """
        Convert a list or tuple of values to a list of floats.
        If any value is not convertible, it raises a ValueError.
        """
        if not isinstance(lst, (list, tuple)):
            raise TypeError("Input must be a list or tuple.")
        
        try:
            return [float(x) for x in lst]
        except ValueError as e:
            raise ValueError(f"Failed to convert one or more elements to float in {lst}") from e
        
    def log_prior(self, theta):
        z_known = bool(self.client_agent_config.mcmc_configs.Z_known)

        if z_known:
            logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing = theta
        else:
            logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, z = theta

        loge0_range = self.ensure_float_list(self.client_agent_config.mcmc_configs.logE0_range)
        thetaobs_range = self.ensure_float_list(self.client_agent_config.mcmc_configs.thetaObs_range)
        thetacore_range = self.ensure_float_list(self.client_agent_config.mcmc_configs.thetaCore_range)
        logn0_range = self.ensure_float_list(self.client_agent_config.mcmc_configs.logn0_range)
        logepsilon_e_range = self.ensure_float_list(self.client_agent_config.mcmc_configs.logEpsilon_e_range)
        logepsilon_b_range = self.ensure_float_list(self.client_agent_config.mcmc_configs.logEpsilon_B_range)
        p_range = self.ensure_float_list(self.client_agent_config.mcmc_configs.P_range)
        thetawing_range = self.ensure_float_list(self.client_agent_config.mcmc_configs.thetaWing_range)
        z_range = self.ensure_float_list(self.client_agent_config.mcmc_configs.Z_range)
        
        if z_known:
            if (
                loge0_range[0] <= logE0 <= loge0_range[1]
                and thetaobs_range[0] <= thetaObs < thetaobs_range[1]
                and thetawing_range[0] <= thetaWing < thetawing_range[1]
                and thetacore_range[0] <= thetaCore < thetacore_range[1]
                and p_range[0] < p < p_range[1]
                and logepsilon_e_range[0] < logepsilon_e <= logepsilon_e_range[1]
                and logepsilon_b_range[0] < logepsilon_B <= logepsilon_b_range[1]
                and logn0_range[0] < logn0 < logn0_range[1]
            ):
                return 0.0
            return -np.inf

        else:
            if (
                loge0_range[0] <= logE0 <= loge0_range[1]
                and thetaobs_range[0] <= thetaObs < thetaobs_range[1]
                and thetawing_range[0] <= thetaWing < thetawing_range[1]
                and thetacore_range[0] <= thetaCore < thetacore_range[1]
                and p_range[0] < p < p_range[1]
                and logepsilon_e_range[0] < logepsilon_e <= logepsilon_e_range[1]
                and logepsilon_e_range[0] < logepsilon_B <= logepsilon_e_range[1]
                and logn0_range[0] < logn0 < logn0_range[1]
                and z_range[0] < z < z_range[1]
            ):
                return 0.0
            return -np.inf


    def log_probability(self, theta, nu, x, y, yerr):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, nu, x, y, yerr)
    
    #-------------------------------------------------
    # LOCAL MCMC FUNCTION FOR EACH SITE
    #-------------------------------------------------
    def save_sampler_state(self, sampler, filename):
        with open(filename, "wb") as f:
            pickle.dump(sampler, f)
            
    def run_local_mcmc(self, local_data, pos, niter, nwalkers, ndim):
        # Extract arrays from this site's data subset
        t = np.array(local_data["t"])
        nu = np.array(local_data["frequency"])
        fnu = np.array(local_data["flux"])
        err = np.array(local_data["err"])
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, args=(nu, t, fnu, err), pool=pool,
            moves=[(mvs.StretchMove(a=1.1), 0.7), (mvs.WalkMove(10), 0.3)])
            # sampler.run_mcmc(pos, niter, progress=True)
        
            state = sampler.run_mcmc(pos, niter, progress=True)

        gc.collect()

        # Save local results
        site_folder = self.client_agent_config.fitting_configs.save_folder
        os.makedirs(site_folder, exist_ok=True)

        # Get flat samples after burn-in
        burnin = int(self.client_agent_config.mcmc_configs.burnin)
        run_name = self.client_agent_config.fitting_configs.run_name

        flat_samples = sampler.get_chain(discard=burnin, flat=True)
        np.save(f"{site_folder}/{run_name}_flat_samples.npy", flat_samples)
        
        # Save log probability
        log_prob = sampler.get_log_prob(flat=False)
        np.save(f"{site_folder}/{run_name}_log_prob.npy", log_prob)

        # Save sampler state
        self.save_sampler_state(sampler, f"{site_folder}/sampler.pkl")

        # Create local parameter summary - this will be sent to the server
        # mean = np.mean(flat_samples, axis=0)
        # cov = np.cov(flat_samples, rowvar=False)
        
        # Create plots for local site
        z_known = bool(self.client_agent_config.mcmc_configs.Z_known)

        if z_known:
            params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(eps_e)','log(eps_B)','p', 'thetaWing']
        else:
            params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(eps_e)','log(eps_B)','p', 'thetaWing', 'z']
        
        # Plotting function present in generator.inference_utils
        make_posterior_hists(flat_samples, burnin, nwalkers, ndim, params, f"{site_folder}/{run_name}_PosteriorHists.png")
        
        true_values = self.client_agent_config.mcmc_configs.true_values
        display_truths_on_corner = self.client_agent_config.mcmc_configs.display_truths_on_corner
        make_corner_plots(flat_samples, burnin, nwalkers, ndim, params, true_values, display_truths_on_corner, f"{site_folder}/{run_name}_CornerPlots.png")
        make_Log_Likelihood_plot(log_prob, burnin, nwalkers, f"{site_folder}/{run_name}_LogProb.png")

    # Return statistics for consensus (not raw data)
    # return mean, cov

        # Discard burn-in and thin the chain
        # self.chain = sampler.get_chain(discard=10, thin=2, flat=True)
        self.chain = sampler.get_chain(discard=burnin, flat=True)

        print(f"Chain data type: {self.chain.dtype}", flush=True)
        # Compute summary statistics to send (without sending raw data).
        # self.min_time = local_data["t"].min()
        # self.max_time = local_data["t"].max()
        # self.freqs = local_data["frequency"].unique()

    
    def compute_local_log_likelihood(self, theta, site_data):
        """
        Function that takes proposed theta from server and computes local log-likelihood
        """

        # Extract site data arrays.
        t = np.array(site_data["t"])
        nu = np.array(site_data["frequency"])
        fnu = np.array(site_data["flux"])
        err = np.array(site_data["err"])
        return self.log_likelihood(theta, nu, t, fnu, err)