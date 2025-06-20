import torch
from typing import Tuple, Dict, Optional, Any
from omegaconf import DictConfig, OmegaConf, ListConfig
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
        client_agent_config,
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
        z_known = self.processed_mcmc_configs.Z_known
        z_fixed = self.processed_mcmc_configs.Z_fixed

        dl_known = self.processed_mcmc_configs.DL_known
        dl_fixed = self.processed_mcmc_configs.DL_fixed

        epsilon_e_fixed = self.processed_mcmc_configs.epsilon_e_fixed  # loge = -2.0
        epsilon_b_fixed = self.processed_mcmc_configs.epsilon_b_fixed #logb = -3.7
        thetacore_fixed = self.processed_mcmc_configs.thetacore_fixed
        thetawing_fixed = self.processed_mcmc_configs.thetawing_fixed
        p_fixed = self.processed_mcmc_configs.p_fixed

        # if z_known == True and dl_known == True:
        #     logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing = theta
        #     z = z_fixed
        #     DL = dl_fixed 

        # elif z_known == True and dl_known == False:
        #     logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, DL = theta
        #     z = z_fixed

        # elif z_known == False and dl_known == True:
        #     logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, z = theta
        #     DL = dl_fixed
        # else:
        #     logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, z, DL = theta

        if z_known == True and dl_known == True:
            logE0, thetaObs, logn0 = theta  # REMOVED thetaCore, p, thetaWing, logepsilon_e, logepsilon_B
            z = z_fixed
            DL = dl_fixed

        elif z_known == True and dl_known == False:
            logE0, thetaObs, logn0, DL = theta  # REMOVED thetaCore, p, thetaWing, logepsilon_e, logepsilon_B
            z = z_fixed

        elif z_known == False and dl_known == True:
            logE0, thetaObs, logn0, z = theta  # REMOVED thetaCore, p, thetaWing, logepsilon_e, logepsilon_B
            DL = dl_fixed

        else:
            logE0, thetaObs, logn0, z, DL = theta  # REMOVED thetaCore, p, thetaWing, logepsilon_e, logepsilon_B

        # DL = cosmo.luminosity_distance(z).to(u.cm)
        # DL = DL.value

        DL = (DL*u.Mpc).to(u.cm).value #Turn it to cm for the fitting



        E0 = 10 ** logE0
        n0 = 10 ** logn0
        # epsilon_e = 10 ** logepsilon_e
        # epsilon_B = 10 ** logepsilon_B

        # USE FIXED VALUES INSTEAD OF FITTING
        epsilon_e = epsilon_e_fixed
        epsilon_B = epsilon_b_fixed
        thetaCore = thetacore_fixed  # NEW FIXED VALUE
        p = p_fixed                  # NEW FIXED VALUE
        thetaWing = thetawing_fixed  # NEW FIXED VALUE


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

            
    def log_prior(self, theta):
        z_known = self.processed_mcmc_configs.Z_known
        dl_known = self.processed_mcmc_configs.DL_known

        if z_known == True and dl_known == True:
            logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing = theta

        elif z_known == True and dl_known == False:
            logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, DL = theta

        elif z_known == False and dl_known == True:
            logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, z = theta
            
        else:
            logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, z, DL = theta

        loge0_range = self.processed_mcmc_configs.logE0_range
        thetaobs_range = self.processed_mcmc_configs.thetaObs_range
        thetacore_range = self.processed_mcmc_configs.thetaCore_range
        logn0_range = self.processed_mcmc_configs.logn0_range
        logepsilon_e_range = self.processed_mcmc_configs.logEpsilon_e_range
        logepsilon_b_range = self.processed_mcmc_configs.logEpsilon_B_range
        p_range = self.processed_mcmc_configs.P_range
        thetawing_range = self.processed_mcmc_configs.thetaWing_range
        z_range = self.processed_mcmc_configs.Z_range
        dl_range = self.processed_mcmc_configs.DL_range

        
        if z_known == True and dl_known == True:
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
        
        elif z_known == True and dl_known == False:
            if (
                loge0_range[0] <= logE0 <= loge0_range[1]
                and thetaobs_range[0] <= thetaObs < thetaobs_range[1]
                and thetawing_range[0] <= thetaWing < thetawing_range[1]
                and thetacore_range[0] <= thetaCore < thetacore_range[1]
                and p_range[0] < p < p_range[1]
                and logepsilon_e_range[0] < logepsilon_e <= logepsilon_e_range[1]
                and logepsilon_b_range[0] < logepsilon_B <= logepsilon_b_range[1]
                and logn0_range[0] < logn0 < logn0_range[1]
                and dl_range[0] < DL < dl_range[1]
            ):
                return 0.0
            return -np.inf

        elif z_known == False and dl_known == True:
            if (
                loge0_range[0] <= logE0 <= loge0_range[1]
                and thetaobs_range[0] <= thetaObs < thetaobs_range[1]
                and thetawing_range[0] <= thetaWing < thetawing_range[1]
                and thetacore_range[0] <= thetaCore < thetacore_range[1]
                and p_range[0] < p < p_range[1]
                and logepsilon_e_range[0] < logepsilon_e <= logepsilon_e_range[1]
                and logepsilon_b_range[0] < logepsilon_B <= logepsilon_b_range[1]
                and logn0_range[0] < logn0 < logn0_range[1]
                and z_range[0] < z < z_range[1]
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
                and logepsilon_b_range[0] < logepsilon_B <= logepsilon_b_range[1]
                and logn0_range[0] < logn0 < logn0_range[1]
                and z_range[0] < z < z_range[1]
                and dl_range[0] < DL < dl_range[1]
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
            
    def run_local_mcmc(self, local_data):
        
        self.process_mcmc_config()

        # For boolean values
        z_known = self.processed_mcmc_configs.Z_known
        dl_known = self.processed_mcmc_configs.DL_known


        ndim = 8 if z_known else 9
        nwalkers = self.processed_mcmc_configs.nwalkers
        niters = self.processed_mcmc_configs.niters
        
        # Safe extraction of configuration values
        logn0_range = self.processed_mcmc_configs.logn0_range
        logepsilon_e_range = self.processed_mcmc_configs.logEpsilon_e_range
        logepsilon_b_range = self.processed_mcmc_configs.logEpsilon_B_range
        p_range = self.processed_mcmc_configs.P_range
        thetawing_range = self.processed_mcmc_configs.thetaWing_range
        z_range = self.processed_mcmc_configs.Z_range
        dl_range = self.processed_mcmc_configs.DL_range
        thetaobs_range = self.processed_mcmc_configs.thetaObs_range
        thetacore_range = self.processed_mcmc_configs.thetaCore_range
        loge0_range = self.processed_mcmc_configs.logE0_range


        # Define parameter bounds
        if z_known == True and dl_known == True:
            ndim = 8
            lower_bounds = np.array([
                loge0_range[0],        # log10(E0)
                thetaobs_range[0],     # thetaObs
                thetacore_range[0],    # thetaCore
                logn0_range[0],        # log10(n0)
                logepsilon_e_range[0], # log10(eps_e)
                logepsilon_b_range[0], # log10(eps_B)
                p_range[0],            # p
                thetawing_range[0]     # thetaWing 
            ])
            
            upper_bounds = np.array([
                loge0_range[1],        # log10(E0)
                thetaobs_range[1],     # thetaObs
                thetacore_range[1],    # thetaCore
                logn0_range[1],        # log10(n0)
                logepsilon_e_range[1], # log10(eps_e)
                logepsilon_b_range[1], # log10(eps_B)
                p_range[1],            # p
                thetawing_range[1]     # thetaWing
            ])
        elif z_known == True and dl_known == False:
            ndim = 9
            lower_bounds = np.array([
                loge0_range[0],        # log10(E0)
                thetaobs_range[0],       # thetaObs
                thetacore_range[0],      # thetaCore
                logn0_range[0],        # log10(n0)
                logepsilon_e_range[0],        # log10(eps_e)
                logepsilon_b_range[0],        # log10(eps_B)
                p_range[0],       # p
                thetawing_range[0],        # thetaWing
                dl_range[0]
            ])
            
            upper_bounds = np.array([
                loge0_range[1],        # log10(E0)
                thetaobs_range[1],       # thetaObs
                thetacore_range[1],      # thetaCore
                logn0_range[1],        # log10(n0)
                logepsilon_e_range[1],        # log10(eps_e)
                logepsilon_b_range[1],        # log10(eps_B)
                p_range[1],       # p
                thetawing_range[1],        # thetaWing
                dl_range[1]
            ])

        elif z_known == False and dl_known == True:
            ndim = 9
            lower_bounds = np.array([
                loge0_range[0],        # log10(E0)
                thetaobs_range[0],       # thetaObs
                thetacore_range[0],      # thetaCore
                logn0_range[0],        # log10(n0)
                logepsilon_e_range[0],        # log10(eps_e)
                logepsilon_b_range[0],        # log10(eps_B)
                p_range[0],       # p
                thetawing_range[0],        # thetaWing
                z_range[0]
            ])
            
            upper_bounds = np.array([
                loge0_range[1],        # log10(E0)
                thetaobs_range[1],       # thetaObs
                thetacore_range[1],      # thetaCore
                logn0_range[1],        # log10(n0)
                logepsilon_e_range[1],        # log10(eps_e)
                logepsilon_b_range[1],        # log10(eps_B)
                p_range[1],       # p
                thetawing_range[1],        # thetaWing
                z_range[1]
            ])

        else:
            ndim = 10
            lower_bounds = np.array([
                loge0_range[0],        # log10(E0)
                thetaobs_range[0],       # thetaObs
                thetacore_range[0],      # thetaCore
                logn0_range[0],        # log10(n0)
                logepsilon_e_range[0],        # log10(eps_e)
                logepsilon_b_range[0],        # log10(eps_B)
                p_range[0],       # p
                thetawing_range[0],        # thetaWing
                z_range[0],
                dl_range[0]
            ])
            
            upper_bounds = np.array([
                loge0_range[1],        # log10(E0)
                thetaobs_range[1],       # thetaObs
                thetacore_range[1],      # thetaCore
                logn0_range[1],        # log10(n0)
                logepsilon_e_range[1],        # log10(eps_e)
                logepsilon_b_range[1],        # log10(eps_B)
                p_range[1],       # p
                thetawing_range[1],        # thetaWing
                z_range[1],
                dl_range[1] 
            ])


        np.random.seed(self.processed_mcmc_configs.random_seed)

        # Prepare initial walker positions (in 7 dimensions).
        # pos = ( [ np.log10(Z["E0"]),
        #             Z["thetaObs"],
        #             Z["thetaCore"],
        #             np.log10(Z["n0"]),
        #             np.log10(Z["epsilon_e"]),
        #             np.log10(Z["epsilon_B"]),
        #             Z["p"] ] 
        #         + 0.005 * np.random.randn(nwalkers, ndim) )

        # position it locally and uniformly
        pos = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(nwalkers, ndim))

        # Extract arrays from this site's data subset
        t = np.array(local_data["t"])
        nu = np.array(local_data["frequency"])
        fnu = np.array(local_data["flux"])
        err = np.array(local_data["err"])
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, args=(nu, t, fnu, err), pool=pool,
            moves=[(mvs.StretchMove(a=1.1), 0.7), (mvs.WalkMove(10), 0.3)])
            # sampler.run_mcmc(pos, niter, progress=True)
        
            state = sampler.run_mcmc(pos, niters, progress=True)

        gc.collect()

        # Save local results
        site_folder = self.client_agent_config.fitting_configs.save_folder
        os.makedirs(site_folder, exist_ok=True)

        # Get flat samples after burn-in
        burnin = self.processed_mcmc_configs.burnin
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
        z_known = self.processed_mcmc_configs.Z_known
        dl_known = self.processed_mcmc_configs.DL_known


        if z_known == True and dl_known == True:
            params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(eps_e)','log(eps_B)','p', 'thetaWing']

        elif z_known == False and dl_known == True:
            params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(eps_e)','log(eps_B)','p', 'thetaWing', 'z']

        elif z_known == True and dl_known == False:
            params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(eps_e)','log(eps_B)','p', 'thetaWing', 'DL']

        else:
            params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(eps_e)','log(eps_B)','p', 'thetaWing', 'z', 'DL']

        print('Local Site parameter values', flush=True)
        # print("Best estimate of parameters", flush=True)
        self.logger.info("Local site: Best estimate of parameters")
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            """
            if i in [0, 3, 4, 5]:
                theta.append(10 ** mcmc[1])
            else:
                theta.append(mcmc[1])
            """
            #keep it in log space
            q = np.diff(mcmc)
            print(f'{params[i]} = {mcmc[1]:.4f} +{q[1]:.4f} -{q[0]:.4f}')
            self.logger.info(f'{params[i]} = {mcmc[1]:.4f} +{q[1]:.4f} -{q[0]:.4f}')

        # Plotting function present in generator.inference_utils
        make_posterior_hists(flat_samples, burnin, nwalkers, ndim, params, f"{site_folder}/{run_name}_PosteriorHists.png")
        
        true_values = self.client_agent_config.mcmc_configs.true_values
        display_truths_on_corner = self.processed_mcmc_configs.display_truths_on_corner
        make_corner_plots(flat_samples, burnin, nwalkers, ndim, params, true_values, display_truths_on_corner, f"{site_folder}/{run_name}_CornerPlots.png")
        
        plot_names = self.client_agent_config.fitting_configs.plot_names
        make_Log_Likelihood_plot(log_prob, burnin, nwalkers, plot_names, f"{site_folder}/{run_name}_LogProb.png")

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

        self.process_mcmc_config()

        # Extract site data arrays.
        t = np.array(site_data["t"])
        nu = np.array(site_data["frequency"])
        fnu = np.array(site_data["flux"])
        err = np.array(site_data["err"])
        return self.log_likelihood(theta, nu, t, fnu, err)
    
    def process_mcmc_config(self):
        """
        Process the YAML configuration loaded via OmegaConf to ensure proper types.
        This function is meant to be used directly in your existing code.
        
        Args:
            client_agent: Your client agent object with config loaded via OmegaConf
            
        Returns:
            A dictionary with processed configuration values
        """
        # Get mcmc_configs from the client agent
        mcmc_configs = self.client_agent_config.mcmc_configs
        
        # Create a dictionary to store processed values
        self.processed_mcmc_configs = {}
        
        # Process boolean values
        boolean_keys = [
            'Z_known', 'DL_known','include_upper_limits_on_lc', 
            'exclude_time_flag', 'exclude_ra_dec_flag', 
            'exclude_name_flag', 'exclude_wrong_name',
            'exclude_outside_ra_dec_uncertainty', 
            'display_truths_on_corner'
        ]
        
        for key in boolean_keys:
            if hasattr(mcmc_configs, key):
                val = getattr(mcmc_configs, key)
                if isinstance(val, str):
                    self.processed_mcmc_configs[key] = val.lower() == 'true'
                else:
                    self.processed_mcmc_configs[key] = bool(val)
        
        # Process range values
        range_keys = [
            'Z_range', 'DL_range', 'thetaObs_range', 'thetaCore_range',
            'P_range', 'thetaWing_range', 'logE0_range',
            'logn0_range', 'logEpsilon_e_range', 'logEpsilon_B_range'
        ]
        
        for key in range_keys:
            if hasattr(mcmc_configs, key):
                val = getattr(mcmc_configs, key)

                # Convert ListConfig to native list
                if isinstance(val, ListConfig):
                    val = list(val)
                if isinstance(val, list):
                    self.processed_mcmc_configs[key] = [float(x) for x in val]
                else:
                    # Handle case where range might be represented as a string "[a, b]"
                    try:
                        if isinstance(val, str):
                            self.processed_mcmc_configs[key] = [float(x.strip()) for x in val.strip("[]").split(",")]
                        else:
                            self.processed_mcmc_configs[key] = [float(val)]  # Single value case
                    except ValueError:
                        self.processed_mcmc_configs[key] = val  # Keep original if conversion fails
        
        # Process numeric values
        numeric_keys = [
            'nwalkers', 'niters', 'burnin', 'random_seed',
            'Z_fixed', 'DL_fixed', 'arcseconds_uncertainty', 'epsilon_e_fixed', 'epsilon_b_fixed', 'thetacore_fixed', 'thetawing_fixed', 'p_fixed'
        ]
        
        for key in numeric_keys:
            if hasattr(mcmc_configs, key):
                val = getattr(mcmc_configs, key)
                try:
                    if isinstance(val, str):
                        if '.' in val:
                            self.processed_mcmc_configs[key] = float(val)
                        else:
                            self.processed_mcmc_configs[key] = int(val)
                    else:
                        self.processed_mcmc_configs[key] = val  # Already numeric
                except ValueError:
                    self.processed_mcmc_configs[key] = val  # Keep original if conversion fails
        
        # to keep do notation
        self.processed_mcmc_configs = OmegaConf.create(self.processed_mcmc_configs)
        # return self.processed_mcmc_config