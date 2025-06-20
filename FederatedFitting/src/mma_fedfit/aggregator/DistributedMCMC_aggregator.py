import torch
from omegaconf import DictConfig, OmegaConf, ListConfig
from typing import Any, Union, List, Dict, Optional
import numpy as np
import os
import matplotlib.pyplot as plt
import corner
from multiprocessing import Pool
import math
import pickle
import matplotlib.colors as mcolors
import gc

from emcee import moves as mvs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import shutil
# import afterglowpy as grb
import emcee
import threading

class DistributedMCMCAggregator():
    """
    DistributedMCMCAggregator:
        Aggregator for federated fitting, which takes in local log-likelihood from each site,
        concatenates them using distinuted MCMC method. The aggregator then uses the global posterior to estimate the best parameters.
        Best parameters are sended back to the clients
        for them to plot the light curve locally.
    """
    def __init__(
        self,
        server_agent_config: DictConfig = DictConfig({}),
        logger: Any | None = None,
    ):

        self.logger = logger
        self.server_agent_config = server_agent_config

        self.ongoing_iteration = 0

        # Our local aggregator: store partial local until we have all the sites
        # aggregator[site_id] = {local_chain: [...], local_time_range: [...], local_filters: [...]}

        self.aggregated_results={}

        # Track which clients have finished
        self.completed_clients = set()

        num_clients = self.server_agent_config.server_configs.aggregator_kwargs.num_clients  # Change this value as needed
        self.expected_clients = {str(i) for i in range(num_clients)}
        
        # for tagging each call
        self.iteration = 0
        self.walker_counter = 0
        self.lock = threading.Lock()

    #-------------------------------------------------
    # AGGREGATION FUNCTION: GAUSSIAN CONSENSUS
    #-------------------------------------------------
    def log_prior(self, theta):
        z_known = self.processed_mcmc_config.Z_known
        dl_known = self.processed_mcmc_config.DL_known


        # if z_known == True and dl_known == True:
        #     logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing = theta

        # elif z_known == True and dl_known == False:
        #     logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, DL = theta

        # elif z_known == False and dl_known == True:
        #     logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, z = theta
            
        # else:
        #     logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, z, DL = theta
    
        if z_known == True and dl_known == True:
            logE0, thetaObs, logn0 = theta  # REMOVED thetaCore, p, thetaWing, logepsilon_e, logepsilon_B

        elif z_known == True and dl_known == False:
            logE0, thetaObs, logn0, DL = theta  # REMOVED thetaCore, p, thetaWing, logepsilon_e, logepsilon_B

        elif z_known == False and dl_known == True:
            logE0, thetaObs, logn0, z = theta  # REMOVED thetaCore, p, thetaWing, logepsilon_e, logepsilon_B

        else:
            logE0, thetaObs, logn0, z, DL = theta  # REMOVED thetaCore, p, thetaWing, logepsilon_e, logepsilon_B



        loge0_range = self.processed_mcmc_config.logE0_range
        thetaobs_range = self.processed_mcmc_config.thetaObs_range
        thetacore_range = self.processed_mcmc_config.thetaCore_range
        logn0_range = self.processed_mcmc_config.logn0_range
        logepsilon_e_range = self.processed_mcmc_config.logEpsilon_e_range
        logepsilon_b_range = self.processed_mcmc_config.logEpsilon_B_range
        p_range = self.processed_mcmc_config.P_range
        thetawing_range = self.processed_mcmc_config.thetaWing_range
        z_range = self.processed_mcmc_config.Z_range
        dl_range = self.processed_mcmc_config.DL_range

        
        # if z_known == True and dl_known == True:
        #     if (
        #         loge0_range[0] <= logE0 <= loge0_range[1]
        #         and thetaobs_range[0] <= thetaObs < thetaobs_range[1]
        #         and thetawing_range[0] <= thetaWing < thetawing_range[1]
        #         and thetacore_range[0] <= thetaCore < thetacore_range[1]
        #         and p_range[0] < p < p_range[1]
        #         and logepsilon_e_range[0] < logepsilon_e <= logepsilon_e_range[1]
        #         and logepsilon_b_range[0] < logepsilon_B <= logepsilon_b_range[1]
        #         and logn0_range[0] < logn0 < logn0_range[1]
        #     ):
        #         return 0.0
        #     return -np.inf

        # elif z_known == True and dl_known == False:
        #     if (
        #         loge0_range[0] <= logE0 <= loge0_range[1]
        #         and thetaobs_range[0] <= thetaObs < thetaobs_range[1]
        #         and thetawing_range[0] <= thetaWing < thetawing_range[1]
        #         and thetacore_range[0] <= thetaCore < thetacore_range[1]
        #         and p_range[0] < p < p_range[1]
        #         and logepsilon_e_range[0] < logepsilon_e <= logepsilon_e_range[1]
        #         and logepsilon_b_range[0] < logepsilon_B <= logepsilon_b_range[1]
        #         and logn0_range[0] < logn0 < logn0_range[1]
        #         and dl_range[0] < DL < dl_range[1]
        #     ):
        #         return 0.0
        #     return -np.inf

        # elif z_known == False and dl_known == True:
        #     if (
        #         loge0_range[0] <= logE0 <= loge0_range[1]
        #         and thetaobs_range[0] <= thetaObs < thetaobs_range[1]
        #         and thetawing_range[0] <= thetaWing < thetawing_range[1]
        #         and thetacore_range[0] <= thetaCore < thetacore_range[1]
        #         and p_range[0] < p < p_range[1]
        #         and logepsilon_e_range[0] < logepsilon_e <= logepsilon_e_range[1]
        #         and logepsilon_b_range[0] < logepsilon_B <= logepsilon_b_range[1]
        #         and logn0_range[0] < logn0 < logn0_range[1]
        #         and z_range[0] < z < z_range[1]
        #     ):
        #         return 0.0
        #     return -np.inf

        # else:
        #     if (
        #         loge0_range[0] <= logE0 <= loge0_range[1]
        #         and thetaobs_range[0] <= thetaObs < thetaobs_range[1]
        #         and thetawing_range[0] <= thetaWing < thetawing_range[1]
        #         and thetacore_range[0] <= thetaCore < thetacore_range[1]
        #         and p_range[0] < p < p_range[1]
        #         and logepsilon_e_range[0] < logepsilon_e <= logepsilon_e_range[1]
        #         and logepsilon_e_range[0] < logepsilon_B <= logepsilon_e_range[1]
        #         and logn0_range[0] < logn0 < logn0_range[1]
        #         and z_range[0] < z < z_range[1]
        #         and dl_range[0] < DL < dl_range[1]
        #     ):
        #         return 0.0
        #     return -np.inf

        if z_known == True and dl_known == True:
            if (
                loge0_range[0] <= logE0 <= loge0_range[1]
                and thetaobs_range[0] <= thetaObs < thetaobs_range[1]
                and logn0_range[0] < logn0 < logn0_range[1]
                # REMOVED thetaCore, p, thetaWing, epsilon constraints
            ):
                return 0.0
            return -np.inf

        elif z_known == True and dl_known == False:
            if (
                loge0_range[0] <= logE0 <= loge0_range[1]
                and thetaobs_range[0] <= thetaObs < thetaobs_range[1]
                and logn0_range[0] < logn0 < logn0_range[1]
                and dl_range[0] < DL < dl_range[1]
                # REMOVED thetaCore, p, thetaWing, epsilon constraints
            ):
                return 0.0
            return -np.inf

        elif z_known == False and dl_known == True:
            if (
                loge0_range[0] <= logE0 <= loge0_range[1]
                and thetaobs_range[0] <= thetaObs < thetaobs_range[1]
                and logn0_range[0] < logn0 < logn0_range[1]
                and z_range[0] < z < z_range[1]
                # REMOVED thetaCore, p, thetaWing, epsilon constraints
            ):
                return 0.0
            return -np.inf

        else:
            if (
                loge0_range[0] <= logE0 <= loge0_range[1]
                and thetaobs_range[0] <= thetaObs < thetaobs_range[1]
                and logn0_range[0] < logn0 < logn0_range[1]
                and z_range[0] < z < z_range[1]
                and dl_range[0] < DL < dl_range[1]
                # REMOVED thetaCore, p, thetaWing, epsilon constraints
            ):
                return 0.0
            return -np.inf

        
    # Global log-probability function (aggregates local log-likelihoods)
    def global_log_probability(self, theta, num_sites):
        
        # lp = self.log_prior(theta)
        # if not np.isfinite(lp):
        #     return -np.inf

        # # Send MCMC tasks to sites
        # self.communicator.send_proposed_theta(theta, self.ongoing_iteration)

        # # Collect local log-likelihoods
        # log_likelihoods = self.communicator.collect_local_likelihoods(num_sites, self.ongoing_iteration)

        # self.ongoing_iteration = self.ongoing_iteration + 1
        # return lp + sum(log_likelihoods.values())

        # Atomically grab this call’s (iter, walker) identifiers
        with self.lock:
            iteration_no = self.iteration
            walker_no = self.walker_counter
            self.walker_counter += 1
            if self.walker_counter >= self.nwalkers:
                self.walker_counter = 0
                self.iteration += 1

        self.logger.info(f"[Server] Computing global log prob. Iteration no: {iteration_no}, walker={walker_no}")

        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf

        # 1) broadcast θ to all sites
        self.communicator.send_proposed_theta(theta, iteration_no, walker_no)

        # 2) wait until each site returns its local likelihood
        local_lls = self.communicator.collect_local_likelihoods(
            num_sites, iteration_no, walker_no
        )

        # 3) aggregate
        return lp + sum(local_lls.values())



    def save_sampler_state(self, sampler, filename):
        with open(filename, "wb") as f:
            pickle.dump(sampler, f)
            
    def run_distributed_MCMC(self, communicator):
        """Run Distributed MCMC workflow"""
        self.communicator = communicator
        self.process_mcmc_config()

        # Prepare MCMC parameters
        z_known = self.processed_mcmc_config.Z_known
        dl_known = self.processed_mcmc_config.DL_known

        ndim = 8 if z_known else 9
        self.nwalkers = self.processed_mcmc_config.nwalkers
        niters = self.processed_mcmc_config.niters

        loge0_range = self.processed_mcmc_config.logE0_range
        thetaobs_range = self.processed_mcmc_config.thetaObs_range
        thetacore_range = self.processed_mcmc_config.thetaCore_range
        logn0_range = self.processed_mcmc_config.logn0_range
        logepsilon_e_range = self.processed_mcmc_config.logEpsilon_e_range
        logepsilon_b_range = self.processed_mcmc_config.logEpsilon_B_range
        p_range = self.processed_mcmc_config.P_range
        thetawing_range = self.processed_mcmc_config.thetaWing_range
        z_range = self.processed_mcmc_config.Z_range
        dl_range = self.processed_mcmc_config.DL_range

        
        # Define parameter bounds
        # if z_known == True and dl_known == True:
        #     ndim = 8
        #     lower_bounds = np.array([
        #         loge0_range[0],        # log10(E0)
        #         thetaobs_range[0],       # thetaObs
        #         thetacore_range[0],      # thetaCore
        #         logn0_range[0],        # log10(n0)
        #         logepsilon_e_range[0],        # log10(eps_e)
        #         logepsilon_b_range[0],        # log10(eps_B)
        #         p_range[0],       # p
        #         thetawing_range[0]        # thetaWing
        #     ])
            
        #     upper_bounds = np.array([
        #         loge0_range[1],        # log10(E0)
        #         thetaobs_range[1],       # thetaObs
        #         thetacore_range[1],      # thetaCore
        #         logn0_range[1],        # log10(n0)
        #         logepsilon_e_range[1],        # log10(eps_e)
        #         logepsilon_b_range[1],        # log10(eps_B)
        #         p_range[1],       # p
        #         thetawing_range[1]        # thetaWing
        #     ])

        # elif z_known == True and dl_known == False:
        #     ndim = 9
        #     lower_bounds = np.array([
        #         loge0_range[0],        # log10(E0)
        #         thetaobs_range[0],       # thetaObs
        #         thetacore_range[0],      # thetaCore
        #         logn0_range[0],        # log10(n0)
        #         logepsilon_e_range[0],        # log10(eps_e)
        #         logepsilon_b_range[0],        # log10(eps_B)
        #         p_range[0],       # p
        #         thetawing_range[0],        # thetaWing
        #         dl_range[0]
        #     ])
            
        #     upper_bounds = np.array([
        #         loge0_range[1],        # log10(E0)
        #         thetaobs_range[1],       # thetaObs
        #         thetacore_range[1],      # thetaCore
        #         logn0_range[1],        # log10(n0)
        #         logepsilon_e_range[1],        # log10(eps_e)
        #         logepsilon_b_range[1],        # log10(eps_B)
        #         p_range[1],       # p
        #         thetawing_range[1],        # thetaWing
        #         dl_range[1]
        #     ])

        # elif z_known == False and dl_known == True:
        #     ndim = 9
        #     lower_bounds = np.array([
        #         loge0_range[0],        # log10(E0)
        #         thetaobs_range[0],       # thetaObs
        #         thetacore_range[0],      # thetaCore
        #         logn0_range[0],        # log10(n0)
        #         logepsilon_e_range[0],        # log10(eps_e)
        #         logepsilon_b_range[0],        # log10(eps_B)
        #         p_range[0],       # p
        #         thetawing_range[0],        # thetaWing
        #         z_range[0]
        #     ])
            
        #     upper_bounds = np.array([
        #         loge0_range[1],        # log10(E0)
        #         thetaobs_range[1],       # thetaObs
        #         thetacore_range[1],      # thetaCore
        #         logn0_range[1],        # log10(n0)
        #         logepsilon_e_range[1],        # log10(eps_e)
        #         logepsilon_b_range[1],        # log10(eps_B)
        #         p_range[1],       # p
        #         thetawing_range[1],        # thetaWing
        #         z_range[1]
        #     ])

        # else:
        #     ndim = 10
        #     lower_bounds = np.array([
        #         loge0_range[0],        # log10(E0)
        #         thetaobs_range[0],       # thetaObs
        #         thetacore_range[0],      # thetaCore
        #         logn0_range[0],        # log10(n0)
        #         logepsilon_e_range[0],        # log10(eps_e)
        #         logepsilon_b_range[0],        # log10(eps_B)
        #         p_range[0],       # p
        #         thetawing_range[0],        # thetaWing
        #         z_range[0],
        #         dl_range[0]
        #     ])
            
        #     upper_bounds = np.array([
        #         loge0_range[1],        # log10(E0)
        #         thetaobs_range[1],       # thetaObs
        #         thetacore_range[1],      # thetaCore
        #         logn0_range[1],        # log10(n0)
        #         logepsilon_e_range[1],        # log10(eps_e)
        #         logepsilon_b_range[1],        # log10(eps_B)
        #         p_range[1],       # p
        #         thetawing_range[1],        # thetaWing
        #         z_range[1],
        #         dl_range[1] 
        #     ])


        if z_known == True and dl_known == True:
            ndim = 3  # REDUCED FROM 6 (only logE0, thetaObs, logn0)
            lower_bounds = np.array([
                loge0_range[0],        # log10(E0)
                thetaobs_range[0],     # thetaObs
                logn0_range[0],        # log10(n0)
                # REMOVED thetaCore, p, thetaWing bounds
            ])

            upper_bounds = np.array([
                loge0_range[1],        # log10(E0)
                thetaobs_range[1],     # thetaObs
                logn0_range[1],        # log10(n0)
                # REMOVED thetaCore, p, thetaWing bounds
            ])

        elif z_known == True and dl_known == False:
            ndim = 4  # REDUCED FROM 7
            lower_bounds = np.array([
                loge0_range[0],        # log10(E0)
                thetaobs_range[0],     # thetaObs
                logn0_range[0],        # log10(n0)
                dl_range[0]            # DL
                # REMOVED thetaCore, p, thetaWing bounds
            ])

            upper_bounds = np.array([
                loge0_range[1],        # log10(E0)
                thetaobs_range[1],     # thetaObs
                logn0_range[1],        # log10(n0)
                dl_range[1]            # DL
                # REMOVED thetaCore, p, thetaWing bounds
            ])

        elif z_known == False and dl_known == True:
            ndim = 4  # REDUCED FROM 7
            lower_bounds = np.array([
                loge0_range[0],        # log10(E0)
                thetaobs_range[0],     # thetaObs
                logn0_range[0],        # log10(n0)
                z_range[0]             # z
                # REMOVED thetaCore, p, thetaWing bounds
            ])

            upper_bounds = np.array([
                loge0_range[1],        # log10(E0)
                thetaobs_range[1],     # thetaObs
                logn0_range[1],        # log10(n0)
                z_range[1]             # z
                # REMOVED thetaCore, p, thetaWing bounds
            ])

        else:
            ndim = 5  # REDUCED FROM 8
            lower_bounds = np.array([
                loge0_range[0],        # log10(E0)
                thetaobs_range[0],     # thetaObs
                logn0_range[0],        # log10(n0)
                z_range[0],            # z
                dl_range[0]            # DL
                # REMOVED thetaCore, p, thetaWing bounds
            ])

            upper_bounds = np.array([
                loge0_range[1],        # log10(E0)
                thetaobs_range[1],     # thetaObs
                logn0_range[1],        # log10(n0)
                z_range[1],            # z
                dl_range[1]            # DL
                # REMOVED thetaCore, p, thetaWing bounds
            ])


        np.random.seed(self.processed_mcmc_config.random_seed)

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
        pos = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(self.nwalkers, ndim))

                

        # Run MCMC
        num_sites = self.server_agent_config.server_configs.aggregator_kwargs.num_clients
        # sampler = emcee.EnsembleSampler(nwalkers, ndim, lambda theta: global_log_probability(theta, num_sites))
        # sampler.run_mcmc(pos, niters, progress=True)

        # Start background thread that listens to all the events and store "LogLikelihoodComputed" Events
        self.communicator._start_background_poller()


        # with Pool() as pool:
            # Create the sampler that will coordinate with all sites
        # Multi-processing not useful in this method. We query in each iteration
        sampler = emcee.EnsembleSampler(
            self.nwalkers, ndim, self.global_log_probability, args=(num_sites,),
            moves=[(mvs.StretchMove(a=1.1), 0.7), (mvs.WalkMove(10), 0.3)])
        
        print('Central coordinator: starting sampling')
        state = sampler.run_mcmc(pos, niters, progress=True)
        
        gc.collect()

        # Save results
        save_folder = self.server_agent_config.server_configs.aggregator_kwargs.save_folder
        os.makedirs(save_folder, exist_ok=True)

        # Get flat samples after burn-in
        run_name = self.server_agent_config.server_configs.aggregator_kwargs.run_name
        burnin = self.processed_mcmc_config.burnin

        flat_samples_withoutBurnin = sampler.get_chain(discard=0, flat=True)
        np.save(os.path.join(save_folder, f"{run_name}_distributed_flat_samples_withoutBurnin.npy"), flat_samples_withoutBurnin)
        
        flat_samples = sampler.get_chain(discard=burnin, flat=True)
        np.save(os.path.join(save_folder, f"{run_name}_distributed_flat_samples.npy"), flat_samples)
        

        # Save sampler state
        # self.save_sampler_state(sampler, os.path.join(save_folder, f"{run_name}_distributed_sampler.pkl"))
        

        # Save log probability
        log_prob = sampler.get_log_prob(flat=False)
        np.save(os.path.join(save_folder, f"{run_name}_distributed_log_prob.npy"), log_prob)
        

       
        # copied from consensus aggregator
        # Create consensus plots
        z_known = self.processed_mcmc_config.Z_known
        dl_known = self.processed_mcmc_config.DL_known

        # if z_known == True and dl_known == True:
        #     params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(eps_e)','log(eps_B)','p', 'thetaWing']

        # elif z_known == False and dl_known == True:
        #     params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(eps_e)','log(eps_B)','p', 'thetaWing', 'z']

        # elif z_known == True and dl_known == False:
        #     params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(eps_e)','log(eps_B)','p', 'thetaWing', 'DL']

        # else:
        #     params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(eps_e)','log(eps_B)','p', 'thetaWing', 'z', 'DL']

        # 6. UPDATE PARAMETER NAMES FOR RESULTS
        if z_known == True and dl_known == True:
            params = ['log(E0)','thetaObs','log(n0)']  # REMOVED thetaCore, p, thetaWing

        elif z_known == False and dl_known == True:
            params = ['log(E0)','thetaObs','log(n0)', 'z']  # REMOVED thetaCore, p, thetaWing

        elif z_known == True and dl_known == False:
            params = ['log(E0)','thetaObs','log(n0)', 'DL']  # REMOVED thetaCore, p, thetaWing

        else:
            params = ['log(E0)','thetaObs','log(n0)', 'z', 'DL']  # REMOVED thetaCore, p, thetaWing

        
        ndim = len(params)
        
        self.make_posterior_hists(
            flat_samples, ndim, params, 
            f"{save_folder}/{run_name}_distributed_PosteriorHists.png"
        )
        
        true_values = self.server_agent_config.client_configs.mcmc_configs.true_values
        display_truths_on_corner = self.processed_mcmc_config.display_truths_on_corner


        self.make_corner_plots(
            flat_samples, ndim, params, 
            true_values, display_truths_on_corner,
            f"{save_folder}/{run_name}_distributed_CornerPlots.png"
        )
        
        # Send consensus parameters back to clients for their plotting
        # consensus_medians = np.median(consensus_samples, axis=0)

        # print('Consensus parameter values:')
        # for i in range(ndim):
        #     print(f"{params[i]}: {consensus_medians[i]:.4f}")

        # print('Distributed parameter values', flush=True)
        # print("Best estimate of parameters", flush=True)
        self.logger.info("Distributed Log-likelihood Best estimate of parameters")
        # theta = []
        results_dict = {}

        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            """
            if i in [0, 3, 4, 5]:
                theta.append(10 ** mcmc[1])
            else:
                theta.append(mcmc[1])
            """
            #keep it in log space
            # theta.append(mcmc[1])
            q = np.diff(mcmc)
            print(f'{params[i]} = {mcmc[1]:.4f} +{q[1]:.4f} -{q[0]:.4f} (68% CI)')
            self.logger.info(f'{params[i]} = {mcmc[1]:.4f} +{q[1]:.4f} -{q[0]:.4f} (68% CI)')

            results_dict[params[i]] = {}
            results_dict[params[i]]['median'] = mcmc[1]
            results_dict[params[i]]['LL'] = mcmc[0]
            results_dict[params[i]]['UL'] = mcmc[2]


        # ─── 3) Log participation summary ──────────────────────────────────────────
        self.logger.info(f"[Server] Participation Summary:")
        for sid, info in self.communicator.site_summary.items():
            if info["has_data"]:
                self.logger.info(
                    f"  • Site {sid}: {info['n_data_points']} pts, day threshold = {info['day_threshold']}"
                )
            else:
                self.logger.info(f"  • Site {sid}: no data, day threshold = {info['day_threshold']}")

        #Save everything in sight:
        # Save results
        print(f"Saved results in {save_folder} after {niters} samples with burnin = {burnin}")
        self.logger.info(f"Saved results in {save_folder} after {niters} samples with burnin = {burnin}")
        
        # Send these results to each client/data site
        return results_dict
    
    def make_posterior_hists(self, samples, ndim, params, save_path):
        """
        Create posterior histograms from consensus samples
        """
        medians = np.median(samples, axis=0)

        # print('Consensus parameter values:')
        # for i in range(ndim):
        #     print(f"{params[i]}: {medians[i]:.4f}")

        # Create subplots
        # if ndim == 8:
        #     fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 2 rows, 4 columns
        # if ndim == 9:
        #     fig, axes = plt.subplots(3, 3, figsize=(16, 10))  # 3 rows, 3 columns
        # if ndim == 10:
        #     fig, axes = plt.subplots(2, 5, figsize=(16, 10))  # 3 rows, 3 columns

        # Create subplots - UPDATE SUBPLOT DIMENSIONS
        if ndim == 3:
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))   # 1 row, 3 columns
        elif ndim == 4:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))   # 2 rows, 2 columns
        elif ndim == 5:
            fig, axes = plt.subplots(2, 3, figsize=(16, 8))   # 2 rows, 3 columns (one empty)

        axes = axes.flatten()

        # Loop over each dimension and create a histogram
        theta = []

        for i in range(ndim):
            theta_component = np.asarray(samples[:, i])
            # lower, upper = np.percentile(theta_component, [15.865, 84.135])
            lower, upper = np.percentile(theta_component, [2.5, 97.5])

            theta.append(theta_component)
            ax = axes[i]
            ax.hist(samples[:, i], bins=20, color="blue", alpha=0.7, label="Samples")

            # Plot mean value as a vertical line
            ax.axvline(medians[i], color="red", linestyle="--", label=f"Median: {medians[i]:.4f}")
            ax.axvline(lower, color="green", linestyle="--", label=f"lower limit 2.5th percentile: {lower:.4f}")
            ax.axvline(upper, color="green", linestyle="--", label=f"upper limit 97.5th percentile: {upper:.4f}")

            ax.set_title(params[i])
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.legend(loc=4)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        
        return medians

    def make_corner_plots(self, samples, ndim, params, true_values, display_truths_on_corner, save_path):
        """
        Create corner plots from consensus samples
        """
        figure = corner.corner(
            samples,
            labels=params,
            show_titles=True,
            title_fmt=".2f",
            # quantiles=[0.05, 0.5, 0.95],  # 90% credible interval
            quantiles=[0.025, 0.5, 0.975],  # 95% credible interval
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 14},
            plot_datapoints=True,
            fill_contours=True,
            levels=(0.68, 0.95,),  # 90% confidence contours
            smooth=1.0,
            smooth1d=1.0,
            truths=true_values[0:ndim] if display_truths_on_corner else None
        )
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')

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
        mcmc_configs = self.server_agent_config.client_configs.mcmc_configs
        
        # Create a dictionary to store processed values
        self.processed_mcmc_config = {}
        
        # Process boolean values
        boolean_keys = [
            'Z_known', 'DL_known', 'include_upper_limits_on_lc', 
            'exclude_time_flag', 'exclude_ra_dec_flag', 
            'exclude_name_flag', 'exclude_wrong_name',
            'exclude_outside_ra_dec_uncertainty', 
            'display_truths_on_corner'
        ]
        
        for key in boolean_keys:
            if hasattr(mcmc_configs, key):
                val = getattr(mcmc_configs, key)
                if isinstance(val, str):
                    self.processed_mcmc_config[key] = val.lower() == 'true'
                else:
                    self.processed_mcmc_config[key] = bool(val)
        
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
                    self.processed_mcmc_config[key] = [float(x) for x in val]
                else:
                    # Handle case where range might be represented as a string "[a, b]"
                    try:
                        if isinstance(val, str):
                            self.processed_mcmc_config[key] = [float(x.strip()) for x in val.strip("[]").split(",")]
                        else:
                            self.processed_mcmc_config[key] = [float(val)]  # Single value case
                    except ValueError:
                        self.processed_mcmc_config[key] = val  # Keep original if conversion fails
        
        # Process numeric values
        numeric_keys = [
            'nwalkers', 'niters', 'burnin', 'random_seed',
            'Z_fixed', 'DL_fixed', 'arcseconds_uncertainty'
        ]
        
        for key in numeric_keys:
            if hasattr(mcmc_configs, key):
                val = getattr(mcmc_configs, key)
                try:
                    if isinstance(val, str):
                        if '.' in val:
                            self.processed_mcmc_config[key] = float(val)
                        else:
                            self.processed_mcmc_config[key] = int(val)
                    else:
                        self.processed_mcmc_config[key] = val  # Already numeric
                except ValueError:
                    self.processed_mcmc_config[key] = val  # Keep original if conversion fails
        
        # to keep do notation
        self.processed_mcmc_config = OmegaConf.create(self.processed_mcmc_config)
        # return self.processed_mcmc_config

            


