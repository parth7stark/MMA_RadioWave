import torch
from omegaconf import DictConfig
from typing import Any, Union, List, Dict, Optional
import numpy as np
import os
import matplotlib.pyplot as plt
import corner
from matplotlib.lines import Line2D

class ConsensusAggregator():
    """
    ConsensusAggregator:
        Aggregator for federated fitting, which takes in local chains from each site,
        concatenates them using consensus MCMC method. The aggregator then uses the global posterior to estimate the best parameters.
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


        # Our local aggregator: store partial local until we have all the sites
        # aggregator[site_id] = {local_chain: [...], local_time_range: [...], local_filters: [...]}

        self.aggregated_results={}

        # Track which clients have finished
        self.completed_clients = set()

        num_clients = self.server_agent_config.server_configs.aggregator_kwargs.num_clients  # Change this value as needed
        self.expected_clients = {str(i) for i in range(num_clients)}
        

    #-------------------------------------------------
    # AGGREGATION FUNCTION: GAUSSIAN CONSENSUS
    #-------------------------------------------------
    def aggregate_gaussian(self, chains):
        """
        Approximate each subposterior as Gaussian and compute the
        product of Gaussians:
        Sigma_full = (sum_i inv(Sigma_i))^-1,
        mu_full = Sigma_full * (sum_i inv(Sigma_i)*mu_i)
        """
        means = []
        covs = []
        for chain in chains:
            mu = np.mean(chain, axis=0)
            cov = np.cov(chain, rowvar=False)
            means.append(mu)
            covs.append(cov)
        precision_sum = np.zeros_like(covs[0])
        weighted_mean_sum = np.zeros_like(means[0])
        for mu, cov in zip(means, covs):
            inv_cov = np.linalg.inv(cov)
            precision_sum += inv_cov
            weighted_mean_sum += inv_cov @ mu
        Sigma_full = np.linalg.inv(precision_sum)
        mu_full = Sigma_full @ weighted_mean_sum
        return mu_full, Sigma_full

    def process_local_MCMC_done_message(self, producer, topic, site_id, status, local_chain):
        """
        Handle "Local_MCMC_done" messages. Wait until all clients are done.
        """    
        if status == "DONE" and site_id is not None:
            print(f"[Server] Received DONE from site {site_id}")
            self.logger.info(f"[Server] Received DONE from site {site_id}")

            # Add the client to the completed set
            self.completed_clients.add(site_id)

            self.aggregated_results[site_id] = {
                "local_chain": local_chain
                # "min_time": min_time,
                # "max_time": max_time,
                # "unique_frequencies": unique_frequencies
            }

            # Check if all expected clients are done
            if self.completed_clients == self.expected_clients:
                print("[Server] All sites are DONE. Invoking global aggregation process...")
                self.logger.info("[Server] All sites are DONE. Invoking global aggregation process...")
                self.global_aggregation(producer, topic)

    def global_aggregation(self, producer, topic):
        """
        Once all site's local MCMC is done, we gather local chains, time_range and frequencies
        """
        if self.aggregated_results:

            # Aggregate the chains from each site.
            chains = [self.aggregated_results[site]["local_chain"] for site in self.aggregated_results]
            mu_full, Sigma_full = self.aggregate_gaussian(chains)
            
        print(f"[Server] Global mu, sigma: {mu_full}, {Sigma_full}", flush=True)  #length of signal = length of output tensor

        consensus_samples = np.random.multivariate_normal(mu_full, Sigma_full, size=10000)

        # Save consensus samples
        save_folder = self.server_agent_config.server_configs.aggregator_kwargs.save_folder
        os.makedirs(save_folder, exist_ok=True)


        run_name = self.server_agent_config.server_configs.aggregator_kwargs.run_name
        np.save(f"{save_folder}/{run_name}_consensus_samples.npy", consensus_samples)
        
        # Create consensus plots
        z_known = self.server_agent_config.client_configs.mcmc_configs.Z_known

        if z_known:
            params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(eps_e)','log(eps_B)','p', 'thetaWing']
        else:
            params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(eps_e)','log(eps_B)','p', 'thetaWing', 'z']
        
        ndim = len(params)
        
        self.make_posterior_hists(
            consensus_samples, ndim, params, 
            f"{save_folder}/{run_name}_consensus_PosteriorHists.png"
        )
        
        true_values = self.server_agent_config.client_configs.mcmc_configs.true_values
        display_truths_on_corner = self.server_agent_config.client_configs.mcmc_configs.display_truths_on_corner


        self.make_corner_plots(
            consensus_samples, ndim, params, 
            true_values, display_truths_on_corner,
            f"{save_folder}/{run_name}_consensus_CornerPlots.png"
        )
        
        # Send consensus parameters back to clients for their plotting
        # consensus_medians = np.median(consensus_samples, axis=0)

        # print('Consensus parameter values:')
        # for i in range(ndim):
        #     print(f"{params[i]}: {consensus_medians[i]:.4f}")

        print('Consensus parameter values', flush=True)
        self.logger.info('Consensus parameter values')

        print("Best estimate of parameters", flush=True)
        self.logger.info("Best estimate of parameters")
        theta = []
        results_dict = {}

        for i in range(ndim):
            mcmc = np.percentile(consensus_samples[:, i], [16, 50, 84])
            """
            if i in [0, 3, 4, 5]:
                theta.append(10 ** mcmc[1])
            else:
                theta.append(mcmc[1])
            """
            #keep it in log space
            theta.append(mcmc[1])
            q = np.diff(mcmc)
            print(f'{params[i]} = {mcmc[1]:.4f} +{q[1]:.4f} -{q[0]:.4f}')
            self.logger.info(f'{params[i]} = {mcmc[1]:.4f} +{q[1]:.4f} -{q[0]:.4f}')

            results_dict[params[i]] = {}
            results_dict[params[i]]['median'] = mcmc[1]
            results_dict[params[i]]['LL'] = mcmc[0]
            results_dict[params[i]]['UL'] = mcmc[2]

        # consensus_result = {
        #     'params': params,
        #     'medians': consensus_medians.tolist()
        # }
        
        # Code to send consensus_result via Kafka would go here
        
        print(f"\nConsensus MCMC complete. Results saved to {save_folder}")
        self.logger.info(f"\nConsensus MCMC complete. Results saved to {save_folder}")
        
        # Compute final aggregated parameters.
        # print("Best estimate of parameters", flush=True)
        # self.logger.info("Best estimate of parameters")
        # theta_est = []
        # params = ['log(E0)','thetaObs','thetaCore','log(n0)',
        #           'log(epsilon_e)','log(epsilon_B)','p']
        # ndim = 7
        
        # for i in range(ndim):
        #     mcmc = np.percentile(final_samples[:, i], [16, 50, 84])
        #     # For parameters stored in log-space (indices 0, 3, 4, 5) convert back.
        #     if i in [0, 3, 4, 5]:
        #         theta_est.append(10 ** mcmc[1])
        #     else:
        #         theta_est.append(mcmc[1])
        #     q = np.diff(mcmc)
        #     print(f'{params[i]} = {mcmc[1]:.2f} +{q[1]:.2f} -{q[0]:.2f}', flush=True)
        #     self.logger.info(f'{params[i]} = {mcmc[1]:.2f} +{q[1]:.2f} -{q[0]:.2f}')
            
        
        # Combine summary statistics from sites: global min and max times, union of frequencies.
        # global_min = min(result["min_time"] for result in self.aggregated_results.values())
        # global_max = max(result["max_time"] for result in self.aggregated_results.values())
        
        # all_freqs = set()
        # for result in self.aggregated_results.values():
        #     all_freqs.update(result["unique_frequencies"])
        # global_freqs = sorted(list(all_freqs))
        
        # Publish the aggregated (global) information on topic 
        # Send detection details to Kafka
        
        producer.send(topic, value={
        
            "EventType": "AggregationDone",
            "theta_est": results_dict
            # "global_min_time": global_min,
            # "global_max_time": global_max,
            # "unique_frequencies": global_freqs
        })
        # producer.flush()
        
        # print("[Server] Published AggregationDone event with best parameter estimates, global time range and frequencies.", flush=True)
        # self.logger.info("[Server] Published AggregationDone event with best parameter estimates, global time range and frequencies.")

        
        
        print("[Server] Published AggregationDone event with best parameter estimates.", flush=True)
        self.logger.info("[Server] Published AggregationDone event with best parameter estimates.")
                         
    def make_posterior_hists(self, samples, ndim, params, save_path):
        """
        Create posterior histograms from consensus samples
        """
        medians = np.median(samples, axis=0)

        # print('Consensus parameter values:')
        # for i in range(ndim):
        #     print(f"{params[i]}: {medians[i]:.4f}")

        # Create subplots
        if ndim == 8:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 2 rows, 4 columns
        if ndim == 9:
            fig, axes = plt.subplots(3, 3, figsize=(16, 10))  # 3 rows, 3 columns

        axes = axes.flatten()

        # Loop over each dimension and create a histogram
        theta = []

        for i in range(ndim):
            theta_component = np.asarray(samples[:, i])
            lower, upper = np.percentile(theta_component, [15.865, 84.135])

            theta.append(theta_component)
            ax = axes[i]
            ax.hist(samples[:, i], bins=20, color="blue", alpha=0.7, label="Samples")

            # Plot mean value as a vertical line
            ax.axvline(medians[i], color="red", linestyle="--", label=f"Median: {medians[i]:.4f}")
            ax.axvline(lower, color="green", linestyle="--", label=f"lower limit: {lower:.4f}")
            ax.axvline(upper, color="green", linestyle="--", label=f"upper limit: {upper:.4f}")

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
            quantiles=[0.05, 0.5, 0.95],  # 90% credible interval
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 14},
            plot_datapoints=True,
            fill_contours=True,
            levels=(0.50, 0.90,),  # 90% confidence contours
            smooth=1.0,
            smooth1d=1.0,
            truths=true_values if len(true_values) == ndim and display_truths_on_corner else None
        )
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
