import torch
from omegaconf import DictConfig
from typing import Any, Union, List, Dict, Optional
import numpy as np

class GlobalAggregator():
    """
    GlobalAggregator:
        Aggregator for federated fitting, which takes in local chains from each site,
        concatenates them using consensus MCMC method. The aggregator then uses the global posterior to estimate the best parameters.
        Best parameters are sended back to the clients
        for them to plot the light curve locally.
    """
    def __init__(
        self,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Any | None = None,
    ):

        self.logger = logger
        self.aggregator_configs = aggregator_configs


        # Our local aggregator: store partial local until we have all the sites
        # aggregator[site_id] = {local_chain: [...], local_time_range: [...], local_filters: [...]}

        self.aggregated_results={}

        # Track which clients have finished
        self.completed_clients = set()

        num_clients = self.aggregator_configs.num_clients  # Change this value as needed
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

    def process_local_MCMC_done_message(self, producer, topic, site_id, status, local_chain, min_time, max_time, unique_frequencies):
        """
        Handle "Local_MCMC_done" messages. Wait until all clients are done.
        """    
        if status == "DONE" and site_id is not None:
            print(f"[Server] Received DONE from site {site_id}")
            self.logger.info(f"[Server] Received DONE from site {site_id}")

            # Add the client to the completed set
            self.completed_clients.add(site_id)

            self.aggregated_results[site_id] = {
                "local_chain": local_chain,
                "min_time": min_time,
                "max_time": max_time,
                "unique_frequencies": unique_frequencies
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

        final_samples = np.random.multivariate_normal(mu_full, Sigma_full, size=10000)

        # Compute final aggregated parameters.
        print("Best estimate of parameters", flush=True)
        self.logger.info("Best estimate of parameters")
        theta_est = []
        params = ['log(E0)','thetaObs','thetaCore','log(n0)',
                  'log(epsilon_e)','log(epsilon_B)','p']
        ndim = 7
        
        for i in range(ndim):
            mcmc = np.percentile(final_samples[:, i], [16, 50, 84])
            # For parameters stored in log-space (indices 0, 3, 4, 5) convert back.
            if i in [0, 3, 4, 5]:
                theta_est.append(10 ** mcmc[1])
            else:
                theta_est.append(mcmc[1])
            q = np.diff(mcmc)
            print(f'{params[i]} = {mcmc[1]:.2f} +{q[1]:.2f} -{q[0]:.2f}', flush=True)
            self.logger.info(f'{params[i]} = {mcmc[1]:.2f} +{q[1]:.2f} -{q[0]:.2f}')
            
        
        # Combine summary statistics from sites: global min and max times, union of frequencies.
        global_min = min(result["min_time"] for result in self.aggregated_results.values())
        global_max = max(result["max_time"] for result in self.aggregated_results.values())
        
        all_freqs = set()
        for result in self.aggregated_results.values():
            all_freqs.update(result["unique_frequencies"])
        global_freqs = sorted(list(all_freqs))
        
        # Publish the aggregated (global) information on topic 
        # Send detection details to Kafka
        producer.send(topic, value={
        
            "EventType": "AggregationDone",
            "theta_est": theta_est,
            "global_min_time": global_min,
            "global_max_time": global_max,
            "unique_frequencies": global_freqs
        })
        producer.flush()
        
        print("[Server] Published AggregationDone event with best parameter estimates, global time range and frequencies.", flush=True)
        self.logger.info("[Server] Published AggregationDone event with best parameter estimates, global time range and frequencies.")
