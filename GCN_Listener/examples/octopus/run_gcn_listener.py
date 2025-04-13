import argparse
from omegaconf import OmegaConf
from mma_fedfit.agent import ServerAgent
from mma_fedfit.communicator.octopus import OctopusServerCommunicator
import json
import traceback
import time
import numpy as np
import emcee


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config", 
    type=str, 
    default="examples/configs/server.yaml",
    help="Path to the configuration file."
)

args = argparser.parse_args()

# Load config from YAML (via OmegaConf)
server_agent_config = OmegaConf.load(args.config)

# Initialize server-side modules
server_agent = ServerAgent(server_agent_config=server_agent_config)

# Create server-side communicator
communicator = OctopusServerCommunicator(
    server_agent,
    logger=server_agent.logger,
)

if server_agent_config.client_configs.fitting_configs.use_approach == "1":

    # Consensus MCMC workflow
    # Publish "ServerStarted" event with config so that clients can pick it up
    communicator.publish_server_started_event()


    print("[Server] Listening for messages...", flush=True)
    server_agent.logger.info("[Server] Listening for messages...")

    for msg in communicator.consumer:
        topic = msg.topic
        try:
            data_str = msg.value.decode("utf-8")  # decode to string
            data = json.loads(data_str)          # parse JSON to dict

            Event_type = data["EventType"]

            if Event_type == "LocalMCMCDone":
                communicator.handle_local_MCMC_done_message(data)

            elif Event_type == "AggregationDone":
                # not triggering anything on server side
                continue

            elif Event_type == "SiteReady":  
                # Site connected and ready for fitting the curve
                # not triggering anything on server side, just publishing event to octopus fabric
                # Keep on listening other events
                continue 

                # Later we will keep track of connected Sites and check if anyone got disconnected

            elif Event_type == "ServerStarted":
                # Continue listening other events
                continue

            else:
                print(f"[Server] Unknown Event Type in topic ({topic}): {Event_type}", flush=True)
                server_agent.logger.info(f"[Server] Unknown Event Type in topic ({topic}): {Event_type}")

        except json.JSONDecodeError as e:
            # Handle invalid JSON messages
            print(f"[Server] JSONDecodeError for message from topic ({topic}): {e}", flush=True)
            server_agent.logger.error(f"[Server] JSONDecodeError for message from topic ({topic}): {e}")
        
        except Exception as e:
            # Catch-all for other unexpected exceptions
            """Octopus down or got a message which doesn't have 'EventType' key"""
            
            # Log the traceback
            tb = traceback.format_exc()

            print(f"[Server] Unexpected error while processing message from topic ({topic}): {e}", flush=True)
            print(f"[Server] Raw message: {msg}", flush=True)
            print(f"[Server] Traceback: {tb}", flush=True)

            server_agent.logger.error(f"[Server] Unexpected error while processing message from topic ({topic}): {e}")
            server_agent.logger.error(f"[Server] Raw message: {msg}")
            server_agent.logger.error(f"[Server] Traceback: {tb}")
else:

    # Sum of log likelihood approach workflow
    print("Running sum of log likelihood approach", flush=True )
    server_agent.logger.info("Running sum of log likelihood approach")
    
    # Publish "ServerStarted" event with config so that clients can pick it up
    communicator.publish_server_started_event()

    s = 0

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
        
    # Global log-probability function (aggregates local log-likelihoods)
    def global_log_probability(theta, num_sites):
        
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf

        # Send MCMC tasks to sites
        communicator.send_proposed_theta(theta, s)

        # Collect local log-likelihoods
        log_likelihoods = communicator.collect_local_likelihoods(num_sites)

        s = s + 1
        return lp + sum(log_likelihoods.values())
    

    start_time = time.time()
    print("Running global MCMC")
    server_agent.logger.info("Running global MCMC")


    Z = server_agent_config.client_configs.initial_guess
    nwalkers = server_agent_config.client_configs.mcmc_configs.nwalkers
    niters = server_agent_config.client_configs.mcmc_configs.niters
    ndim = 7

    # Prepare initial walker positions (in 7 dimensions).
    pos = ( [ np.log10(Z["E0"]),
                Z["thetaObs"],
                Z["thetaCore"],
                np.log10(Z["n0"]),
                np.log10(Z["epsilon_e"]),
                np.log10(Z["epsilon_B"]),
                Z["p"] ] 
            + 0.005 * np.random.randn(nwalkers, ndim) )
            

    # Run MCMC
    num_sites = server_agent_config.server_configs.aggregator_kwargs.num_clients
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lambda theta: global_log_probability(theta, num_sites))
    sampler.run_mcmc(pos, niters, progress=True)

    print("[Server] Final aggregated MCMC completed!")
    server_agent.logger.info("[Server] Final aggregated MCMC completed!")


    flat_samples = sampler.get_chain(discard=100, thin=2, flat=True)

    theta_est = []
    params = ['log(E0)','thetaObs','thetaCore','log(n0)',
                'log(epsilon_e)','log(epsilon_B)','p']
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        if i in [0, 3, 4, 5]:
            theta_est.append(10 ** mcmc[1])
        else:
            theta_est.append(mcmc[1])
        q = np.diff(mcmc)
        print(f'{params[i]} = {mcmc[1]:.2f} +{q[1]:.2f} -{q[0]:.2f}')
    
    # -------------------------------------------------
    # Plot 1: Final aggregated light curve with points differentiated by site.
    # -------------------------------------------------
    # Combine all site data (with site labels)
    # data_all = pd.concat([data_site1, data_site2, data_site3])
    # plot_final_lc_by_site(data_all, theta_est, args.final_plot)