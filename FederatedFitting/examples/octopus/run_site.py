import argparse
from omegaconf import OmegaConf
from mma_fedfit.agent import ClientAgent
from mma_fedfit.communicator.octopus import OctopusClientCommunicator

from mma_fedfit.generator.inference_utils import *
import time
import json
import pandas as pd
import numpy as np
import os


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config", 
    type=str, 
    default="examples/configs/client1.yaml",
    help="Path to the configuration file."
)
args = argparser.parse_args()

# Load config from YAML (via OmegaConf)
client_agent_config = OmegaConf.load(args.config)

# Initialize client-side modules
client_agent = ClientAgent(client_agent_config=client_agent_config)

# Create client-side communicator
client_communicator = OctopusClientCommunicator(
    client_agent,
    client_id = client_agent.get_id(),
    logger=client_agent.logger,

)

print(f"[Site {client_agent.get_id()}] Waiting for ServerStarted event...", flush=True)
client_agent.logger.info(f"[Site {client_agent.get_id()}] Waiting for ServerStarted event...")

# 1) Wait for ServerStarted event
for msg in client_communicator.consumer:
    client_agent.logger.info(f"[Site {client_agent.get_id()}] msg: {msg}")

    data_str = msg.value.decode("utf-8")
    data = json.loads(data_str)

    Event_type = data["EventType"]

    if Event_type == "ServerStarted":        
        client_communicator.on_server_started(data)
        break  # We can break from the loop as we only need that single event
    

#  Start running local MCMC
print(f"[Site {client_agent.get_id()}] ready for curve fitting. Now computing local posterior samples and sending to server...")
client_agent.logger.info(f"[Site {client_agent.get_id()}] ready for curve fitting. Now computing local posterior samples and sending to server...")



if client_agent.client_agent_config.fitting_configs.use_approach=="2":
    # Use Global likelihood MCMC approach
    # print("Workflow to be implemented")
    # Implement Workflow using Octopus [TODO]

    # Read the inference dataset (replaced glob, performing inference on only 1 hdf5 file)
    data_dir = client_agent.client_agent_config.fitting_configs.dataset_path
    dataset_name = data_dir.split('/')[-1].split('_')[0]

    print(f"Computing log-likelihood on {dataset_name} dataset", flush=True)
    client_agent.logger.info(f"[Site {client_agent.get_id()}] Computing log-likelihood on {dataset_name} dataset")

    # Load flux-time at the site (local data)
    local_data = pd.read_csv(data_dir)

    if local_data.shape[1] == 1:
        local_data = pd.read_csv(data_dir, delim_whitespace=True)
    
    """ 
    Preprocess local data
    """
    preprocessed_local_data = interpret(local_data, client_agent.client_agent_config)
    preprocessed_local_data_UL = interpret_ULs(local_data, client_agent.client_agent_config)

    # Listen for incoming proposed theta
    for message in client_communicator.consumer:
        data_str = msg.value.decode("utf-8")  # decode to string
        data = json.loads(data_str)          # parse JSON to dict

        Event_type = data["EventType"]

        if Event_type == "ProposedTheta":
            client_communicator.handle_proposed_theta_message(data, preprocessed_local_data)
        elif Event_type == "AggregationDone":     
            print(f"[Site {client_agent.get_id()}] Received distributed MCMC results", flush=True)
            client_agent.logger.info(f"[Site {client_agent.get_id()}] Received distributed MCMC results")

            theta_est = client_communicator.get_best_estimate(data)

            # distributed_result = {}  # This would be populated from Kafka message
        
            # # Parse distributed parameters
            # distributed_params = np.array(distributed_result['medians'])
            distributed_params = np.array(theta_est)


            # Skip plotting for now
            break


else:
    ### Consensus MCMC workflow ###

    # Read the inference dataset (replaced glob, performing inference on only 1 hdf5 file)
    data_dir = client_agent.client_agent_config.fitting_configs.dataset_path
    dataset_name = data_dir.split('/')[-1].split('_')[0]

    print(f"Running MCMC on {dataset_name} dataset", flush=True)
    client_agent.logger.info(f"[Site {client_agent.get_id()}] Running MCMC on {dataset_name} dataset")

    # Load flux-time at the site (local data)
    local_data = pd.read_csv(data_dir)

    if local_data.shape[1] == 1:
        local_data = pd.read_csv(data_dir, delim_whitespace=True)
    
    """ 
    Preprocess local data
    """
    preprocessed_local_data = interpret(local_data, client_agent.client_agent_config)
    preprocessed_local_data_UL = interpret_ULs(local_data, client_agent.client_agent_config)

    """
    Take initial guess from client_config:
        client_agent.run_local_mcmc(processed_local_data)
        local_result = client_agent.get_parameters()
        client_communicator.send_results(local_result)
    """
    start_time = time.time()
    print("Running local MCMC")

    # Prepare MCMC parameters
    z_known = client_agent.client_agent_config.mcmc_configs.Z_known
    ndim = 8 if z_known else 9
    nwalkers = client_agent.client_agent_config.mcmc_configs.nwalkers
    niters = client_agent.client_agent_config.mcmc_configs.niters

    loge0_range = client_agent.client_agent_config.mcmc_configs.logE0_range
    thetaobs_range = client_agent.client_agent_config.mcmc_configs.thetaObs_range
    thetacore_range = client_agent.client_agent_config.mcmc_configs.thetaCore_range
    logn0_range = client_agent.client_agent_config.mcmc_configs.logn0_range
    logepsilon_e_range = client_agent.client_agent_config.mcmc_configs.logEpsilon_e_range
    logepsilon_b_range = client_agent.client_agent_config.mcmc_configs.logEpsilon_B_range
    p_range = client_agent.client_agent_config.mcmc_configs.P_range
    thetawing_range = client_agent.client_agent_config.mcmc_configs.thetaWing_range
    z_range = client_agent.client_agent_config.mcmc_configs.Z_range
    
    # Define parameter bounds
    if z_known:
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
    else:
        lower_bounds = np.array([
            loge0_range[0],        # log10(E0)
            thetaobs_range[0],     # thetaObs
            thetacore_range[0],    # thetaCore
            logn0_range[0],        # log10(n0)
            logepsilon_e_range[0], # log10(eps_e)
            logepsilon_b_range[0], # log10(eps_B)
            p_range[0],            # p
            thetawing_range[0],    # thetaWing
            z_range[0]             # z
        ])
        
        upper_bounds = np.array([
            loge0_range[1],        # log10(E0)
            thetaobs_range[1],     # thetaObs
            thetacore_range[1],    # thetaCore
            logn0_range[1],        # log10(n0)
            logepsilon_e_range[1], # log10(eps_e)
            logepsilon_b_range[1], # log10(eps_B)
            p_range[1],            # p
            thetawing_range[1],    # thetaWing
            z_range[1]             # z
        ])
    

    np.random.seed(client_agent.client_agent_config.mcmc_configs.random_seed)

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

    
    client_agent.run_local_mcmc(preprocessed_local_data, pos, niters, nwalkers, ndim)
        
    local_result = client_agent.get_parameters()
        
    """
    local_result is a dictionary of format
        return {
            'chain': self.chain,
            'min_time': self.min_time,
            'max_time': self.max_time,
            'unique_frequencies': self.freqs
        }

        in updated code, just sending chain

    """


    client_communicator.send_local_results_Octopus(local_result)
        
    
    elapsed_time = time.time() - start_time
    print(f"Time to run local MCMC: {elapsed_time:.2f} seconds", flush=True)
    
    # Listen for AggregationDone Event and plot local graphs
    #  Listen for consensus results from server
    for msg in client_communicator.consumer:
        client_agent.logger.info(f"[Site {client_agent.get_id()}] msg: {msg}")

        data_str = msg.value.decode("utf-8")
        data = json.loads(data_str)

        Event_type = data["EventType"]

        if Event_type == "AggregationDone":     
            print(f"[Site {client_agent.get_id()}] Received consensus MCMC results", flush=True)
            client_agent.logger.info(f"[Site {client_agent.get_id()}] Received consensus MCMC results")

            theta_est = client_communicator.get_best_estimate(data)

            # consensus_result = {}  # This would be populated from Kafka message
        
            # # Parse consensus parameters
            # consensus_params = np.array(consensus_result['medians'])
            consensus_params = np.array(theta_est)


            # Skip plotting for now

            # print(f"[Site {client_agent.get_id()}] Plotting light curve using consensus mcmc model paramters", flush=True)
            # client_agent.logger.info(f"[Site {client_agent.get_id()}] Plotting light curve using consensus mcmc model paramters")
   
            # # Plot local data with consensus model
            # include_upper_limits_on_lc = client_agent.client_agent_config.mcmc_configs.include_upper_limits_on_lc
            
            # site_folder = client_agent.client_agent_config.fitting_configs.save_folder
            # os.makedirs(site_folder, exist_ok=True)
            
            # site_id = client_agent.get_id()
            # run_name = client_agent.client_agent_config.fitting_configs.run_name

            # if include_upper_limits_on_lc:
            #     plot_lc_wUL(data, preprocessed_local_data_UL, consensus_params, site_id, f"{site_folder}/{run_name}_lightcurve_consensus.png")
            # else:
            #     plot_lc_noUL(data, consensus_params, site_id, f"{site_folder}/{run_name}_lightcurve_consensus.png")
            
            # print(f"Site {site_id} processing complete. Results saved to {site_folder}")

            # theta_est, global_min_time, global_max_time, unique_frequencies = client_communicator.get_best_estimate(data)
            
            # # Plot the global light curve locally using the aggregated info.
            # output_filename = f"{client_agent.client_agent_config.fitting_configs.logging_output_dirname}/global_light_curve_Site_{client_agent.get_id()}.png"

            # # Function present in generator utils
            # plot_global_light_curve(preprocessed_local_data, theta_est, global_min_time, global_max_time, unique_frequencies, output_filename)
            break  # We can break from the loop as we only need that single event
        
    # trigger clean up function
    # close producer, consumer and clean up other things

        
        
       