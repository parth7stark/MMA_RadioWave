import argparse
from omegaconf import OmegaConf
from mma_fedfit.agent import ClientAgent
from mma_fedfit.communicator.octopus import OctopusClientCommunicator

from mma_fedfit.generator.inference_utils import *
import time
import json
import pandas as pd


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
    print("Workflow to be implemented")
    # Implement Workflow using Octopus [TODO]

    # Read the inference dataset (replaced glob, performing inference on only 1 hdf5 file)
    data_dir = client_agent.client_agent_config.fitting_configs.dataset_path
    dataset_name = data_dir.split('/')[-1].split('_')[0]

    print(f"Computing log-likelihood on {dataset_name} dataset", flush=True)
    client_agent.logger.info(f"[Site {client_agent.get_id()}] Computing log-likelihood on {dataset_name} dataset")

    # Load flux-time at the site (local data)
    local_data = pd.read_csv(data_dir)

    """ 
    Preprocess local data
    """
    preprocessed_local_data = preprocess(local_data)

    # Listen for incoming proposed theta
    for message in client_communicator.consumer:
        data_str = msg.value.decode("utf-8")  # decode to string
            data = json.loads(data_str)          # parse JSON to dict

            Event_type = data["EventType"]

            if Event_type == "ProposedTheta":
                client_communicator.handle_proposed_theta_message(data, preprocessed_local_data)


else:
    ### Consensus MCMCC workflow ###

    # Read the inference dataset (replaced glob, performing inference on only 1 hdf5 file)
    data_dir = client_agent.client_agent_config.fitting_configs.dataset_path
    dataset_name = data_dir.split('/')[-1].split('_')[0]

    print(f"Running MCMC on {dataset_name} dataset", flush=True)
    client_agent.logger.info(f"[Site {client_agent.get_id()}] Running MCMC on {dataset_name} dataset")

    # Load flux-time at the site (local data)
    local_data = pd.read_csv(data_dir)

    """ 
    Preprocess local data
    """
    preprocessed_local_data = preprocess(local_data)

    """
    Take initial guess from client_config:
        client_agent.run_local_mcmc(processed_local_data)
        local_result = client_agent.get_parameters()
        client_communicator.send_results(local_result)
    """
    start_time = time.time()
    print("Running local MCMC")

    Z = self.client_agent.client_agent_config.fitting_configs.intial_guess
    nwalkers = self.client_agent.client_agent_config.fitting_configs.mcmc_configs.nwalkers
    niters = self.client_agent.client_agent_config.fitting_configs.mcmc_configs.niters
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
    
    client_agent.run_local_mcmc(preprocessed_local_data, pos, niter, nwalkers, ndim)
        
    local_result = client_agent.get_parameters()
        
        """
        local_result is a dictionary of format
            return {
                'chain': self.chain,
                'min_time': self.min_time,
                'max_time': self.max_time,
                'unique_frequencies': self.freqs
            }

       """


    client_communicator.send_local_results_Octopus(local_result)
        
    
    elapsed_time = time.time() - start_time
    print(f"Time to run local MCMC: {elapsed_time:.2f} seconds", flush=True)
    
    # Listen for AggregationDone Event and plot local graphs

    for msg in client_communicator.consumer:
        client_agent.logger.info(f"[Site {client_agent.get_id()}] msg: {msg}")

        data_str = msg.value.decode("utf-8")
        data = json.loads(data_str)

        Event_type = data["EventType"]

        if Event_type == "AggregationDone":     
             print(f"[Site {client_agent.get_id()}] Plotting global light curve", flush=True)
            client_agent.logger.info(f"[Site {client_agent.get_id()}] Plotting global light curve")
   
            theta_est, global_min_time, global_max_time, unique_frequencies = client_communicator.get_best_estimate(data)
            
            # Plot the global light curve locally using the aggregated info.
            output_filename = f"global_light_curve_Site_{client_agent.get_id()}.png"

            # Function present in generator utils
            plot_global_light_curve(theta_est, global_min, global_max, frequencies, output_filename)
            break  # We can break from the loop as we only need that single event
        
    # trigger clean up function
    # close producer, consumer and clean up other things
