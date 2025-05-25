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
import traceback
import threading


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config", 
    type=str, 
    default="examples/configs/client1.yaml",
    help="Path to the configuration file."
)

argparser.add_argument("--day",
        type=str, required=True,
        help='Max day to include (e.g. "3") or "all"')

args = argparser.parse_args()
day_threshold = args.day

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
# server_started = False
# while True:
#     msg_pack = client_communicator.consumer.poll(timeout_ms=1000)
#     for tp, messages in msg_pack.items():
#         for msg in messages:
    client_agent.logger.info(f"[Site {client_agent.get_id()}] msg: {msg}")

    data_str = msg.value.decode("utf-8")
    data = json.loads(data_str)

    Event_type = data["EventType"]
    day_threshold_in_msg = data["threshold"]

    # ignore runs/msgs not meant for this threshold
    if str(day_threshold_in_msg) != str(day_threshold):
        continue

    if Event_type == "ServerStarted":        
        client_communicator.on_server_started(data)
        break
        # server_started = True
# if server_started:
# break            
                # break  # We can break from the loop as we only need that single event
    

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
    
    # Publish Site ready event with metadata
    # load & slice local CSV ──
    if day_threshold != "all":
        local_data = local_data[local_data["days"] <= float(day_threshold)]

    n_pts = len(local_data)
    has_data = n_pts > 0

    # ─── 3) Report readiness & data stats ─────────────────────────────────
    client_communicator.publish_site_ready(day_threshold, has_data, n_pts)

    if not has_data:
        # client_agent.logger.info(f"[Site {client_agent.get_id()}] No data available — will wait for results.")
        client_agent.logger.info(f"[Site {client_agent.get_id()}] No data available — using 0.0 log-likelihood each time in aggregation")
        for message in client_communicator.consumer:
            topic = message.topic
            # try:
            data_str = message.value.decode("utf-8")  # decode to string
            data = json.loads(data_str)          # parse JSON to dict

            # client_agent.logger.info(f"[Site {client_agent.get_id()}] msg: {data}")
            Event_type = data["EventType"]

            day_threshold_in_msg = data["threshold"]

            # ignore runs/msgs not meant for this threshold
            if str(day_threshold_in_msg) != str(day_threshold):
                continue

            if Event_type == "AggregationDone":     
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
        """ 
        Preprocess local data
        """
        preprocessed_local_data = interpret(local_data, client_agent.client_agent_config)
        preprocessed_local_data_UL = interpret_ULs(local_data, client_agent.client_agent_config)

        # Listen for incoming proposed theta
        # while True:
        #     msg_pack = client_communicator.consumer.poll(timeout_ms=1000)
        #     for tp, messages in msg_pack.items():
        #         for message in messages:
        for message in client_communicator.consumer:
            topic = message.topic
            # try:
            data_str = message.value.decode("utf-8")  # decode to string
            data = json.loads(data_str)          # parse JSON to dict

            # client_agent.logger.info(f"[Site {client_agent.get_id()}] msg: {data}")
            Event_type = data["EventType"]
            day_threshold_in_msg = data["threshold"]

            # ignore runs/msgs not meant for this threshold
            if str(day_threshold_in_msg) != str(day_threshold):
                continue

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
            elif Event_type == "SiteReady":  
                    # Site connected and ready for fitting the curve
                    # not triggering anything on server side, just publishing event to octopus fabric
                    # Keep on listening other events
                    continue 

                    # Later we will keep track of connected Sites and check if anyone got disconnected

            elif Event_type == "ServerStarted":
                # Continue listening other events
                continue
            elif Event_type == "LogLikelihoodComputed":
                # Continue listening other events
                continue
            else:
                print(f"[Site {client_agent.get_id()}] Unknown Event Type in topic ({topic}): {Event_type}", flush=True)
                client_agent.logger.info(f"[Site {client_agent.get_id()}] Unknown Event Type in topic ({topic}): {Event_type}")

            # except json.JSONDecodeError as e:
            #     # Handle invalid JSON messages
            #     print(f"[Site {client_agent.get_id()}] JSONDecodeError for message from topic ({topic}): {e}", flush=True)
            #     client_agent.logger.error(f"[Site {client_agent.get_id()}] JSONDecodeError for message from topic ({topic}): {e}")
            
            # except Exception as e:
            #     # Catch-all for other unexpected exceptions
            #     """Octopus down or got a message which doesn't have 'EventType' key"""
                
            #     # Log the traceback
            #     tb = traceback.format_exc()

            #     print(f"[Site {client_agent.get_id()}] Unexpected error while processing message from topic ({topic}): {e}", flush=True)
            #     print(f"[Site {client_agent.get_id()}] Raw message: {msg}", flush=True)
            #     print(f"[Site {client_agent.get_id()}] Traceback: {tb}", flush=True)

            #     client_agent.logger.error(f"[Site {client_agent.get_id()}] Unexpected error while processing message from topic ({topic}): {e}")
            #     client_agent.logger.error(f"[Site {client_agent.get_id()}] Raw message: {msg}")
            #     client_agent.logger.error(f"[Site {client_agent.get_id()}] Traceback: {tb}")



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
    
    # Publish Site ready event with metadata
    # load & slice local CSV ──
    if day_threshold != "all":
        local_data = local_data[local_data["days"] <= float(day_threshold)]

    n_pts = len(local_data)
    has_data = n_pts > 0

    # ─── 3) Report readiness & data stats ─────────────────────────────────
    client_communicator.publish_site_ready(day_threshold, has_data, n_pts)
    
    if not has_data:
        client_agent.logger.info(f"[Site {client_agent.get_id()}] No data available — will wait for results.")
        client_communicator.send_local_results_Octopus(
            local_results=None,
            status="SKIPPED"
        )
    else:
    
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
    
        
        #  Start running local MCMC
        print(f"[Site {client_agent.get_id()}] ready for curve fitting. Now computing local posterior samples and sending to server...")
        client_agent.logger.info(f"[Site {client_agent.get_id()}] ready for curve fitting. Now computing local posterior samples and sending to server...")


        # client_agent.run_local_mcmc(preprocessed_local_data)
            
        # local_result = client_agent.get_parameters()
            
        # """
        # local_result is a dictionary of format
        #     return {
        #         'chain': self.chain,
        #         'min_time': self.min_time,
        #         'max_time': self.max_time,
        #         'unique_frequencies': self.freqs
        #     }

        #     in updated code, just sending chain

        # """


        # client_communicator.send_local_results_Octopus(local_result)

        def run_mcmc_and_send():
            client_agent.logger.info(f"[Site {client_agent.get_id()}] Starting local MCMC...")
            client_agent.run_local_mcmc(preprocessed_local_data)
            local_result = client_agent.get_parameters()
            client_communicator.send_local_results_Octopus(local_result, "DONE")

        # Start MCMC in background thread
        mcmc_thread = threading.Thread(target=run_mcmc_and_send)
        mcmc_thread.start()        
        
        elapsed_time = time.time() - start_time
        print(f"Time to run local MCMC: {elapsed_time:.2f} seconds", flush=True)
        client_agent.logger.info(f"Time to run local MCMC: {elapsed_time:.2f} seconds")
        
    # Listen for AggregationDone Event and plot local graphs
    #  Listen for consensus results from server
    # for msg in client_communicator.consumer:
    client_agent.logger.info(f"[Site {client_agent.get_id()}] Waiting for AggregationDone event from server...")
        
    # aggregation_done = False
    # while max_attempts is None or attempts < max_attempts:
    # Begin polling loop
    # while True:
    for msg in client_communicator.consumer:

        # # Poll messages every second
        # msg_pack = client_communicator.consumer.poll(timeout_ms=5000)
        # # Use poll() with timeout instead of iterator

        # for topic_partition, messages in msg_pack.items():
        #     for msg in messages:
                          
        # client_agent.logger.info(f"[Site {client_agent.get_id()}] msg: {msg}")

        data_str = msg.value.decode("utf-8")
        data = json.loads(data_str)

        Event_type = data["EventType"]
        day_threshold_in_msg = data["threshold"]

        # ignore runs/msgs not meant for this threshold
        if str(day_threshold_in_msg) != str(day_threshold):
            continue

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
            # aggregation_done = True  # Done

        # Optionally break the loop if everything's finished
        # if aggregation_done:
        #     break
    # trigger clean up function
    # close producer, consumer and clean up other things

        
        
       