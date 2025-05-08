import argparse
from omegaconf import OmegaConf
from mma_fedfit.agent import ServerAgent
from mma_fedfit.communicator.octopus import OctopusServerCommunicator
import json
import traceback
import time
import numpy as np
import emcee
import gc
from emcee import moves as mvs
from multiprocessing import Pool



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

    #save a copy of the .ini to the save folder:
    # shutil.copy(ini_file_path, os.path.join(save_folder, run_name + '_params.ini'))


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

    start_time = time.time()
    print("Running global MCMC")
    server_agent.logger.info("Running global MCMC")

    results_dict = server_agent.run_distributed_MCMC(communicator)

    communicator.send_aggregation_results(results_dict)

    print("[Server] Final aggregated MCMC completed!")
    server_agent.logger.info("[Server] Final aggregated MCMC completed!")

    # -------------------------------------------------
    # Plot 1: Final aggregated light curve with points differentiated by site.
    # -------------------------------------------------
    # Combine all site data (with site labels)
    # data_all = pd.concat([data_site1, data_site2, data_site3])
    # plot_final_lc_by_site(data_all, theta_est, args.final_plot)