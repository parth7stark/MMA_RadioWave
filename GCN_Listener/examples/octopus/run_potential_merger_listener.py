import argparse
from omegaconf import OmegaConf
from mma_gcn.agent import PotentialMergerAgent
from mma_gcn.communicator.octopus import OctopusPMCommunicator
import json
import traceback
import time
import os
from glob import glob

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config", 
    type=str, 
    default="examples/configs/gcn_listener_config.yaml",
    help="Path to the configuration file."
)

args = argparser.parse_args()

# Load config from YAML (via OmegaConf)
merger_agent_config = OmegaConf.load(args.config)

# Initialize gcn-side modules
merger_agent = PotentialMergerAgent(merger_agent_config=merger_agent_config)

# Create Octopus communicator
octopuscommunicator = OctopusPMCommunicator(
    merger_agent,
    logger=merger_agent.logger,
)

# Listen for potential merger events on Octopus
print("[Potential Merger Listener] Listening for potential merger events from GW module...", flush=True)
merger_agent.logger.info("[Potential Merger Listener] Listening for potential merger events from GW module...")

"""
    Listens for PotentialMerger events from Octopus Kafka topic
    and stores them for later correlation with LVK notices.
"""
if merger_agent_config.gcn_listener_configs.simulate_events == "no":
    for msg in octopuscommunicator.consumer:
        topic = msg.topic
        merger_agent.logger.info(f"[octopus msg: {msg}")

        data_str = msg.value.decode("utf-8")  # decode to string
        data = json.loads(data_str)          # parse JSON to dict

        Event_type = data["EventType"]

        if Event_type == "PotentialMerger":
            octopuscommunicator.handle_potential_merger_message(data)
        else:
            merger_agent.logger.debug(f"Received non-merger event: {data.get('EventType')}")
        
else:
    """
    Simulate potential merger listener by reading events from files.
    """
    event_dir = merger_agent_config.gcn_listener_configs.simulation_datadir
    event_files = sorted(glob(os.path.join(event_dir, "*.json")))
    print(f"[Simulator] Found {len(event_files)} merger event files to process.")
    
    for filepath in event_files:
        with open(filepath, 'r') as f:
            data_str = f.read()
        data = json.loads(data_str)
        
        topic = "simulated.topic.gw"  # Simulated topic
        merger_agent.logger.info(f"[Simulator] Processing file: {filepath}")

        Event_type = data.get("EventType", "UNKNOWN")
        if Event_type == "PotentialMerger":
            octopuscommunicator.handle_potential_merger_message(data)
        else:
            merger_agent.logger.debug(f"[Simulator] Received non-merger event: {Event_type}")

        time.sleep(0.5)  # optional: simulate delay between events   