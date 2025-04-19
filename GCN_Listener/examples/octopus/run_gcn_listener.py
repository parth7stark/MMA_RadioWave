import argparse
from omegaconf import OmegaConf
from mma_gcn.agent import GCNAgent
from mma_gcn.communicator.octopus import OctopusGCNCommunicator
from gcn_kafka import Consumer as GCNConsumer
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
gcn_agent_config = OmegaConf.load(args.config)

# Initialize gcn-side modules
gcn_agent = GCNAgent(gcn_agent_config=gcn_agent_config)

# Create Octopus communicator for publishing events to Radio topic - 
octopuscommunicator = OctopusGCNCommunicator(
    gcn_agent,
    logger=gcn_agent.logger,
)

# Create communicator for GCN Kafka
# For testing, we can have earliest but change to latest for deployment and add group id to read form last read message

gcnconsumer = GCNConsumer(
    config={'auto.offset.reset': 'earliest'},  # Start from earliest message
    client_id=gcn_agent_config.gcn_listener_configs.comm_configs.gcn_kafka_configs.gcn_client_id,
    client_secret=gcn_agent_config.gcn_listener_configs.comm_configs.gcn_kafka_configs.gcn_client_secret,
    domain=gcn_agent_config.gcn_listener_configs.comm_configs.gcn_kafka_configs.kafka_broker
)

gcn_topics = gcn_agent_config.gcn_listener_configs.comm_configs.gcn_kafka_configs.kafka_gcn_topic
gcnconsumer.subscribe(gcn_topics)


octopuscommunicator.publish_GCN_listener_started_event()

print("[GCN Listener] Started listening for LVK notices and circulars...", flush=True)
gcn_agent.logger.info("[GCN Listener] Started listening for LVK notices and circulars...")

# refer igwn, gcn and jupyter-notebook sample code
if gcn_agent_config.gcn_listener_configs.simulate_events == "no":
    for msg in gcnconsumer:
        topic = msg.topic
        
        # try:
        data_str = msg.value.decode("utf-8")  # decode the notice/circular content to string
        
        if topic == "gcn.classic.voevent.LVC_COUNTERPART":
            octopuscommunicator.handle_lvk_counterpart_notice(data_str)
        elif topic == "gcn.circulars" :
            octopuscommunicator.handle_json_circulars(data_str)
        elif topic == "igwn.gwalert":
            octopuscommunicator.handle_json_lvk_notices(data_str)
        else:
            print(f"[GCN Listener] Message from unknown topic encountered ({topic})", flush=True)
            gcn_agent.logger.error(f"[GCN Listener] Message from unknown topic encountered ({topic})")
else:
    """
    Simulate GCN listener by reading notices/circulars from files.
    """
    event_dir = gcn_agent_config.gcn_listener_configs.simulation_datadir
    event_files = sorted(glob(os.path.join(event_dir, "*.*")))
    print(f"[Simulator] Found {len(event_files)} GCN event files to process.")
    
    for filepath in event_files:
        topic = None
        ext = os.path.splitext(filepath)[1].lower()
        
        # Infer topic from filename or extension
        if "counterpart" in filepath:
            topic = "gcn.classic.voevent.LVC_COUNTERPART"
        elif "circular" in filepath:
            topic = "gcn.circulars"
        elif "initial" in filepath or "update" in filepath or "retraction" in filepath:
            topic = "igwn.gwalert"
        else:
            gcn_agent.logger.warning(f"[Simulator] Unknown file format or naming: {filepath}")
            continue

        with open(filepath, 'r') as f:
            data_str = f.read()

        print(f"[Simulator] Processing file: {filepath}, Topic: {topic}")
        gcn_agent.logger.info(f"[Simulator] Processing GCN event from file: {filepath}")

        if topic == "gcn.classic.voevent.LVC_COUNTERPART":
            octopuscommunicator.handle_lvk_counterpart_notice(data_str)
        elif topic == "gcn.circulars":
            octopuscommunicator.handle_json_circulars(data_str)
        elif topic == "igwn.gwalert":
            octopuscommunicator.handle_json_lvk_notices(data_str)
        else:
            print(f"[Simulator] Unknown topic encountered ({topic})")
            gcn_agent.logger.error(f"[Simulator] Unknown topic encountered ({topic})")    
 

    # except json.JSONDecodeError as e:
    #     # Handle invalid JSON messages
    #     print(f"[Server] JSONDecodeError for message from topic ({topic}): {e}", flush=True)
    #     server_agent.logger.error(f"[Server] JSONDecodeError for message from topic ({topic}): {e}")
    
    # except Exception as e:
    #     # Catch-all for other unexpected exceptions
    #     """Octopus down or got a message which doesn't have 'EventType' key"""
        
    #     # Log the traceback
    #     tb = traceback.format_exc()

    #     print(f"[Server] Unexpected error while processing message from topic ({topic}): {e}", flush=True)
    #     print(f"[Server] Raw message: {msg}", flush=True)
    #     print(f"[Server] Traceback: {tb}", flush=True)

    #     server_agent.logger.error(f"[Server] Unexpected error while processing message from topic ({topic}): {e}")
    #     server_agent.logger.error(f"[Server] Raw message: {msg}")
    #     server_agent.logger.error(f"[Server] Traceback: {tb}")