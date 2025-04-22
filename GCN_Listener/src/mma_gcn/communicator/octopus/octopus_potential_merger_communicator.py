import json
import logging
from omegaconf import OmegaConf, DictConfig
from typing import Union, Dict, OrderedDict, Tuple, Optional, Any
from mma_gcn.agent import MergerListenerAgent
from mma_gcn.logger import ServerAgentFileLogger
import numpy as np
from diaspora_event_sdk import KafkaProducer, KafkaConsumer


class OctopusPMCommunicator:
    """
    Octopus communicator for potential merger event listener.
    It consumes potential merger events from GW module and stores it in a file
    Contains functions to produce/consume/handle different events
    """
    def __init__(
        self, 
        merger_agent: MergerListenerAgent,
        logger: Optional[ServerAgentFileLogger] = None,
    ):
        
        self.merger_agent = merger_agent
        self.logger = logger if logger is not None else self._default_logger()
        
        # Merger topic: topic where GW event publishes merger events
        self.merger_topic = self.merger_agent.merger_agent_config.gcn_listener_configs.comm_configs.octopus_configs.merger_topic.topic
        self.radio_topic = self.merger_agent.merger_agent_config.gcn_listener_configs.comm_configs.octopus_configs.radio_topic.topic


        # Kafka producer for control messages and sending Embeddings
        self.producer = KafkaProducer()


        merger_listener_group_id = self.merger_agent.merger_agent_config.gcn_listener_configs.comm_configs.octopus_configs.merger_topic.topic
        self.consumer = KafkaConsumer(
            self.merger_topic,
            enable_auto_commit=True,
            auto_offset_reset="earliest",  # This ensures it reads all past messages
            group_id=merger_listener_group_id
        )

    def on_Merger_Listener_started(self):
        """
        Publishes an event to the control topic indicating that the Merger listener has started listening for Merger events from GW module
        """

        # Now publish "MergerListenerStarted"
        ready_msg = {
            "EventType": "MergerListenerStarted",
            "details": "[Merger Listener] Started listening for potential merger events from GW module...",
        }
        self.producer.send(self.radio_topic, ready_msg)
        self.producer.flush()

        print("Published Merger Listener started event.", flush=True)
        self.logger.info("Published Merger Listener started event.")


    def handle_potential_merger_message(self, data):
        """
        Encountered a potential merger event, handle it (store it)
        :param data: Potential Merger event message
        value={
        
            "EventType": "PotentialMerger",
            "detection_details": detection_details
        }
        :return: None for async and if sync communication return Metadata containing the server's acknowledgment status.
        """
        
        self.logger.info(f"[Merger Listener] Received PotentialMerger event")
        
        # Extract detection details
        detection_details = data.get("detection_details", [])
        
        if detection_details:
            # Store the potential merger event
            self.merger_agent.storage.store_merger_event(detection_details)
            self.logger.info(f"Stored {len(detection_details)} detection details")
            
            # Log some details for debugging
            for detail in detection_details:
                self.logger.info(f"GPS Time: {detail.get('GPS_time')} -> UTC Time: {detail.get('UTC_time')}")
        else:
            self.logger.warning("Received PotentialMerger event with no detection details")
    
    
    def _default_logger(self):
        """Create a default logger for the server if no logger provided."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter('[%(asctime)s %(levelname)-4s server]: %(message)s')
        s_handler = logging.StreamHandler()
        s_handler.setLevel(logging.INFO)
        s_handler.setFormatter(fmt)
        logger.addHandler(s_handler)
        return logger

