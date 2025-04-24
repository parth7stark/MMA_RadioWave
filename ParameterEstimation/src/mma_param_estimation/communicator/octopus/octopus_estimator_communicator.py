import json
import logging
from typing import Optional
from omegaconf import OmegaConf
from mma_gcn.agent import ParamEstimatorAgent
from mma_gcn.logger import ServerAgentFileLogger
from diaspora_event_sdk import KafkaProducer, KafkaConsumer
import json

class OctopusEstimatorCommunicator:
    """
    Octopus communicator for Parameter Estimator. It contains functions that parse the incoming alerts and publish a message to octopus event fabric
    Contains functions to produce different events
    """

    def __init__(
        self, 
        estimator_agent: ParamEstimatorAgent,
        logger: Optional[ServerAgentFileLogger] = None,
    ):
        
        self.estimator_agent = estimator_agent
        self.logger = logger if logger is not None else self._default_logger()
        
        # MMA topic: topic where Estimator publishes posterior samples for overllaped analysis
        self.radio_topic = self.estimator_agent.estimator_agent_config.gcn_listener_configs.comm_configs.octopus_configs.radio_topic.topic
        self.mma_topic = self.estimator_agent.estimator_agent_config.gcn_listener_configs.comm_configs.octopus_configs.mma_topic.topic


        # Kafka producer for control messages and sending Embeddings
        self.producer = KafkaProducer()


        estimator_group_id = self.estimator_agent.estimator_agent_config.gcn_listener_configs.comm_configs.octopus_configs.radio_topic.group_id
        self.consumer = KafkaConsumer(
            self.radio_topic,
            enable_auto_commit=True,
            auto_offset_reset="earliest",  # This ensures it reads all past messages
            group_id=estimator_group_id
        )

    def publish_estimator_started_event(self):
        """
        Publishes an event to the control topic indicating that the Merger listener has started listening for Merger events from GW module
        """

        # Now publish "ParameterEstimatorStarted"
        ready_msg = {
            "EventType": "ParameterEstimatorStarted",
            "details": "[Parameter Estimator] Started Parameter Estimation Process...",
        }
        self.producer.send(self.mma_topic, ready_msg)
        self.producer.flush()

        print("Published Parameter Estimator started event.", flush=True)
        self.logger.info("Published Parameter Estimator started event.")


    def send_posterior_samples(self, data):
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
            self.estimator_agent.storage.store_merger_event(detection_details)
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
