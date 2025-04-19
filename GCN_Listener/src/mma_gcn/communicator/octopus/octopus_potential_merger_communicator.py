import json
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
        self.topic = self.merger_agent.gcn_agent_config.gcn_listener_configs.comm_configs.octopus_configs.merger_topic.topic


        # Kafka producer for control messages and sending Embeddings
        self.producer = KafkaProducer()


        merger_listener_group_id = self.merger_agent.gcn_agent_config.gcn_listener_configs.comm_configs.octopus_configs.merger_topic.topic
        self.consumer = KafkaConsumer(
            self.topic,
            enable_auto_commit=True,
            auto_offset_reset="earliest",  # This ensures it reads all past messages
            group_id=merger_listener_group_id
        )

    def on_Merger_Listener_started(self, data):
        """
        Publishes an event to the control topic indicating that the Merger listener has started listening for Merger events from GW module
        """

        # Now publish "MergerListenerStarted"
        ready_msg = {
            "EventType": "MergerListenerStarted",
            "details": "[Merger Listener] Started listening for potential merger events from GW module...",
        }
        self.producer.send(self.topic, ready_msg)
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
        
        self.logger.info(f"Received PotentialMerger event")
        
        # Extract detection details
        detection_details = data.get("detection_details", [])
        
        if detection_details:
            # Store the potential merger event
            self.storage.store_merger_event(detection_details)
            self.logger.info(f"Stored {len(detection_details)} detection details")
            
            # Log some details for debugging
            for detail in detection_details:
                self.logger.info(f"GPS Time: {detail.get('GPS_time')} -> UTC Time: {detail.get('UTC_time')}")
        else:
            self.logger.warning("Received PotentialMerger event with no detection details")
    

        # Build the JSON payload
        # Convert NumPy arrays to lists to avoid serialization issues
        data = {
            "EventType": "LocalMCMCDone",
            "site_id": client_id,
            "status": "DONE",
            'chain': chains_b64,
            # 'chain': local_results["chain"].tolist() if isinstance(local_results["chain"], np.ndarray) else local_results["chain"],
            'min_time': float(local_results["min_time"]) if isinstance(local_results["min_time"], np.generic) else local_results["min_time"],
            'max_time': float(local_results["max_time"]) if isinstance(local_results["max_time"], np.generic) else local_results["max_time"],
            'unique_frequencies': local_results["unique_frequencies"].tolist() if isinstance(local_results["unique_frequencies"], np.ndarray) else local_results["unique_frequencies"]
        }

    
        self.producer.send(
            self.topic,
            value=data
        )

        self.producer.flush()

        print(f"[Site {client_id}] Sent Local Posterior Samples", flush=True)
        self.logger.info(f"[Site {client_id}] Sent Local Posterior Samples")

        return
    
    def get_best_estimate(self, data):
        theta_est = data["theta_est"]
        global_min_time = data["global_min_time"]
        global_max_time = data["global_max_time"]
        unique_frequencies = data["unique_frequencies"]

        print("Best estimate of parameters", flush=True)
        self.logger.info("Best estimate of parameters")
        params = ['log(E0)','thetaObs','thetaCore','log(n0)',
                  'log(epsilon_e)','log(epsilon_B)','p']
        
        ndim = 7
        for i in range(ndim):
            if i in [0, 3, 4, 5]:  # Reverse the transformation for log-space parameters
                log_value = np.log10(theta_est[i])  # Convert back to log-space
                print(f'{params[i]} = {log_value:.2f}', flush=True)
                self.logger.info(f'{params[i]} = {log_value:.2f}')
            else:
                print(f'{params[i]} = {theta_est[i]:.2f}', flush=True)
                self.logger.info(f'{params[i]} = {theta_est[i]:.2f}')

        return theta_est, global_min_time, global_max_time, unique_frequencies

    def handle_proposed_theta_message(self, data, local_data):
        
        if '_client_id' in kwargs:
            client_id = str(kwargs["_client_id"])
            del kwargs["_client_id"]
        else:
            client_id = str(self.client_id)

        iteration_no = data["iteration_no"]
        theta = data["theta"]


        print(f"Site {client_id} received proposed theta.")
        self.logger.info(f"Site {client_id} received proposed theta.")

        
        log_likelihood = client_agent.compute_log_likelihood(theta, local_data)
        
        # Build the JSON payload
        data = {
            "EventType": "LogLikelihoodComputed",
            "site_id": client_id,
            'local_likelihood': log_likelihood,
        }
    
        self.producer.send(
            self.topic,
            value=data
        )

        self.producer.flush()

        print(f"[Site {client_id}] Sent Local log-likelihood", flush=True)
        self.logger.info(f"[Site {client_id}] Sent Local log-likelihood")

        return
    
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

