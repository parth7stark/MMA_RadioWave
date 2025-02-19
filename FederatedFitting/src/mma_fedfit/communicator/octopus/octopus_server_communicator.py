import json
import logging
from typing import Optional
from omegaconf import OmegaConf
from proxystore.proxy import extract
from mma_gw.agent import ServerAgent
from mma_gw.logger import ServerAgentFileLogger
from .utils import serialize_tensor_to_base64, deserialize_tensor_from_base64

from diaspora_event_sdk import KafkaProducer, KafkaConsumer

class OctopusServerCommunicator:
    """
    Octopus communicator for federated learning server.
    Contains functions to produce/consume/handle different events
    """

    def __init__(
        self,
        server_agent: ServerAgent,
        logger: Optional[ServerAgentFileLogger] = None,
    ) -> None:

        self.server_agent = server_agent
        self.logger = logger if logger is not None else self._default_logger()

        self.topic = self.server_agent.server_agent_config.server_configs.comm_configs.octopus_configs.topic


        # Kafka producer for publishing messages
        self.producer = KafkaProducer()

        # Kafka consumer to listen for control events AND embeddings
        self.consumer = KafkaConsumer(
            self.topic,
            enable_auto_commit=True,
            group_id=self.server_agent.server_agent_config.server_configs.comm_configs.octopus_configs.group_id
        )

        # Track readiness, which detector is connected and ready for inference
        self.detectors_ready = set()

        # This is to handle scenario: You do not wait for all detectors to say “ready” if you want to begin listening to embeddings as soon as any client is ready.

    def publish_server_started_event(self):
        """
        Publishes an event to the control topic indicating that the server has started,
        along with the configuration shared among all clients.
        """
        client_config =  self.server_agent.get_client_configs()
        client_config_dict = OmegaConf.to_container(client_config, resolve=True)

        event = {
            "EventType": "ServerStarted",
            "site_config": client_config_dict,
        }

        self.producer.send(self.topic, value=event)
        self.producer.flush()
        
        print("[Server] Published ServerStarted event with config.", flush=True)
        self.logger.info("[Server] Published ServerStarted event with config.")

    def handle_local_MCMC_done_message(self, data):
        """
        Message of type "LocalMCMCDone" is detected/consumed. Handle it
        Example of Message
        msg:  ConsumerRecord(topic='mma-GWwave-Triggers', partition=0, offset=7705, timestamp=1736957074944, timestamp_type=0, key=None, value=b'{"EventType": "PostProcess", "detector_id": "1", "status": "DONE", "details": "DONE -> Invoke post process pipeline", "GPS_start_time": 1264314069}', headers=[], checksum=None, serialized_key_size=-1, serialized_value_size=147, serialized_header_size=-1)
        """

        site_id = data["site_id"]    # 0 or 1
        status = data["status"]
        local_chain = data["chain"]
        min_time = data["min_time"]
        max_time = data["max_time"]
        unique_frequencies = data["unique_frequencies"]
        
        self.server_agent.aggregator.process_local_MCMC_done_message(self.producer, self.topic, site_id, status, local_chain, min_time, max_time, unique_frequencies)


    def send_proposed_theta(self, theta):

        event = {
            "EventType": "ProposedTheta",
            "iteration_no": iteration_no
            "theta": theta,
        }

        self.producer.send(self.topic, value=event)
        self.producer.flush()
        
        print("[Server] Published ProposedTheta event.", flush=True)
        self.logger.info("[Server] Published ProposedTheta event.")

    def collect_local_likelihoods(self, num_sites):
        
        log_likelihoods = {}
        # received_sites = 0

        # Track which clients have finished
        completed_clients = set()
        expected_clients = {str(i) for i in range(num_sites)}

        # while received_sites < num_sites:
        #     for message in consumer:
        #         data_str = msg.value.decode("utf-8")  # decode to string
        #         data = json.loads(data_str)          # parse JSON to dict

        #         Event_type = data["EventType"]

        #         if Event_type == "LogLikelihoodComputed"
        #             log_likelihoods[message.value["site_id"]] = message.value["log_likelihood"]
        #             received_sites += 1
        #             print(f"Received log-likelihood from Site {message.value['site_id']}")
        #             if received_sites == num_sites:
        #                 break

        for message in consumer:
            data_str = msg.value.decode("utf-8")  # decode to string
            data = json.loads(data_str)          # parse JSON to dict

            Event_type = data["EventType"]

            if Event_type == "LogLikelihoodComputed":
                site_id = data["site_id"]
                local_likelihood = data["local_likelihood"]

                print(f"[Server] Received LogLikelihoodComputed Event from site {site_id}")
                self.logger.info(f"[Server] Received LogLikelihoodComputed from site {site_id}")

                # Add the client to the completed set
                completed_clients.add(site_id)

                log_likelihoods[site_id] = local_likelihood

                if completed_clients == expected_clients:
                    print("[Server] Collected partial log likelihoods from all the sites. Compute global posterior...")
                    self.logger.info("[Server] Collected partial log likelihoods from all the sites. Compute global posterior...")
                    return log_likelihoods


        return log_likelihoods

    def handle_embeddings_message(self, data):
        """
        Message of type "SendEmbeddings" is detected/consumed. Handle it
        Example of Message
        msg:  ConsumerRecord(topic='mma-GWwave-Triggers', partition=0, offset=0, timestamp=1736989957681, timestamp_type=0, key=None, value=b'{"EventType": "SendEmbeddings", "detector_id": "0", "batch_id": 0, "shift": "preds_0", "embedding": "UEsDBAAACAg
        """

        # Extract metadata
        batch_id = data["batch_id"]
        shift = data["shift"]           # "preds_0" or "preds_5"
        det_id = data["detector_id"]    # 0 or 1
        
        # Extract and deserialize the embedding
        embedding_b64 = data["embedding"]
        local_embedding = deserialize_tensor_from_base64(embedding_b64)

        self.server_agent.aggregator.process_embeddings_message(batch_id, shift, det_id, local_embedding)
    
    def handle_post_process_message(self, data):
        """
        Message of type "PostProcess" is detected/consumed. Handle it
        Example of Message
        msg:  ConsumerRecord(topic='mma-GWwave-Triggers', partition=0, offset=7705, timestamp=1736957074944, timestamp_type=0, key=None, value=b'{"EventType": "PostProcess", "detector_id": "1", "status": "DONE", "details": "DONE -> Invoke post process pipeline", "GPS_start_time": 1264314069}', headers=[], checksum=None, serialized_key_size=-1, serialized_value_size=147, serialized_header_size=-1)
        """

        det_id = data["detector_id"]    # 0 or 1
        status = data["status"]
        GPS_start_time = data["GPS_start_time"]

        self.server_agent.aggregator.process_post_process_message(self.producer, self.topic, det_id, status, GPS_start_time)

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


