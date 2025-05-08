import logging
from typing import Optional
from omegaconf import OmegaConf
from mma_param_estimation.agent import ParamEstimatorAgent
from mma_param_estimation.logger import ServerAgentFileLogger
from diaspora_event_sdk import KafkaProducer, KafkaConsumer
from .utils import serialize_tensor_to_base64, deserialize_tensor_from_base64
import torch

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
        self.radio_topic = self.estimator_agent.estimator_config.bns_parameter_estimation_configs.comm_configs.octopus_configs.radio_topic.topic
        self.mma_topic = self.estimator_agent.estimator_config.bns_parameter_estimation_configs.comm_configs.octopus_configs.mma_topic.topic


        # Kafka producer for control messages and sending Embeddings
        self.producer = KafkaProducer()


        estimator_group_id = self.estimator_agent.estimator_config.bns_parameter_estimation_configs.comm_configs.octopus_configs.radio_topic.group_id
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


    def send_posterior_samples(self, posterior_df):
        """
        We need to send the posterior_samples of inclination and distance to the MMA_module for overlap analysis
        :param posterior_dict: 
        Sample posterior_dict: Instead of dictionary, sending dataframe
        {
          "theta_jn": [0.4, 0.41, 0.39, 0.42, ...],
          "luminosity_distance": [40.2, 41.0, 39.8, 40.5, ...]
        }
        :return: None for async and if sync communication return Metadata containing the server's acknowledgment status.
        """
        param_names = posterior_df.columns.tolist()

        # Step 2: Convert to tensor for transfer
        tensor_samples = torch.tensor(posterior_df.to_numpy())  # Shape [N, num_params]

        # Step 3: Proxy if needed
        if self.estimator_agent.use_proxystore:
            tensor_samples = self.estimator_agent.proxystore.proxy(tensor_samples)
            self.logger.info(f"Posterior samples proxied via ProxyStore.")

        # Step 4: Serialize tensor to base64
        payload_b64 = serialize_tensor_to_base64(tensor_samples)

        # Step 5: Build the Kafka JSON payload
        data = {
            "EventType": "DingoPosteriorSamplesReady",
            "parameters": param_names,
            "posterior_samples": payload_b64
        }

        self.producer.send(
            self.mma_topic,
            value=data
        )

        self.producer.flush()

        self.logger.info(f"Send Dingo Posterior Samples to MMA module")
        return    
    
    # Add this on MMA module side
    def handle_dingo_posterior_samples_message(self, data):
        """
        Handle a Kafka message containing posterior samples payload.
        This extracts the proxied or normal posterior tensor and returns it as a list.

        Args:
            data (dict): Kafka message already parsed as dict (after json.loads).

        Returns:
            dict: A dictionary mapping parameter names to list of posterior samples.
        """
        event_type = data.get("EventType")
        if event_type != "PosteriorSamplesReady":
            self.logger.warning(f" Received unexpected event type: {event_type}")
            return None

        param_names = data["parameters"]  # List of parameter names
        posterior_b64 = data["posterior_samples"]

        # Step 1: Deserialize tensor
        posterior_tensor = deserialize_tensor_from_base64(posterior_b64)

        # Step 2: Extract if Proxy
        if isinstance(posterior_tensor, Proxy):
            self.logger.info("🔗 Extracting posterior tensor from ProxyStore.")
            posterior_tensor = extract(posterior_tensor)

        # Step 3: Convert tensor to structured dictionary
        posterior_array = posterior_tensor.numpy()  # Shape: [num_samples, num_parameters]
        posterior_dict = {param: posterior_array[:, idx].tolist() for idx, param in enumerate(param_names)}

        self.logger.info(f" Received posterior samples for parameters: {param_names}")
        return posterior_dict

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
