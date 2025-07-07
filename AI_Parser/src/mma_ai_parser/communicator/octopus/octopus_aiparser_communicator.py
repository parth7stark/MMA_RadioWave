import logging
from typing import Optional
from omegaconf import OmegaConf
from mma_overlap_analysis.agent import AIParserAgent
from mma_overlap_analysis.logger import ServerAgentFileLogger
from diaspora_event_sdk import KafkaProducer, KafkaConsumer

class OctopusAIParserCommunicator:
    """
    Octopus communicator for AI Parser module. It contains functions that parse the incoming alerts and publish a message to octopus event fabric
    Contains functions to produce different events
    """

    def __init__(
        self, 
        ai_parser_agent: AIParserAgent,
        logger: Optional[ServerAgentFileLogger] = None,
    ):
        
        self.ai_parser_agent = ai_parser_agent
        self.logger = logger if logger is not None else self._default_logger()
        
        # MMA topic: topic where Estimator publishes posterior samples for overllaped analysis
        self.ai_parser_topic = self.ai_parser_agent.ai_parser_config.overlap_analysis_configs.comm_configs.octopus_configs.ai_parser_topic.topic


        # Kafka producer for control messages and sending Embeddings
        self.producer = KafkaProducer()


        ai_parser_group_id = self.ai_parser_agent.ai_parser_config.overlap_analysis_configs.comm_configs.octopus_configs.ai_parser_topic.group_id
        # estimator_ai_parser_group_id = self.ai_parser_agent.ai_parser_config.overlap_analysis_configs.comm_configs.octopus_configs.afterglow_topic.group_id
        
        self.consumer = KafkaConsumer(
            self.ai_parser_topic,
            enable_auto_commit=True,
            auto_offset_reset="earliest",  # This ensures it reads all past messages
            group_id=ai_parser_group_id
        )

        # self.estimator_topic_consumer = KafkaConsumer(
        #     self.estimator_topic,
        #     enable_auto_commit=True,
        #     auto_offset_reset="earliest",  # This ensures it reads all past messages
        #     group_id=estimator_ai_parser_group_id
        # )

    def publish_ai_parser_started_event(self):
        """
        Publishes an event to the control topic indicating that the AI Parser has started listening for incoming GCN
        """

        # Now publish "ParameterEstimatorStarted"
        ready_msg = {
            "EventType": "AIParserStarted",
            "details": "[AI Parser] Started AI Parser...",
        }
        self.producer.send(self.ai_parser_topic, ready_msg)
        self.producer.flush()

        print("Published AI Parser started event.", flush=True)
        self.logger.info("Published AI Parser started event.")

    def send_NewRadioGCN_event(self, filepath, supereventID):
        """
        Publishes an event to the control topic indicating that the GCN is a radio GCN and contains flux time information
        """

        # Now publish "ParameterEstimatorStarted"
        ready_msg = {
            "EventType": "NewRadioGCN",
            "SuperEventID": supereventID,
            "details": "[AI Parser] New Radio GCN detected for superevent...",
            "filepath": filepath
        }
        self.producer.send(self.ai_parser_topic, ready_msg)
        self.producer.flush()

        print("Published NewRadioGCN detected event.", flush=True)
        self.logger.info("Published NewRadioGCN detected event.")

    def send_NewFluxTimeDataAdded_event(self, supereventID, flux_time_info, csv_filepath):
        """
        Publishes an event to the control topic indicating that a new flux time information is added to the csv
        """

        # Now publish "ParameterEstimatorStarted"
        ready_msg = {
            "EventType": "NewFluxTimeDataAdded",
            "details": "[AI Parser] New Flux Time Data found in GCN...",
            "SuperEventID": supereventID,
            "flux-time-info": flux_time_info,
            "filepath": csv_filepath
        }
        self.producer.send(self.ai_parser_topic, ready_msg)
        self.producer.flush()

        print("Published NewRadioGCN detected event.", flush=True)
        self.logger.info("Published NewRadioGCN detected event.")

    
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
