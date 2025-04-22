import json
import logging
from typing import Optional
from omegaconf import OmegaConf
from mma_gcn.agent import GCNAgent
from mma_gcn.logger import ServerAgentFileLogger
from diaspora_event_sdk import KafkaProducer, KafkaConsumer
import json
import time
from datetime import datetime

class OctopusGCNCommunicator:
    """
    Octopus communicator for GCN listener. It contains functions that parse the incoming alerts and publish a message to octopus event fabric
    Contains functions to produce different events
    """

    def __init__(
        self,
        gcn_agent: GCNAgent,
        logger: Optional[ServerAgentFileLogger] = None,
    ) -> None:

        self.gcn_agent = gcn_agent
        self.logger = logger if logger is not None else self._default_logger()

        self.topic = self.gcn_agent.gcn_agent_config.gcn_listener_configs.comm_configs.octopus_configs.radio_topic.topic

        # Kafka producer for publishing messages
        self.producer = KafkaProducer()

    def publish_GCN_listener_started_event(self):
        """
        Publishes an event to the control topic indicating that the GCN listener has started listening for LVK notices and circulars
        """
        
        event = {
            "EventType": "GCNListenerStarted",
            "details": "[GCN Listener] Started listening for LVK notices and circulars..."
        }

        self.producer.send(self.topic, value=event)
        self.producer.flush()
        
        print("[GCN Listener] Published GCN Listener started event.", flush=True)
        self.logger.info("[GCN Listener] Published GCN Listener started event.")

    def handle_lvk_counterpart_notice(self, data_str):
        """
        Received an counterpart notice from LVK in VOEvent format. Use voevent parser to process it
        """

        alertofinterest, superevent_id, file_path  = self.gcn_agent.parser.CounterpartNoticeParser(data_str)
        if alertofinterest == "Yes":
            # Publish appropriate message on octopus 
            message = {
            "EventType": "CounterpartDetected",
            "superevent_id": superevent_id,
            "filepath": file_path,
            "timestamp": datetime.utcnow().isoformat()
            }
            
            self.producer.send(self.topic, value=message)
            self.producer.flush()
            
            # print("[GCN Listener] Published CounterpartDetected event.", flush=True)
            self.logger.info("[GCN Listener] Published CounterpartDetected event.")

    def handle_json_lvk_notices(self, data_str):
        """
        Received other notice from LVK and circulars in JSON format. Use json parser to process it
        """

        alertofinterest, superevent_id, data, file_path = self.gcn_agent.parser.JSONNoticeParser(data_str)        
        # If superevent id is None then ignore that alert. Alert not of interest       
        if alertofinterest == "Yes":
            alertType = data["alert_type"]
            # Publish appropriate message on octopus
            if alertType == "INITIAL":
                self._send_bns_detection_message(superevent_id, data, file_path)
            elif alertType in ["UPDATE", "RETRACTION"]:
                self._send_received_update_message(superevent_id, data, file_path)
        
    
    def handle_json_circulars(self, data_str):
        """
        Received other notice from LVK and circulars in JSON format. Use json parser to process it
        """

        alertofinterest, superevent_id, data, file_path = self.gcn_agent.parser.JSONCircularParser(data_str) 

        # If superevent id is None then ignore that alert. Alert not of interest       
        if alertofinterest == "Yes":
            message = {
            "EventType": "SupereventReceivedCircular",
            "superevent_id": superevent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "filepath": file_path
            }
            
            self.producer.send(self.topic, value=message)
            self.producer.flush()

            print("[GCN Listener] Published SupereventReceivedCircular event.", flush=True)
            self.logger.info("[GCN Listener] Published SupereventReceivedCircular event.")

            

    def _send_bns_detection_message(self, superevent_id, data, file_path):
        """
        Send a message to Octopus about a new BNS detection.
        
        Args:
            superevent_id: Superevent ID
            data: Original notice data
        """
        message = {
            "EventType": "NewBNSSuperevent",
            "superevent_id": superevent_id,
            "bns_probability": data.get("bns_probability"),
            "event_time": data.get("event_time"),
            "filepath": file_path,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.producer.send(self.topic, value=message)
        self.producer.flush()
        
        print("[GCN Listener] Published NewBNSSuperevent event.", flush=True)
        self.logger.info("[GCN Listener] Published NewBNSSuperevent event.")

    
    def _send_received_update_message(self, superevent_id, data, file_path):
        """
        Send a message to Octopus about a notice update.
        
        Args:
            superevent_id: Superevent ID
            notice_type: Notice type (update, retraction, counterpart)
        """
        notice_type = data.get("alert_type")
        message = {
            "EventType": "SupereventReceivedUpdate",
            "superevent_id": superevent_id,
            "notice_type": notice_type,
            "filepath": file_path,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.producer.send(self.topic, value=message)
        self.producer.flush()

        print("[GCN Listener] Published SupereventReceivedUpdate event.", flush=True)
        self.logger.info("[GCN Listener] Published SupereventReceivedUpdate event.")


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


