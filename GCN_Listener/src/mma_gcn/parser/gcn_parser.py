import os
import voeventparse
import json
import time
from datetime import datetime

class GCNParser():
    """
    GCNParser:
        This class contains parser for different notice and circular type
    """  
    def __init__(
        self,
        gcn_config,
        storage_object,
        logger,
        **kwargs
    ):
        
        self.gcn_config = gcn_config
        self.logger = logger
        self.__dict__.update(kwargs)
        
        # Set up storage
        self.storage = storage_object

                                
    def CounterpartNoticeParser(self, xml_str):
        """
        Process LVK VOEvent observation notices and save to graceid directory.
        
        Args:
            xml_str: VOEvent XML string
            output_base_dir: Base directory to save gcn alerts
        
        Returns:
            Path to saved file if processed, None if not observation event
        """

        v = voeventparse.loads(xml_str)
    
        # Check if observation event
        if v.attrib['role'] != 'observation':
            alertofinterest = "No"
            return alertofinterest, None, None
        
        alertofinterest = "Yes"

        # Extract GraceID from parameters
        params = voeventparse.get_grouped_params(v)
        graceid = params[None]['GraceID']['value']  # None group = top-level params
        
        self.logger.info(f"Processing counterpart notice for {graceid}")
        
        # Create output directory
        # Save notice to event directory
        event_dir = self.storage.get_event_directory(graceid)
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        counterpartfile = os.path.join(event_dir, f"{graceid}_Counterpart_{timestamp}.xml")
        
        with open(counterpartfile, 'w') as f:
            f.write(xml_str)
            
        # Old lines
        # output_dir = os.path.join(self.gcn_data_dir, graceid)
        # os.makedirs(output_dir, exist_ok=True)
        
        # Save original XML content
        # output_path = os.path.join(output_dir, f"{graceid}.xml")
        # with open(output_path, 'w') as f:
        #     f.write(xml_str)
        
        return alertofinterest, graceid, counterpartfile
    
    def JSONNoticeParser(self, data_str):
        """
        
        We return alertofInterest variable and metadata for octopus message like superevent_id, notice contents
        Returns:
            alertofInterest: variable indicates whether event in gcn alert is of interest or not
            superevent_id: Id of the event
            data: Notice data in json format
        """

        record = json.loads(data_str)

        # Check if superevent event. Ignore mock and test event (MS or TS)
        if record['superevent_id'][0] != 'S':
            alertofinterest = "No"
            return alertofinterest, None, None, None

        if record['alert_type'] == 'INITIAL':
            alertofinterest, superevent_id, data, file_path = self._process_initial_notice(record)
            # print(record['superevent_id'], 'was detected')
            return alertofinterest, superevent_id, data, file_path
        elif record['alert_type'] in ["UPDATE", "RETRACTION"]:
            alertofinterest, superevent_id, data, file_path = self._process_followup_notice(record)
            # print(record['superevent_id'], 'was detected')
            return alertofinterest, superevent_id, data, file_path
        else:
            #  ignore message 
            alertofinterest = "No"
            return alertofinterest, None, None, None
    
    def JSONCircularParser(self, data_str):
        """
        
        We return alertofInterest variable and metadata for octopus message like superevent_id, notice contents
        Returns:
            alertofInterest: variable indicates whether event in gcn alert is of interest or not
            superevent_id: Id of the event
            data: Notice data in json format
        """

        record = json.loads(data_str)
        alertofinterest, superevent_id, data, file_path = self._process_followup_circular(record)
        return alertofinterest, superevent_id, data, file_path

    def _process_initial_notice(self, data):
        """
        Process an initial LVK notice. We use initial notice to determine whether the detected event is of interest or not
        
        Args:
            data: Notice data in json format
        """
        # Extract relevant fields
        classification = data.get("event", {}).get("classification", {})
        bns_probability = classification.get("BNS", 0.0)
        superevent_id = data.get("superevent_id", "unknown_event")
        event_time = data.get("event", {}).get("time", {})
        
        if not superevent_id:
            self.logger.warning("Received Counterpart notice without superevent_id")
            return "No", None, None, None
        
        self.logger.info(f"Processing initial notice for {superevent_id}, BNS prob: {bns_probability}")
        
        # Check if it's a BNS candidate
        if bns_probability > 0.5:
            self.logger.info(f"BNS candidate detected: {superevent_id}")
            
            # Find matching potential merger
            matching_merger = self.storage.find_matching_merger(
                event_time, 
                tolerance_minutes=self.time_tolerance
            )
            
            if matching_merger:
                self.logger.info(f"Found matching merger event for {superevent_id}")
                
                # Create event directory and save notice
                event_dir = self.storage.get_event_directory(superevent_id)
                timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
                notice_file = os.path.join(event_dir, f"{superevent_id}_initial_notice_{timestamp}.json")
                
                with open(notice_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Store as event of interest
                self.storage.store_event_of_interest(superevent_id, {
                    "bns_probability": bns_probability,
                    "event_time": event_time,
                    "matching_merger": matching_merger
                })
                
                # Send message to Octopus
                # self._send_bns_detection_message(superevent_id, data)

                # Return arguments to the above function and call _send_bns_detection_message in communicator.py
                return "Yes", superevent_id, data, notice_file
            else:
                self.logger.info(f"No matching merger event found for {superevent_id}")
                return "No", None, None, None
        else:
            self.logger.info(f"Not a BNS candidate (prob={bns_probability}): {superevent_id}")
            return "No", None, None, None
    
    def _process_followup_notice(self, data):
        """
        Process a followup LVK notice like Update and retraction, counterpart (counterpart have seprate function)
        
        Args:
            data: Notice data
        """

        # Extract relevant fields
        superevent_id = data.get("superevent_id", "unknown_event")
        notice_type = data.get("alert_type")
        
        self.logger.info(f"Processing {notice_type} notice for {superevent_id}")
        
        # Check if it's an event of interest
        if self.storage.is_event_of_interest(superevent_id):
            self.logger.info(f"Routing {notice_type} notice for event of interest: {superevent_id}")
            
            # Save notice to event directory
            event_dir = self.storage.get_event_directory(superevent_id)
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
            notice_file = os.path.join(event_dir, f"{superevent_id}_{notice_type}_notice_{timestamp}.json")
            
            with open(notice_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            # Send message about new notice
            # self._send_notice_update_message(superevent_id, notice_type)
            return "Yes", superevent_id, data, notice_file
        else:
            self.logger.info(f"Ignoring {notice_type} notice for {superevent_id} (not an event of interest)")
            return "No", None, None, None
        
    def _process_followup_circular(self, circular):
        """
        Process a followup circulars
        
        Args:
            data: Circular data in JSON format
        """
        subject = circular.get('subject', '')
        event_id = circular.get('eventId', '')
        body = circular.get('body', '')
        circular_id = circular.get('circularId', '')
        
        self.logger.info(f"Processing circular {circular_id}: {subject}")
        
        # First, check if any of our known BNS event IDs are mentioned
        bns_event_id = self.find_bns_event_in_circular(subject, event_id, body)
        
        if bns_event_id:
            self.logger.info(f"Circular is related to BNS event {bns_event_id}")
            
            # Save the circular
            # Save notice to event directory
            event_dir = self.storage.get_event_directory(bns_event_id)
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
            circular_path = os.path.join(event_dir, f"{bns_event_id}_circular_{circular_id}_{timestamp}.json")
            

            # event_dir = os.path.join(BASE_DIR, bns_event_id)
            # os.makedirs(event_dir, exist_ok=True)
            
            # file_path = os.path.join(event_dir, f"circular_{circular_id}.json")
            
            with open(circular_path, 'w') as f:
                json.dump(circular, f, indent=2)
                
            self.logger.info(f"Saved circular to {circular_path}")
            
            # Send message to Octopus
            # self.send_to_octopus("New Radio circular added", 
            #                     metadata={"file_path": file_path, "event_id": bns_event_id})
            return "Yes", bns_event_id, circular, circular_path
    
    def find_bns_event_in_circular(self, subject, event_id, body):
        """
        Check if the circular mentions any known BNS event
        
        Args:
            subject: Circular subject
            event_id: Circular event ID
            body: Circular body text
            
        Returns:
            BNS event ID if found, None otherwise
        """
        # Check all our known BNS events
        # for bns_id in self.bns_events:
        #     if (bns_id in subject or bns_id in event_id or bns_id in body):
        #         return bns_id
        
        bns_events = self.storage.get_known_bns_events()
        for bns_id in bns_events:
            if (bns_id in subject or bns_id in event_id or bns_id in body):
                return bns_id
        
        return None
    
    
    def _send_bns_detection_message(self, superevent_id, data):
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
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._send_kafka_message(message)

    def _send_notice_update_message(self, superevent_id, notice_type):
        """
        Send a message to Octopus about a notice update.
        
        Args:
            superevent_id: Superevent ID
            notice_type: Notice type (update, retraction, counterpart)
        """
        message = {
            "EventType": "SupereventNoticeUpdate",
            "superevent_id": superevent_id,
            "notice_type": notice_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._send_kafka_message(message)
    
    def _send_kafka_message(self, message):
        """
        Send a message to the Kafka topic.
        
        Args:
            message: Message to send
        """
        try:
            self.producer.produce(
                self.radio_topic,
                value=json.dumps(message).encode('utf-8')
            )
            self.producer.flush()
            self.logger.info(f"Sent message to {self.radio_topic}: {message.get('EventType')}")
        except Exception as e:
            self.logger.error(f"Failed to send Kafka message: {str(e)}")