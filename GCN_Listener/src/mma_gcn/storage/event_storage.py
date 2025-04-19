import os
import json
import threading
from datetime import datetime, timedelta
from .time_utils import parse_utc_time

class EventStorage():
    """
    EventStorage:

    Handles storage and retrieval of potential merger events from GW module and superevent ids from GCN
    """  
    def __init__(
        self,
        gcn_config,
        logger,
        **kwargs
    ):
        
        self.gcn_config = gcn_config
        self.logger = logger
        self.__dict__.update(kwargs)
        
        # Set up storage
        # Path to save gcn alerts in
        self.storage_dir = gcn_config.gcn_listener_configs.storage.gcn_alerts_storage_path
        self.potential_mergers_file = gcn_config.gcn_listener_configs.storage.gw_potential_merger_filepath
        self.events_of_interest_file = gcn_config.gcn_listener_configs.storage.events_of_interest_filepath

        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize storage files if they don't exist
        self._initialize_storage_files()

        # Thread lock for file operations
        # self.file_lock = threading.Lock()

    def _initialize_storage_files(self):
        """Initialize storage files if they don't exist."""
        if not os.path.exists(self.potential_mergers_file):
            with open(self.potential_mergers_file, 'w') as f:
                json.dump([], f)
                
        if not os.path.exists(self.events_of_interest_file):
            with open(self.events_of_interest_file, 'w') as f:
                json.dump([], f)
                
    def store_merger_event(self, detection_details):
        """
        Store a potential merger event that comes from GW module
        
        Args:
            detection_details: List of detection details
            value={
        
            "EventType": "PotentialMerger",
            "detection_details": detection_details
        })
        """
        # with self.file_lock:
        # Load existing data
        try:
            with open(self.potential_mergers_file, 'r') as f:
                mergers = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            mergers = []
        
        # Add new detection details with timestamp
        timestamp = datetime.utcnow().isoformat()
        # This timestamp is the time when merger event was added to file
        entry = {
            "time_added": timestamp,
            "detection_details": detection_details
        }
        
        # Append new entry
        mergers.append(entry)
        
        # Write back to file
        with open(self.potential_mergers_file, 'w') as f:
            json.dump(mergers, f, indent=2)
                
    def find_matching_merger(self, utc_time, tolerance_minutes=5):
        """
        Find a matching merger event based on time proximity.
        
        Args:
            utc_time: UTC time to match
            Time of event in GCN notice is in following format (UTC, ISO-8601), e.g. 2018-11-01T22:22:46.654Z
            tolerance_minutes: Time tolerance in minutes
            
        Returns:
            dict: Matching detection details or None
        """
        # with self.file_lock:
        try:
            with open(self.potential_mergers_file, 'r') as f:
                mergers = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
            
        # Parse the input time
        if isinstance(utc_time, str):
            target_time = parse_utc_time(utc_time)
        else:
            target_time = utc_time
            
        # Calculate time window
        time_window_start = target_time - timedelta(minutes=tolerance_minutes)
        time_window_end = target_time + timedelta(minutes=tolerance_minutes)
        
        # Check each merger event
        for merger in mergers:
            for detail in merger.get("detection_details", []):
                detection_time = parse_utc_time(detail.get("UTC_time"))
                
                # Check if detection time is within tolerance window -- they all are datetime objects so I can compare
                if time_window_start <= detection_time <= time_window_end:
                    return detail
                    
        return None
            
    def store_event_of_interest(self, event_id, event_data):
        """
        Store an event of interest in events of interest file.
        In this file we maintain list of observed bns events until now
        
        Args:
            event_id: Event ID/superevent ID
            event_data: Event data to store
            # Store as event of interest
                self.storage.store_event_of_interest(superevent_id, {
                    "bns_probability": bns_probability,
                    "event_time": event_time,
                    "matching_merger": matching_merger
                })
        """
        # with self.file_lock:
        try:
            with open(self.events_of_interest_file, 'r') as f:
                events = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            events = []
            
        # Check if event already exists
        for event in events:
            if event.get("event_id") == event_id:
                # Update existing event
                event.update(event_data)
                break
        else:
            # Add new event
            event_data["event_id"] = event_id
            event_data["created_at"] = datetime.utcnow().isoformat()
            events.append(event_data)
            
        # Write back to file
        with open(self.events_of_interest_file, 'w') as f:
            json.dump(events, f, indent=2)

    def get_known_bns_events(self):
        """
        Get all the events of interest stored in events of interest file.
        Return a list of all known BNS event IDs from the events of interest file.
        Returns:
            List of event IDs (strings)
        """
        # with self.file_lock:
        try:
            with open(self.events_of_interest_file, 'r') as f:
                events = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

        return [event.get("event_id") for event in events if "event_id" in event]
            
    def is_event_of_interest(self, event_id):
        """
        Check if an event is of interest.        
        Args:
            event_id: Event ID to check
            
        Returns:
            bool: True if event is of interest, False otherwise
        """
        # with self.file_lock:
        try:
            with open(self.events_of_interest_file, 'r') as f:
                events = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return False
            
        # Check if event exists
        for event in events:
            if event.get("event_id") == event_id:
                return True
                
        return False
            
    def get_event_directory(self, event_id):
        """
        Get the directory for an event.
        
        Args:
            event_id: Event ID
            
        Returns:
            str: Path to event directory
        """
        event_dir = os.path.join(self.storage_dir, event_id)
        os.makedirs(event_dir, exist_ok=True)
        return event_dir