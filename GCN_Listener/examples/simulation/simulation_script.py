"""
GW Monitoring System Utilities

Utilities for simulating and testing the GW monitoring system.
"""
import json
import time
import random
import argparse
import datetime
from typing import Dict, Any
from confluent_kafka import Producer

# Example LVC Initial Notice
EXAMPLE_INITIAL_NOTICE = {
    "alert_type": "INITIAL",
    "time_created": "2025-04-15T12:36:25Z",
    "superevent_id": "S250415a",
    "urls": {
        "gracedb": "https://example.org/superevents/S250415a/view/"
    },
    "event": {
        "time": "2025-04-15T12:22:46.654Z",
        "far": 9.11069936486e-14,
        "significant": True,
        "instruments": ["H1", "L1", "V1"],
        "group": "CBC",
        "pipeline": "gstlal",
        "search": "AllSky",
        "properties": {
            "HasNS": 0.95,
            "HasRemnant": 0.91,
            "HasMassGap": 0.01
        },
        "classification": {
            "BNS": 0.95,
            "NSBH": 0.01,
            "BBH": 0.03,
            "Terrestrial": 0.01
        },
        "duration": None,
        "central_frequency": None,
        "skymap": "U0lNUExFICA9ICAgICAgICAgICAgICAgICAgICBUIC8gY29uZm..."
    },
    "external_coinc": None
}

# Example LVC Update Notice
EXAMPLE_UPDATE_NOTICE = {
    "alert_type": "UPDATE",
    "time_created": "2025-04-15T13:36:25Z",
    "superevent_id": "S250415a",
    "urls": {
        "gracedb": "https://example.org/superevents/S250415a/view/"
    },
    "event": {
        "time": "2025-04-15T12:22:46.654Z",
        "far": 5.11069936486e-14,
        "significant": True,
        "instruments": ["H1", "L1", "V1"],
        "group": "CBC",
        "pipeline": "gstlal",
        "search": "AllSky",
        "properties": {
            "HasNS": 0.97,
            "HasRemnant": 0.93,
            "HasMassGap": 0.01
        },
        "classification": {
            "BNS": 0.97,
            "NSBH": 0.01,
            "BBH": 0.01,
            "Terrestrial": 0.01
        },
        "duration": None,
        "central_frequency": None,
        "skymap": "U0lNUExFICA9ICAgICAgICAgICAgICAgICAgICBUIC8gY29uZm..."
    },
    "external_coinc": None
}

# Example LVC Retraction Notice
EXAMPLE_RETRACTION_NOTICE = {
    "alert_type": "RETRACTION",
    "time_created": "2025-04-15T14:36:23Z",
    "superevent_id": "S250415a",
    "urls": {
        "gracedb": "https://example.org/superevents/S250415a/view/"
    },
    "event": None,
    "external_coinc": None
}

# Example GCN Circular
EXAMPLE_CIRCULAR = {
    "subject": "LIGO/Virgo S250415a: Radio Telescope Observation",
    "eventId": "LIGO/Virgo S250415a",
    "bibcode": "2025GCN.12345....1L",
    "createdOn": int(time.time() * 1000),
    "circularId": 12345,
    "submitter": "Radio Astronomer at University <astronomer@university.edu>",
    "email": "astronomer@university.edu",
    "body": """The Radio Telescope Collaboration reports:

We have conducted follow-up observations of the binary neutron star merger 
candidate S250415a (GCN Circular 12340) using our radio telescope array.

Observations began at 2025-04-15T13:30:00 UTC, approximately 1 hour after 
the GW trigger. We covered the 90% localization region with a series of 
pointed observations at 3 GHz.

No significant radio counterpart was detected in our initial scan to a 
flux density limit of 0.5 mJy. We will continue monitoring the region 
over the coming days and report any detection.

For further information, please contact: astronomer@university.edu
"""
}

# Example Potential Merger Event
def create_potential_merger_event(gps_time=None, utc_time=None):
    """Create a potential merger event"""
    if gps_time is None:
        # Current time in GPS seconds (approx)
        gps_time = time.time() + 315964800 - 18
    
    if utc_time is None:
        # Current UTC time
        utc_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        "EventType": "PotentialMerger",
        "detection_details": [
            {
                "GPS_time": gps_time,
                "UTC_time": utc_time
            }
        ]
    }


def send_kafka_message(bootstrap_servers: str, topic: str, message: Dict[str, Any]) -> None:
    """Send a message to a Kafka topic"""
    producer = Producer({
        'bootstrap.servers': bootstrap_servers,
    })
    
    producer.produce(
        topic,
        key="test",
        value=json.dumps(message).encode('utf-8')
    )
    producer.flush()
    print(f"Sent message to topic {topic}")


def simulate_gw_detection_sequence(bootstrap_servers: str, topic: str) -> None:
    """Simulate a GW detection sequence"""
    # 1. Send a potential merger event
    now = datetime.datetime.utcnow()
    gps_time = time.time() + 315964800 - 18  # Approximate GPS time
    utc_time = now.strftime("%Y-%m-%d %H:%M:%S")
    
    potential_merger = create_potential_merger_event(gps_time, utc_time)
    send_kafka_message(bootstrap_servers, topic, potential_merger)
    print(f"Sent potential merger event at {utc_time}")
    time.sleep(2)
    
    # 2. Update the superevent_id and event time in the initial notice
    initial_notice = EXAMPLE_INITIAL_NOTICE.copy()
    initial_notice["superevent_id"] = f"S{now.strftime('%y%m%d')}a"
    initial_notice["event"]["time"] = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    
    # Modify BNS probability randomly
    bns_prob = random.uniform(0.7, 0.99)
    initial_notice["event"]["classification"]["BNS"] = bns_prob
    initial_notice["event"]["classification"]["BBH"] = 1 - bns_prob - 0.01
    
    # Send as an LVC notice
    lvc_message = {
        "EventType": "LVC_NOTICE",
        "notice_type": "INITIAL",
        "payload": initial_notice
    }
    send_kafka_message(bootstrap_servers, topic, lvc_message)
    print(f"Sent initial LVC notice for {initial_notice['superevent_id']}")
    time.sleep(5)
    
    # 3. Send an update notice
    update_notice = EXAMPLE_UPDATE_NOTICE.copy()
    update_notice["superevent_id"] = initial_notice["superevent_id"]
    update_notice["event"]["time"] = initial_notice["event"]["time"]
    
    # Send as an LVC notice
    lvc_update_message = {
        "EventType": "LVC_NOTICE",
        "notice_type": "UPDATE",
        "payload": update_notice
    }
    send_kafka_message(bootstrap_servers, topic, lvc_update_message)
    print(f"Sent update LVC notice for {update_notice['superevent_id']}")
    time.sleep(5)
    
    # 4. Send a circular
    circular = EXAMPLE_CIRCULAR.copy()
    circular["eventId"] = f"LIGO/Virgo {initial_notice['superevent_id']}"
    circular["subject"] = f"LIGO/Virgo {initial_notice['superevent_id']}: Radio Telescope Observation"
    
    # Send as a GCN circular
    gcn_message = {
        "EventType": "GCN_CIRCULAR",
        "payload": circular
    }
    send_kafka_message(bootstrap_servers, topic, gcn_message)
    print(f"Sent GCN circular for {circular['eventId']}")
    time.sleep(5)
    
    # 5. Optionally send a retraction (uncomment to test)
    """
    retraction_notice = EXAMPLE_RETRACTION_NOTICE.copy()
    retraction_notice["superevent_id"] = initial_notice["superevent_id"]
    
    # Send as an LVC notice
    lvc_retraction_message = {
        "EventType": "LVC_NOTICE",
        "notice_type": "RETRACTION",
        "payload": retraction_notice
    }
    send_kafka_message(bootstrap_servers, topic, lvc_retraction_message)
    print(f"Sent retraction LVC notice for {retraction_notice['superevent_id']}")
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GW Monitoring System Utilities")
    parser.add_argument("--bootstrap-servers", default="localhost:9092", help="Kafka bootstrap servers")
    parser.add_argument("--topic", default="octopus_events", help="Kafka topic")
    parser.add_argument("--action", choices=["simulate"], default="simulate", help="Action to perform")
    
    args = parser.parse_args()
    
    if args.action == "simulate":
        simulate_gw_detection_sequence(args.bootstrap_servers, args.topic)