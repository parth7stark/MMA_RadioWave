from datetime import datetime, timedelta

def parse_utc_time(time_str):
    """
    Parse a UTC time string to a datetime object for comparison
    
    Args:
        time_str: UTC time string
        Time of event (UTC, ISO-8601), e.g. 2018-11-01T22:22:46.654Z
        
    Returns:
        datetime: Parsed datetime
    """
    # Try different formats
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%fZ",  # <-- Added to handle the Z
        "%Y-%m-%dT%H:%M:%SZ"      # Also allow seconds-only Z
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue
    
    # If all formats fail, raise exception
    raise ValueError(f"Could not parse time string: {time_str}")
    
def gps_to_utc(gps_time):
    """
    Convert GPS time to UTC time.
    
    Args:
        gps_time: GPS time
        
    Returns:
        datetime: UTC time
    """
    # GPS time starts from January 6, 1980, and doesn't include leap seconds
    # This is a simplified conversion for demonstration
    gps_epoch = datetime(1980, 1, 6)
    delta_seconds = float(gps_time)
    
    return gps_epoch + timedelta(seconds=delta_seconds)