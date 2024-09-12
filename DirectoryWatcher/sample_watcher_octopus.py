'''
An updated version of the original script that adds KafkaProducer functionality. The script will send a message to a Kafka topic every time an image is created or modified in the directory.
'''

import time
import os
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from diaspora_event_sdk import KafkaProducer

# Initialize Kafka Producer
producer = KafkaProducer()

# Set up logging configuration
logging.basicConfig(
    filename='file_watcher.log',   # Log output to file
    level=logging.INFO,            # Log level: INFO (suitable for production)
    format='%(asctime)s - %(message)s',  # Log format with timestamps
    datefmt='%Y-%m-%d %H:%M:%S'    # Date format for logs
)

# Custom event handler class for handling file system events
class ImageHandler(FileSystemEventHandler):
    """
    Handles file system events for image files.
    """

    def on_created(self, event):
        """
        Triggered when a new file or directory is created.
        """
        # Supported image file extensions
        IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')

        # Check if the new file is an image (by extension)
        if not event.is_directory and event.src_path.lower().endswith(IMAGE_EXTENSIONS):
            file_name = os.path.basename(event.src_path)
            logging.info(f'New image added: {file_name}')  # Log the event
            print(f'New image added: {file_name}')         # Also print to the console
            
            # Publish event to Octopus
            self.publish_to_kafka(event.src_path)
            # self.publish_to_kafka(file_name)  #can just send filename instead of sending full path


    def publish_to_kafka(self, file_path):
        """Publish the file event to a Kafka topic."""
        message = {
            'message': 'new image added in NRAO machine 1',
            'file_name': file_path,
            'timestamp': time.time()
        }
        producer.send('mma-radiowave-DirectoryWatcher', message)
        logging.info(f'Sent event to Octopus: {message}')
        print(f'Sent event to Octopus: {message}')

# Main function to monitor the directory
def monitor_directory(directory_to_watch):

    # Initialize the observer to watch the directory
    observer = Observer()
    event_handler = ImageHandler()

    # Schedule the observer with the event handler to monitor the directory
    observer.schedule(event_handler, directory_to_watch, recursive=True)

    # Start the observer in the background
    observer.start()
    logging.info(f'Started monitoring directory: {directory_to_watch}')
    print(f'Started monitoring directory: {directory_to_watch}')

    try:
        # Keep the script running and observer alive indefinitely
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Gracefully stop the observer when interrupted (Handle Ctrl+C)
        observer.stop()
        logging.info('Stopped monitoring directory.')
        print('Stopped monitoring directory.')
    finally:
        # Ensure observer thread finishes
        observer.join()



if __name__ == "__main__":
    # Directory to monitor for new images
    directory = './directory_to_watch'  # Replace with the actual directory path to watch

    # Ensure the directory exists before starting
    if not os.path.exists(directory):
        logging.error(f'Directory not found: {directory}')
        print(f'Error: Directory not found: {directory}')
    else:
        monitor_directory(directory)
