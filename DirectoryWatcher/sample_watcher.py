'''
A simple Python script that monitors the directory_to_watch/ directory and logs/prints a message whenever a new image file is added. This script serves as the foundation for more complex implementations
'''

import time
import os
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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

