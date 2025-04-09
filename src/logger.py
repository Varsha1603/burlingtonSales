import logging
import os   
import sys
from datetime import datetime

# Create a directory for logs if it doesn't exist
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)
# Configure logging 
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(levelname)s %(message)s',
    level=logging.INFO,
    
)
if __name__ == "__main__":
    logging.info("Logging has been configured.")
    logging.info(f"Log file path: {LOG_FILE_PATH}")
    logging.info("This is an info message.")
    logging.error("This is an error message.")
    logging.debug("This is a debug message.")
    logging.warning("This is a warning message.")
    logging.critical("This is a critical message.")