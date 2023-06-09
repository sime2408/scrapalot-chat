import logging
import os
import sys

from scripts.user_environment import args


def initialize_logging():
    # Setup logging if enabled
    log_folder = "logs"  # Specify the folder where you want to store the log files
    os.makedirs(log_folder, exist_ok=True)  # Create the log folder if it doesn't exist
    print(os.getcwd())
    log_file = os.path.join(log_folder, f'{os.path.basename(__file__)}.log')  # Path to the log file
    file_handler = logging.FileHandler(filename=log_file)
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', handlers=handlers, level=args.log_level, force=True)
