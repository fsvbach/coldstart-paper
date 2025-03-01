import logging

# Configure the root logger
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)  # Default level is WARNING

# Create a console handler
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)