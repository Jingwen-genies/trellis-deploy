import logging

def setup_logging():
    """Setup logging configuration"""
    # Create logger
    logger = logging.getLogger('trellis-api')
    logger.setLevel(logging.DEBUG)  # Set the base log level
    
    # Avoid duplicate handlers if this function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create file handler which logs even debug messages
    file_handler = logging.FileHandler('trellis-api.log')
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
