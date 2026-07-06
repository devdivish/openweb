import logging
import logging.handlers
import os


def create_logger_object(filename: str):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [ %(filename)s:%(lineno)s | %(funcName)s() ] %(message)s'
    )
    os.makedirs('./logs', exist_ok=True)
    log_file_name = os.path.join('./logs', filename)
    handler = logging.handlers.RotatingFileHandler(log_file_name, maxBytes=20 * 1024 * 1024, backupCount=8)
    handler.setFormatter(formatter)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    return logger


logger = create_logger_object('ocr_detection_worker.logs')
