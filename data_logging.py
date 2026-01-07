# libraries
import logging


# custom preprocessing batch formatter
class CustomBatchFormatter(logging.Formatter):
    """
    Custom formatter to control the log message structure for INFO and WARNING logs.
    """

    # format for the INFO line (metadata)
    INFO_FMT = '***INFO***: PID: %(PID)s; AFCT_SIDE: %(AFCT_SIDE)s; %(FCS_SIDE)s; TID: %(TID)s; FPATH: %(FPATH)s'

    # format for the WARNING line (consecutive NaNs)
    WARNING_FMT = '***WARNING***: REP_NAN: %(REP_NAN)s'

    def format(self, record):
        if record.levelno == logging.INFO:
            self._style._fmt = self.INFO_FMT
        elif record.levelno == logging.WARNING:
            self._style._fmt = self.WARNING_FMT
        else:
            # Use default format for other levels (e.g., DEBUG, ERROR)
            self._style._fmt = logging.BASIC_FORMAT

        return super().format(record)


def setup_batch_logger(log_file_path: str = 'batch_processing.log'):
    """
    Initializes the logger with a file handler and the custom formatter.

    Args:
        log_file_path (str): Path to store the log file
    Returns:
        logger (logging.Logger): Logger object.
    """

    logger = logging.getLogger('BatchProcessor')

    # set the lowest log level to capture INFO and WARNING
    logger.setLevel(logging.INFO)

    # ensure handlers are not duplicated if called multiple times
    if not logger.handlers:
        # create file handler
        file_handler = logging.FileHandler(log_file_path, mode='w') # 'w' overwrites, use 'a' to append

        # set the custom formatter
        formatter = CustomBatchFormatter()
        file_handler.setFormatter(formatter)

        # add handler to the logger
        logger.addHandler(file_handler)

    return logger
