import functools
import inspect
import logging
import os
import sys
import time
from typing import Optional

import torch

def save_bad_data(data, step, save_dir="debug-bad-data"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"bad_data_step_{step}.pt")
    torch.save(data, save_path)
    print(f"Saved bad data to {save_path}")

def log_allocated_gpu_memory(log=None, stage="loading model", device=0):
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated(device)
        msg = f"Allocated GPU memory after {stage}: {allocated_memory/1024/1024/1024:.2f} GB"
        print(msg) if log is None else log.info(msg)

def log_execution_time(logger=None):
    """Decorator to log the execution time of a function"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if logger is None:
                print(f"{func.__name__} took {elapsed_time:.2f} seconds to execute.")
            else:
                logger.info(
                    f"{func.__name__} took {elapsed_time:.2f} seconds to execute."
                )
            return result

        return wrapper

    return decorator

def blockprint():
    """Block printing to stdout and stderr for the current process."""
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

def setup_logger(main_rank: bool,
                 filename: str|None = None,
                 debug: bool = False,
                 name: Optional[str] = None) -> logging.Logger:
    '''
    Set up a logger for the script.
    main_rank: bool, whether this is the main process. We only set up logging for the main process.
    filename: str, the name of the file to log to. If None, logs to stdout.
    debug: bool, whether to log in debug mode.
    name: str, the name of the logger. If None, uses the name of the calling module.
    '''

    if name is None:
        # Use the name of the calling module as the logger name
        logger = logging.getLogger(name=
            os.path.splitext(os.path.basename(inspect.stack()[1].filename))[0]
        )
    else:
        logger = logging.getLogger(name=name)

    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Only set up logging for rank 0
    if main_rank:
        if debug:
            logger.setLevel(logging.DEBUG) # Everything at DEBUG level and above will be logged
        else:
            logger.setLevel(logging.INFO) # Everything at INFO level and above will be logged

        # Create a file handler if filename is provided, otherwise use stdout
        if filename:
            handler = logging.FileHandler(filename)
        else:
            handler = logging.StreamHandler(sys.stdout)

        format_str = "[%(asctime)s,%(msecs)03d][%(name)s:%(lineno)d][%(levelname)s] - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        # Disable logging for non-master processes.
        logger.setLevel(logging.ERROR)
        # Optionally add a NullHandler to absorb logs.
        logger.addHandler(logging.NullHandler())

    logger.propagate = False # so no duplicate, unformatted log in stderr
    return logger

class Timer:
    def __init__(self):
        self._start = time.time()

    def __call__(self, reset=True):
        now = time.time()
        diff = now - self._start
        if reset:
            self._start = now
        return diff
