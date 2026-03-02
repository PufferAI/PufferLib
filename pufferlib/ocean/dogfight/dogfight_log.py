"""File-based logging for dogfight league/training diagnostics.

All diagnostic output goes to league/logs/ via Python's logging module.
Log lines follow: HH:MM:SS [TAG] key=value structured format for grep.

Usage:
    from pufferlib.ocean.dogfight.dogfight_log import init_log, log

    init_log('league/logs', 'league_round')   # creates FileHandler
    log('[ROUND] num=37 event=start')          # writes to log file
"""
import logging
import os
from datetime import datetime

logger = logging.getLogger('dogfight')
logger.setLevel(logging.DEBUG)


def init_log(log_dir='league/logs', run_name=None):
    """Add a FileHandler to the dogfight logger."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    filename = f'{run_name}_{ts}.log' if run_name else f'dogfight_{ts}.log'
    path = os.path.join(log_dir, filename)
    fh = logging.FileHandler(path)
    fh.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)
    logger.info(f'[ROUND] log_started path={path}')
    return path


log = logger.info  # convenience alias
