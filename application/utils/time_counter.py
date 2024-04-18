"""
@author: mading
@license: (C) Copyright: LUCULENT Corporation Limited.
@contact: mading@luculent.net
@file: time_counter.py
@time: 2023/11/4 10:43
@desc: 
"""
import time
from loguru import logger


def log_time_cost(func):

    def _warp(*args, **kwargs):
        start_time = time.time()
        func_name = kwargs.get("func_name")
        results = func(*args, **kwargs)
        logger.info(f"{func_name} cost time {time.time() - start_time}")
        return results

    return _warp
