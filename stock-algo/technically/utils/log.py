#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 11:41:42 2025

@author: cjymain
"""

import json
import time
from pythonjsonlogger import jsonlogger
import logging
import traceback
from functools import wraps
import os
import inspect


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        log_record['level'] = record.levelname
        log_record['message'] = record.msg

        # Formats newlines properly when tracebacks are logged
        if 'error' in log_record and log_record['error'] and isinstance(log_record['error'], str):
            log_record['error'] = log_record['error'].splitlines()

    def format(self, record):
        log_record = {}
        self.add_fields(log_record, record, {})
        return json.dumps(log_record, indent=4)

def log_setup(date, base_path, log_name="default"):
    """
    Configures logging.

    Args:
        date (date): Date of log.
        base_path (str): Base path for log location.
        log_name (str, optional): Defaults to the default log name for automated script execution.
            Set to a custom name in testing scenarios.

    Returns:
        logger
    """
    log_dir = base_path + "/logs/dev/"
    os.makedirs(log_dir, exist_ok=True)

    if log_name == "default":
        log_path = log_dir + f"tc_{date}.log"
    else:
        log_path = log_dir + log_name  # Custom log name

    logger = logging.getLogger("technically")
    logger.setLevel(logging.INFO)

    formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(filename)s %(funcName)s %(lineno)s %(message)s %(item_id)s %(duration_sec)s %(error)s'
    )

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def get_logger():
    logger = logging.getLogger("technically")
    return logger

def timer(_id: str = None, custom_fields: dict = {}, critical: bool = False):
    """
    Times and logs (to JSON) function calls.
    Does not work on classes directly.

    Args:
        item_id (str, optional): An identifier for the function/class.
            Defaults to param value of param name "ticker" if present.
            Other scenarios:
            1. A param name from the function that's wrapped. The value will be "_id".
            2. A custom string that will be the "_id".
            Otherwise, "_id" will be null.
        custom_fields (dict, optional): Additional custom fields to include in log submissions.
            Keys are field names, values are integers representing the index location of the desired arg.
        critical (bool, optional): If True, an uncaught error will terminate execution. Defaults to False.

    Returns:
        Decorator:
            Decorated function.

    Example:
        @timer(custom_fields={"url": 2})
        def func(self, item_id, url):

        Int "2" coincides with url being the third parameter in func.
        Args[2] will be used to derive "url" value.
    """

    def decorator(func):
        @wraps(func)
        def timer_wrapper(*args, **kwargs):
            if args and hasattr(args[0], "__class__"):
                class_instance = args[0].__class__
                exc_type = "Class"
                class_name = class_instance.__name__
                func_name = f"{class_name}.{func.__name__}"
                # Dict of class __init__ argument names: values
                class_args = vars(args[0])
                if "ticker" in class_args.keys():
                    class_id = class_args["ticker"]
            else:
                exc_type = "Function"
                func_name = func.__name__

            # Dict of func argument names: values
            func_param_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            func_args = dict(zip(func_param_names, args))
            func_args.update(kwargs)

            # Checks if _id is a param name for the func
            if _id in func_args.keys():
                item_id = func_args[_id]
            elif 'class_id' in locals():
                item_id = class_id
            else:
                item_id = _id
            frame = inspect.stack()[1]
            file_name = frame.filename.split("/")[-1]
            extra_fields = {k: args[v] for k, v in custom_fields.items()}

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception:
                get_logger().error(
                    "Uncaught exception. Please reference outer location and handle errors there.", extra={
                        "item_id": item_id,
                        "outer_location": func_name,
                        "outer_filename": file_name,
                        "error": traceback.format_exc()
                    } | extra_fields
                )
                if critical:  # Terminates execution
                    raise
                return

            duration = (time.perf_counter() - start)  # Seconds

            get_logger().info(
                f"{exc_type} operation completed.", extra={
                    "item_id": item_id,
                    "duration_sec": round(duration, 4),
                    "outer_location": func_name,
                    "outer_filename": file_name,
                } | extra_fields
            )
            return result
        return timer_wrapper
    return decorator

def analysis():
    None
