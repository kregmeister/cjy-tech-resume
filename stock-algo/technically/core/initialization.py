#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:34:44 2024

@author: cjymain
"""

import os
from technically.const import TC_PATH


def directories_setup():
    """
    Initializes the main directory structure.
    It may, in the future, create and configure a PostgreSQL database.

    Returns:
        None
    """
    os.mkdir(TC_PATH)
    # Creates main data subdirectories
    for subdir in ["parquet", "logs"]:
        os.mkdir(TC_PATH + f"/{subdir}")

    # Creates subdirectories for parquet
    for subdir in ["daily", "quarterly"]:
        os.mkdir(TC_PATH + f"/parquet/{subdir}")

    # Creates subdirectories for logs
    for subdir in ["main", "dev", "resources"]:
        os.mkdir(TC_PATH + f"/logs/{subdir}")

