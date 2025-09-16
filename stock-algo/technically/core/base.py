#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:49:55 2024

@author: cjymain
"""

import os
import sys

from technically.core import (
    initialization,
    metadata,
    coordination,
    cleaning
)
from technically.const import (
    TC_PATH,
    TC_LOG,
    CURRENT_DATE,
    INITIALIZATION_NOTICE
)
from technically.utils.log import get_logger
from technically.utils.date_checks import market_calendar_check


class Foundation:
    """
    Executes each core step of Technically.
    """

    def __init__(self):

        # Creates program data directory tree if it does not exist
        if not os.path.isdir(TC_PATH):
            initialization.directories_setup()
            get_logger().info(
                "Directories setup complete."
            )
            print(INITIALIZATION_NOTICE)
            get_logger().critical(INITIALIZATION_NOTICE)
            sys.exit(1)

    def execute(self):
        get_logger().info(
            "Executing Technically."
        )

        # Checks if NYSE traded today; program only executes on trading days
        trading_day = market_calendar_check(CURRENT_DATE)
        if not trading_day:
            get_logger().info(
                "New York Stock Exchange is not trading today. Not executing."
            )
            sys.exit(1)

        # Distinguishes dev envs from prod env
        if TC_PATH != "/home/craig99/.technically":
            metadata.ManageMetadata(dev=True).execute()
        else:
            metadata.ManageMetadata().execute()
        get_logger().info(
            "Ticker metadata acquired and updated."
        )

        coordination.DataAcquisitionCoordinator().execute()
        get_logger().info(
            "Ticker data acquisition, analysis, backtesting, and feature engineering completed."
        )

        cleaning.DatabaseChecks().execute()
        get_logger().info(
            "Data cleaning tasks completed."
        )

        #modeling.tc_spark(self.date, TC_PATH, "sector")
        #TC_LOG.get_logger().info("PySpark modeling tasks completed.")

        get_logger().info(
            "Execution completed."
        )

def execute_technically():
    executor = Foundation()
    executor.execute()
