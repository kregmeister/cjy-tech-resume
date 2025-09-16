#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 09:38:13 2024

@author: cjymain
"""

import time
from technically.utils.log import get_logger


class LimitAPICalls:
    """
    Uses time intervals and call counts to limit API call frequency.
    """

    def __init__(self, limit: int, time_interval: int):
        self.call_count = 0
        self.limit = limit
        self.time_interval = time_interval
        self.start_time = time.monotonic()  # Starts timer

    def reset_timer(self):
        """
        Resets the timer and call count.

        Returns:
            None
        """
        now = time.monotonic()  # Compared with timer start time
        elapsed = now - self.start_time
        if elapsed >= self.time_interval:  # Resets call count and start_time if 60 seconds has passed
            self.call_count = 0
            self.start_time = now
        return

    def increment(self, i=1):
        """
        Increments the call count.

        Args:
            i (int, optional): Number of calls to increment.

        Returns:
            None
        """
        self.reset_timer()  # If needed
        if self.call_count < self.limit:
            self.call_count += i
        else:
            # Sets how long to wait for the timer and call_count to restart
            wait_for = self.time_interval - \
                (time.monotonic() - self.start_time)

            get_logger().info(f"API limit reached; waiting for {wait_for}")

            time.sleep(wait_for)
            self.reset_timer()
        return
