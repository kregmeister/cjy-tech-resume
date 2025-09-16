#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 19:45:46 2024

@author: cjymain
"""

from datetime import timedelta
import pandas_market_calendars as mcal


def market_calendar_check(today):
    """
    Checks if the current date is a trading day or not.

    Args:
        today: Today's date in YYYY-MM-DD (datetime.date).

    Returns:
        True if trading day, False if not.
    """
    nyse = mcal.get_calendar("NYSE").schedule(
        start_date=str(today), end_date=str(today + timedelta(days=7))
    ).reset_index()

    trading_dates = [d8.date() for d8 in nyse['index']]

    today_date = trading_dates[0]
    if today_date == today:
        return True
    else:
        return False

def which_friday(today):
    """
    Checks which Friday of the month it is.

    Args:
        today: Today's date in YYYY-MM-DD (datetime.date).

    Returns:
        friday (int): nTh Friday of the month it is.
    """
    if today.weekday() != 4:
        return False
    else:
        friday = 0
        day_of_month = today.replace(day=1)
        while day_of_month <= today:
            if day_of_month.weekday() == 4:
                friday += 1
            day_of_month += timedelta(days=1)
        return friday

