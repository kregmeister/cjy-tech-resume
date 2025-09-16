#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:33:42 2024

@author: cjymain
"""
import yfinance as yf
from yfinance.exceptions import YFDataException, YFRateLimitError
from curl_cffi.requests.exceptions import HTTPError as curlHTTPError
from requests.exceptions import HTTPError as requestsHTTPError
import traceback

from technically.const import CURRENT_DATE
from technically.utils.time_limiter import LimitAPICalls
from technically.utils.log import get_logger, timer


class YFinanceAPI:
    """
    Initializes requests.session object for Tiingo API calls.
    """

    def __init__(self):
        self.session = None
        self.headers = None
        # Limits yfinance calls to 2 per second (~7,200 per hour)
        self.call_limiter = LimitAPICalls(2, 1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            get_logger().error(
                "Exception in class YFinanceAPI context.", extra={
                    "error": traceback.format_exc()
                }
            )
            return True

    @timer()
    def fund_profile(self, ticker: str):
        """
        Retrieves ETF categorization data for a given ticker.

        Args:
            ticker (str): Security to get data for.

        Returns:
            list: Three strings. All equal 'unknown' if API did not return data.
        """
        try:
            data = yf.Ticker(ticker)
            overview = data.funds_data.fund_overview

            net_assets = data.info['net_assets']
            fund_family = overview['family']
            fund_category = overview['categoryName']
        except (YFDataException, requestsHTTPError, curlHTTPError, KeyError, YFRateLimitError):
            get_logger().warning(
                "No fund overview found from YFinance API.", extra={
                    "item_id": ticker,
                    "error": traceback.format_exc()
                }
            )
            # fund_category is a partition column in parquet, cannot be None
            return ["unknown", "unknown", "unknown"]

        # Makes 2 api calls
        self.call_limiter.increment(i=2)

        return [net_assets, fund_family, fund_category]

    ### NOT CURRENTLY IN USE ###
    def expected_earnings_call(self, ticker: str):
        """
        Retrieves next expected earnings date for a given ticker.

        Args:
            ticker: Security to get data for.

        Returns:
            next_call (datetime.date): Next expected earnings date. Equals None if no data returned.
        """
        try:
            data = yf.Ticker(ticker)
            next_call = data.calendar['Earnings Date'][0]
            # Ensure returned value is a future date
            if next_call < CURRENT_DATE:
                return None
            else:
                return next_call
        except (YFDataException, requestsHTTPError, curlHTTPError, KeyError, YFRateLimitError):
            get_logger().warning(
                "No fund overview found from YFinance API.", extra={
                    "item_id": ticker,
                    "error": traceback.format_exc()
                }
            )
            return None