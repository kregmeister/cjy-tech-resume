#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:26:06 2024

@author: cjymain
"""
import traceback
from alpaca.trading.client import TradingClient

from technically.utils.handlers.auth import get_credentials
from technically.utils.log import get_logger

ALPACA_API_KEY, ALPACA_API_SECRET = get_credentials(["alpaca_api_key", "alpaca_secret_token"])


class AlpacaTradeAPI:
    """
    Manages a TradingClient session to place orders, track P/L, etc.
    """

    def __enter__(self):
        self.client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            get_logger().error(
                "Exception in class AlpacaTradeAPI context.", extra={
                    "error": traceback.format_exc()
                }
            )
            return True

    def get_portfolio(self):
        """
        Extracts detailed portfolio data from Alpaca API.

        Returns:
            pd.DataFrame
        """
        positions = self.client.get_all_positions()
        return positions
