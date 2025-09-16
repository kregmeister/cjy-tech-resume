#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:45:35 2023

@author: cjymain
"""

from datetime import timedelta, date

from technically.api.tiingo import TiingoAPI
from technically.computation.transactions import DataAcquisitionController
from technically.computation.calculations import TechnicalsExecutor, FundamentalsExecutor
from technically.computation.breadth import BreadthExecutor

from technically.const import CURRENT_DATE
from technically.utils.log import timer
from technically.utils.handlers.calc_queue import CalculationsOrchestrator


class DataAcquisitionCoordinator:
    """
    Coordinates data acquisition and analysis for each ticker.
    """

    def __init__(self, regen=False):
        # When true, new data is not acquired and parquets are regenerated
        self.regen = regen
        # Pass api session and database connections to child classes
        self.handler = DataAcquisitionController()

        # Retrieves ticker metadata from databases
        self.prices_metadata, self.fundamentals_metadata = self.handler.metadata()

        # Retrieves backtesting metrics to apply in technical analysis
        self.backtest_success_rates = self.handler.models()

        # Initializes queue for technical calculations
        self.price_queue = CalculationsOrchestrator(
            TechnicalsExecutor,
            self.backtest_success_rates
        )

        # Queue for fundamentals calculations
        self.fund_queue = CalculationsOrchestrator(
            FundamentalsExecutor,
            None
        )

    def execute(self):
        # Retrieves ticker price data from Tiingo
        self.get_prices()
        self.price_queue.stop_workers_signal()

        # Retrieves ticker fundamental statements from Tiingo
        self.get_statements()
        self.fund_queue.stop_workers_signal()

        # Calculates breadth metrics for all tickers
        BreadthExecutor().execute()

    @timer(critical=True)
    def get_prices(self):
        """
        Parent-level coordinator for ticker-by-ticker price data acquisition,
        cleaning, technical analysis, indicator scoring, and feature engineering.

        Returns:
            None
        """
        with TiingoAPI() as api:
            # Loops through every ticker's metadata
            for ticker, exchange, cap_category, sector, asset_type, newest_db_prices, last_processed, initialized in self.prices_metadata:
                # Prevents retrieval of existing dates from Tiingo
                if initialized:
                    newest_db_prices = newest_db_prices + timedelta(days=1)

                if not self.regen:  # Skips data acquisition
                    # Acquires data, cleans it, and writes it to database
                    resp = self.handler.prices(
                        api,
                        ticker,
                        exchange,
                        cap_category,
                        sector,
                        asset_type,
                        newest_db_prices,
                        initialized
                    )
                    # Ensures all tickers get processed at least once
                    if resp is None and last_processed != '1990-01-01':
                        continue  # No technical analysis

                # Formats required inputs for technical calculations queue
                q_lst = [ticker, exchange, cap_category, sector, last_processed, initialized]
                self.price_queue.add_ticker(q_lst)  # Places in queue
        return

    @timer(critical=True)
    def get_statements(self):
        """
        Parent-level coordinator for ticker-by-ticker statements data acquisition,
        cleaning, fundamental analysis, and feature engineering.

        Returns:
            None
        """
        with TiingoAPI() as api:
            # Loops through every ticker's metadata
            for ticker, exchange, cap_category, sector, newest_db_statements, last_statement_api_update, next_expected_statements_release, last_processed, initialized in self.fundamentals_metadata:
                # Ensures that new statements are only checked for when:
                # 1. It's expected (>3 months after last statements release)
                # 2. API endpoint has been updated since last statements acquisition
                if (next_expected_statements_release <= CURRENT_DATE
                        and last_statement_api_update > newest_db_statements
                        or last_processed == '1990-01-01'):
                    newest_db_statements = newest_db_statements + timedelta(days=1)
                else:
                    continue

                if not self.regen:  # Skips data acquisition
                    # Acquires data, cleans it, and writes it to database
                    resp = self.handler.fundamentals(
                        api,
                        ticker,
                        newest_db_statements,
                        initialized
                    )
                    # Ensures all tickers get processed at least once
                    if resp is None and last_processed != '1990-01-01':
                        continue  # No fundamental analysis

                # Formats required inputs for fundamental calculations queue
                q_lst = [ticker, exchange, cap_category, sector, newest_db_statements, initialized]
                self.fund_queue.add_ticker(q_lst)  # Places in queue
        return
