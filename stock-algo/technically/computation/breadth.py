#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 17:19:08 2024

@author: cjymain
"""

import pandas as pd

from technically.utils.handlers.db import PostgreSQL
from technically.utils.log import timer


class BreadthExecutor:
    """
    Calculates database-wide breadth metrics.
    """

    @timer()
    def execute(self):
        with PostgreSQL() as self.conn:
            # Aggregate outcomes for tickers in exchange (advances, declines, etc.)
            tickers = self.extract()
            df = self.calculate(tickers)
            if df is not None:
                self.load(df)

        return

    def extract(self):
        """
        Gathers the list of tickers to be included in breadth metrics.

        Returns:
            list: list of tickers.
        """
        tickers_query = '''
            SELECT
                table_alias as ticker
            FROM
                prices.metadata
            WHERE
                designation != 'dead_ticker';
            '''
        tickers = self.conn.run_query(
            tickers_query,
            return_as="tuple"
        )
        return [ticker[0] for ticker in tickers]

    def calculate(self, tickers: list):
        """
        Calculates breadth metrics for each ticker and aggregates.

        Args:
            tickers (list): list of tickers.

        Returns:
            breadth_df (pd.DataFrame): DataFrame containing aggregated breadth metrics.
        """
        dates_query = '''
            SELECT 
                date
            FROM 
                prices.pnc
            ORDER BY 
                date DESC;
            '''
        dates = self.conn.run_query(
            dates_query,
            return_as="pandas"
        )

        breadth_df = pd.DataFrame(
            {
                "advances": 0,
                "declines": 0,
                "unchanged": 0,
                "adv_volume": 0,
                "decl_volume": 0,
                "unchanged_volume": 0,
                "new_highs": 0,
                "new_lows": 0,
            },
            index=dates["date"]
        )

        for ticker in tickers:
            # Gathers basic breadth metrics by-ticker
            breadth_query = '''
                SELECT
                    date,
                    CASE
                      WHEN close > LAG(close) OVER (ORDER BY date)
                      THEN 1
                      ELSE 0
                    END AS advances,
                    CASE
                      WHEN close < LAG(close) OVER (ORDER BY date) THEN 1
                      ELSE 0
                    END AS declines,
                    CASE
                      WHEN close = LAG(close) OVER (ORDER BY date) THEN 1
                      ELSE 0
                    END AS unchanged,
                    CASE
                      WHEN close > LAG(close) OVER (ORDER BY date) THEN volume
                      ELSE 0
                    END AS adv_volume,
                    CASE
                      WHEN close < LAG(close) OVER (ORDER BY date) THEN volume
                      ELSE 0
                    END AS decl_volume,
                    CASE
                      WHEN close = LAG(close) OVER (ORDER BY date) THEN volume
                      ELSE 0
                    END AS unchanged_volume,
                    CASE 
                      WHEN close = MIN(close) OVER (
                        ORDER BY date
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        )
                      THEN 1
                      ELSE 0
                    END AS new_lows,
                    CASE
                    WHEN close = MAX(close) OVER (
                        ORDER BY date
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        )
                      THEN 1
                      ELSE 0
                    END AS new_highs
                FROM
                    prices.{table}
                ORDER BY 
                    date DESC;
                '''
            df = self.conn.run_query(
                breadth_query,
                params={"table": ticker},
                return_as="pandas"
            )
            df.index = df["date"]
            df = df.drop(columns=["date"])

            # Adds ticker's breadth metrics from 'df' to cumulative 'breadth_df'
            breadth_df = breadth_df.combine(df, lambda a, b: a + b, fill_value=0)

        breadth_df = breadth_df.astype(int)

        # Calculates breadth indicators
        breadth_df["advance_decline_line"] = (breadth_df["advances"] - breadth_df["declines"]).cumsum()

        ad_ema19 = breadth_df["advance_decline_line"].ewm(span=19).mean()
        ad_ema39 = breadth_df["advance_decline_line"].ewm(span=39).mean()
        breadth_df["mcclellan_oscillator"] = ad_ema19 - ad_ema39

        high_low_idx = breadth_df["new_highs"] / (breadth_df["new_highs"] + breadth_df["new_lows"])
        breadth_df["new_high_low_index"] = high_low_idx.rolling(window=10).mean()

        breadth_df["trin"] = ((breadth_df["advances"] / breadth_df["declines"]) /
                              (breadth_df["adv_volume"] + breadth_df["decl_volume"]))

        breadth_df["breadth_obv"] = (breadth_df["adv_volume"] / breadth_df["decl_volume"]).cumsum()

        # Sets date index as its own column and restores default integer index
        breadth_df = breadth_df.reset_index()

        return breadth_df

    def load(self, df: pd.DataFrame):
        """
        Writes breadth metrics to DuckDB.

        Args:
            df (pd.DataFrame): Resultant DataFrame.

        Returns:
            None
        """
        with self.conn.DataFrameToPostgreSQL(self.conn, "breadth", "prices", df) as databridge:
            if not self.conn.has_table("breadth", "prices"):
                # Initialize
                databridge.create_table_from_df(primary_key=["date"])
            else:
                # Inserts/replaces data
                databridge.insert_or_replace_df_into_table("date")

