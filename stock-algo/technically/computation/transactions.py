#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:17:22 2024

@author: cjymain
"""
import pandas as pd
from technically.const import GROUP_BY
from technically.core.cleaning import (
    adjust_for_splits,
    market_cap_category
)
from technically.utils.handlers.db import PostgreSQL
from technically.utils.log import get_logger
import numpy as np
import traceback

class DataAcquisitionController:
    """
    Controls metadata retrieval, Tiingo data acquisition, and writing data to DuckDB.
    """

    def metadata(self):
        """
        Retrieves metadata for all qualifying tickers.

        Returns:
            [prices_metadata, fundamentals_metadata]: Two tuples containing metadata for prices and fundamentals, respectively.
        """

        with PostgreSQL() as conn:
            prices_metadata_query = '''
                SELECT
                    table_alias as ticker,
                    exchange,
                    cap_category,
                    sector,
                    asset_type,
                    newest_db_prices,
                    last_processed,
                    prices_initialized
                FROM 
                    prices.metadata
                WHERE 
                    designation = 'active'
                    OR 
                        designation = 'delisted' 
                        OR designation = 'recent_dates_inactive'
                        AND prices_initialized = false
                        OR last_processed = '1990-01-01';
                '''
            prices_metadata = conn.run_query(
                prices_metadata_query,
                return_as="tuple"
            )

            fundamentals_metadata_query = '''
                SELECT
                    table_alias as ticker,
                    exchange,
                    cap_category,
                    sector,
                    newest_db_statements,
                    last_statement_api_update,
                    next_expected_statements_release,
                    last_processed,
                    fundamentals_initialized
                FROM 
                    prices.metadata
                WHERE 
                    designation = 'active'
                    AND asset_type = 'stock'
                    AND no_api_statements = false
                    OR 
                        designation != 'active'
                        AND last_statement_api_update IS NOT NULL
                        AND fundamentals_initialized = false;
                '''
            fundamentals_metadata = conn.run_query(
                fundamentals_metadata_query,
                return_as="tuple"
            )

        return [prices_metadata, fundamentals_metadata]

    def models(self):
        """
        Retrieves technical indicator success rates or creates the database table if it doesn't exist.

        Returns:
            backtest_success_rates: DataFrame of indicator success rates.
        """

        with PostgreSQL() as conn:
            if not conn.has_table("backtest_success_rates", "models"):
                get_logger().info(
                    "No indicator success rate table found. Creating empty table."
                )
                create_table_query = '''
                    CREATE TABLE models.backtest_success_rates (
                        date DATE, 
                        ticker VARCHAR, 
                        sector VARCHAR,
                        cap VARCHAR,
                        exchange VARCHAR,
                        indicator VARCHAR,
                        signal_name VARCHAR,
                        reversal_count INTEGER,
                        reversal_success INTEGER,
                        reversal_failure INTEGER,
                        continuation_count INTEGER,
                        continuation_success INTEGER,
                        continuation_failure INTEGER,
                        PRIMARY KEY (ticker, signal_name)
                    );
                    '''
                conn.run_query(
                    create_table_query,
                    commit=True
                )
            extract_success_rates_query = '''
                SELECT 
                    sector, 
                    signal_name, 
                    AVG(CAST(reversal_success AS double precision) / NULLIF(reversal_count, 0)) AS reversal_success_rate, 
                    AVG(CAST(continuation_success AS double precision)  / NULLIF(continuation_count, 0)) AS continuation_success_rate 
                FROM 
                    models.backtest_success_rates 
                GROUP BY
                    {group}, 
                    signal_name;
                '''
            backtest_success_rates = conn.run_query(
                extract_success_rates_query,
                params={"group": GROUP_BY},
                return_as="pandas"
            )

            if backtest_success_rates.empty:
                get_logger().warning(
                    "No indicator success rates found. Indicator scores will be defaults until backtesting is conducted."
                )
                return backtest_success_rates

            bearish_mask = backtest_success_rates["signal_name"].str.startswith("bearish")

            backtest_success_rates.loc[bearish_mask, ["reversal_success_rate", "continuation_success_rate"]] = \
                backtest_success_rates[["reversal_success_rate", "continuation_success_rate"]].apply(lambda x: x * -1)

            # Fills in nulls and infinite values with 0.0
            backtest_success_rates.fillna(0.0, inplace=True)
            backtest_success_rates.replace([np.inf, -np.inf], 0.0, inplace=True)

            return backtest_success_rates

    def prices(self, api_session, ticker: str, exchange: str, cap_category: str, sector: str, asset_type: str, newest_db_prices: str, initialized: bool):
        """
        Handles the ingestion of price data, formats it, and writes it to DuckDB.

        Args:
            api_session: The active HTTPS session used for making API calls.
            ticker (str): Ticker symbol.
            exchange: Ticker exchange.
            cap_category (str): Ticker market cap category.
            sector (str): Ticker sector.
            asset_type (str): Ticker asset type. Can equal "stock" or "etf".
            newest_db_prices (str): Latest data stored in database for ticker, formatted as YYYY-MM-DD.
            initialized (bool): True if database table exists, False if it needs to be created.

        Returns:
            Union[None, True]:
                - None: Any variety of errors/inconsistencies occurred.
                - True: Indicates the data transmission was successful.
        """
        with PostgreSQL() as conn:
            # Obtain the data from Tiingo
            try:
                df = api_session.daily_prices(
                    ticker, asset_type, newest_db_prices
                )
            except Exception:
                get_logger().error(
                    "Unexpected error retrieving price data.", extra={
                        "item_id": ticker,
                        "error": traceback.format_exc()
                    }
                )
                return

            if not isinstance(df, pd.DataFrame):
                if df == "ticker_not_found":
                    change_alias_to_peramticker = '''
                        UPDATE prices.metadata SET use_permaticker = True
                            WHERE ticker = :ticker
                        '''
                    conn.run_query(
                        change_alias_to_peramticker,
                        params={"ticker": ticker},
                        commit=True
                    )
                    get_logger().warning(
                        "Ticker not found. Will retry with permaticker next execution.", extra={
                            "item_id": ticker,
                            "api": "tiingo"
                        }
                    )
                else:
                    get_logger().warning(
                        "Price API call returned no data.", extra={
                            "item_id": ticker,
                            "api": "tiingo"
                        }
                    )
                return

            # Determines cap (if data available)
            if "market_cap" in df.columns:
                current_cap = df["market_cap"].iloc[-1]
                cap_category = market_cap_category(current_cap)

            # Casts volume column as integer
            df["volume"] = df["volume"].astype(int)

            if not initialized:  # Table doesn't exist
                # Fully adjust the newly acquired price data (if a split is present)
                if any(df["split_factor"] != 1.0):
                    df = adjust_for_splits(df.copy())

                try:
                    with conn.DataFrameToPostgreSQL(conn, ticker, "prices", df) as databridge:
                        databridge.create_table_from_df(primary_key=["date"])
                except Exception:
                    get_logger().error(
                        ("Error creating prices table. It may already exist. "
                         "Table will be deleted to allow for proper initialization."),
                        extra={
                            "item_id": ticker,
                            "error": traceback.format_exc()
                        }
                    )
                    drop_query = '''
                        DROP TABLE IF EXISTS prices.{table};
                        '''
                    conn.run_query(
                        drop_query,
                        params={"table": ticker},
                        commit=True
                    )
                    return
            else:  # Table exists
                for i, sf in enumerate(df["split_factor"]):
                    # Adjusts existing ticker's table for a new split (if it's present)
                    if sf != 1.0:
                        update_query = '''
                            UPDATE 
                                prices.{table}
                            SET 
                                open = (open / {sf}),
                                high = (high / {sf}),
                                low = (low / {sf}),
                                close = (close / {sf}),
                                volume = (volume * {sf});
                            '''
                        conn.run_query(
                            update_query,
                            params={"table": ticker, "sf": sf},
                            commit=True
                        )
                with conn.DataFrameToPostgreSQL(conn, ticker, "prices", df) as databridge:
                    status = databridge.insert_or_replace_df_into_table("date")

                    # Indicates that ticker does not have a prices table; metadata is reset
                    if status == "no_table":
                        get_logger().error(
                            ("Ticker is marked as initialized but does not have a price table. ",
                             "Resetting its metadata."),
                            extra={
                                "item_id": ticker,
                                "error": traceback.format_exc()
                            }
                        )
                        # Allows ticker to properly re-initialize next execution
                        metadata_reset_query = '''
                           UPDATE
                               prices.metadata
                           SET newest_db_prices = DEFAULT,
                               prices_initialized = DEFAULT
                           WHERE 
                               table_alias = :ticker;
                           '''
                        conn.run_query(
                            metadata_reset_query,
                            params={"ticker": ticker},
                            commit=True
                        )
                        return
                    elif status == "binder_error":
                        get_logger().warning(
                            ("Price data table likely does not yet have daily fundamentals columns. ",
                             "Attempting to add them now."),
                            extra={
                                "item_id": ticker,
                                "error": traceback.format_exc()
                            }
                        )
                        conn.add_missing_columns(
                            ticker,
                            list(df.columns)
                        )
                        return
                    elif status != True:  # Uncaught error
                        get_logger().error(
                            "Unknown error appending to price table.", extra={
                                "item_id": ticker,
                                "error": status,
                            }
                        )
                        return

            update_query = '''
                UPDATE 
                    prices.metadata 
                SET
                    cap_category = :cap,
                    newest_db_prices = :date,
                    prices_initialized = :init
                WHERE 
                    table_alias = :ticker;
                '''
            conn.run_query(
                update_query,
                params={
                    "ticker": ticker,
                    "cap": cap_category,
                    "date": df["date"].iloc[-1],
                    "init": initialized,
                },
                commit=True
            )
        return True

    def fundamentals(self, api_session, ticker: str, start_date: str, initialized: bool):
        """
        Handles the ingestion of fundamental statements data, formats it, and writes it to DuckDB.

        Args:
            api_session: The active HTTPS session used for making API calls.
            ticker (str): Ticker symbol.
            start_date (str): The earliest date to search for statements for ticker, formatted as YYYY-MM-DD.
            initialized (bool): Indicates if the database table should be initialized.

        Returns:
            Union[None, True]:
                - None: Any variety of errors/inconsistencies occurred.
                - True: Indicates the data transmission was successful.
        """
        try:
            statement_dfs_dict = api_session.fundamentals_statements(
                ticker, start_date
            )
        except Exception:
            get_logger().error(
                "Unexpected error retrieving fundamental statement data.", extra={
                    "item_id": ticker,
                    "error": traceback.format_exc(),
                }
            )
            return

        # Ensures that when the API does not have statements for a ticker, it is not re-checked every execution
        with PostgreSQL() as conn:
            if not isinstance(statement_dfs_dict, dict):
                update_query = '''
                    UPDATE 
                        prices.metadata
                    SET 
                        no_api_statements = true
                    WHERE 
                        table_alias = :ticker
                        AND fundamentals_initialized = false;
                    '''
                conn.run_query(
                    update_query,
                    params={"ticker": ticker},
                    commit=True
                )
                get_logger().info(
                    "Fundamental statements API call returned no data.", extra={
                        "item_id": ticker,
                        "api": "tiingo"
                    }
                )
                return

            # Writes each statement df to its own table
            for stmt_type, df in statement_dfs_dict.items():
                if not isinstance(df, pd.DataFrame):
                    continue

                df = df.sort_values("date")
                table_name = f"{ticker}_{stmt_type}"

                if not initialized:  # If not in system
                    with conn.DataFrameToPostgreSQL(conn, table_name, "fundamentals", df) as databridge:
                        success = databridge.create_table_from_df(
                            primary_key=["date"]
                        )
                        if not success:
                            get_logger().warning(
                                ("Fundamentals table already exists. ",
                                 "Tables will be deleted to allow for proper initialization."), extra={
                                    "item_id": ticker,
                                    "error": traceback.format_exc()
                                }
                            )
                            # Resets fundamentals metadata
                            update_query = '''
                                UPDATE 
                                    prices.metadata
                                SET
                                    newest_db_statements = DEFAULT,
                                    fundamentals_initialized = DEFAULT
                                WHERE
                                    table_alias = :ticker;
                                '''
                            conn.run_query(
                                update_query,
                                params={"ticker": ticker},
                                commit=True
                            )

                            drop_query = '''
                                DROP TABLE IF EXISTS fundamentals.{table_name};
                                '''
                            conn.run_query(
                                drop_query,
                                params={"table_name": table_name},
                                commit=True
                            )
                            continue
                else:  # In system
                    with conn.DataFrameToPostgreSQL(conn, table_name, "fundamentals", df) as databridge:
                        status = databridge.insert_or_replace_df_into_table("date")

                        # Triggered when a ticker does not have data for stmt_type
                        if status == "no_table":
                            get_logger().error(
                                ("Ticker is marked as initialized but does not have a statement table. ",
                                 "Resetting its metadata."),
                                extra={
                                    "item_id": f"{ticker}.{stmt_type}",
                                    "error": traceback.format_exc()
                                }
                            )
                            # Allows ticker to properly re-initialize next execution
                            update_query = '''
                                UPDATE 
                                    prices.metadata
                                SET 
                                    newest_db_statements = DEFAULT,
                                    fundamentals_initialized = DEFAULT
                                WHERE 
                                    table_alias = :ticker;
                                '''
                            conn.run_query(
                                update_query,
                                params={"ticker": ticker},
                                commit=True
                            )
                        elif status == "binder_error":
                            get_logger().warning(
                                "New fundamental statement data columns are mismatched to existing DB table. "
                                "Adding them.", extra={
                                    "item_id": ticker,
                                    "fundamental_statement": table_name
                                }
                            )
                            conn.add_missing_columns(
                                table_name,
                                list(df.columns)
                            )
                            continue
                        elif status != True:  # Uncaught error
                            get_logger().error(
                                "Unknown error appending to statement table.", extra={
                                    "item_id": ticker,
                                    "fundamental_statement": table_name,
                                    "error": status
                                }
                            )
                            continue

                update_query = '''
                    UPDATE 
                        prices.metadata
                    SET 
                        newest_db_statements = :date,
                        fundamentals_initialized = true
                    WHERE 
                        table_alias = :ticker;
                    '''
                conn.run_query(
                    update_query,
                    params={"date": df["date"].iloc[-1], "ticker": ticker},
                    commit=True
                )
        return True
