#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 18:02:18 2023

@author: craigyingling321
"""

import pandas as pd

from technically.const import IS_FRIDAY
from technically.utils.handlers.db import PostgreSQL as PSQL
from technically.api.tiingo import TiingoAPI
from technically.utils.log import timer
from technically.api.yahoofinance import YFinanceAPI
from technically.core.cleaning import market_cap_category


class ManageMetadata:
    """
    Acquires metadata for all tickers and updates/initializes metadata table.
    """

    def __init__(self, dev=False):
        self.is_dev = dev

    @timer(critical=True)
    def execute(self):
        with PSQL() as self.conn:
            tickers_df = self.get_metadata()

            if self.conn.has_table("metadata", "prices"):
                self.update_metadata(tickers_df)
            else:
                self.initialize_metadata(tickers_df)

    def get_metadata(self):
        """
        Acquires and ingests ticker metadata, updating it when/where necessary.

        Returns:
            df (pd.DataFrame): Full, formatted metadata from Tiingo.
        """

        # Retrieves dataframe of ticker metadata
        with TiingoAPI() as api:
            df = api.daily_metadata()
            profile_df = api.fundamentals_meta()

        included_exchanges = ["PINK", "OTCGREY", "EXPM", "OTC", "NYSE", "NASDAQ", "ARCA",
                              "OTCBB", "OTCQB", "OTCCE", "OTCMKTS", "OTCQX", "NYSE ARCA", "NYSE MKT", "BATS"]

        # Development tests only include a small subset of tickers
        if self.is_dev:
            df = df[
                (df["exchange"].isin(included_exchanges)) &
                (df["asset_type"].isin(["Stock", "ETF"])) &
                (df["start_date"].notnull()) &
                (df["start_date"] != df["end_date"]) &
                (df["ticker"].isin(
                    [
                        "ally", "bk", "cfg", "jpm", "mtb", "pnc", "td", "tfc",
                        "wfc", "bac", "c", "gs", "ms", "usb", "cof", "schw",
                        "stt", "bmo", "axp", "hsbc", "fitb", "bcs", "key", "amp",
                        "nee", "gev", "fslr", "nxt", "cwen", "bepc", "ora", "be",
                        "run", "plug", "flnc", "amps", "arry", "rex", "shls", "amrc",
                        "mntk", "gevo", "gpre", "ff", "vgas", "spwr", "cslr", "nrgv", "actv",
                        "usbps", "cntm", "ehsi", "coni"  # Problematic in prod
                    ]
                ))
            ]
        else:
            # Cleans Tiingo metadata dataframe
            df = df[
                (df["exchange"].isin(included_exchanges)) &
                (df["asset_type"].isin(["Stock", "ETF"])) &
                (df["start_date"].notnull()) &
                (df["start_date"] != df["end_date"]) &
                (~df["ticker"].str.contains(r'[^a-z0-9_]', regex=True))
            ]

        # Ensures most recently traded ticker is listed ahead of duplicates
        df.sort_values(["ticker", "start_date"], ascending=[True, False], inplace=True)
        df["use_permaticker"] = df.duplicated("ticker")

        # Map for renaming exchanges
        consolidations = {
            "OTC": ["OTCBB", "OTCQB", "OTCCE", "OTCMKTS", "OTCQX"],
            "ARCA": ["NYSE ARCA", "NYSE MKT", "BATS"],
            "OTCPINK": ["PINK"],
            "OTCEXPM": ["EXPM"]
        }

        def map_to_key(value, dict_map):
            """
            Maps values of dict_map to corresponding keys of dict_map.
            Args:
                value: A value to be mapped.
                dict_map: Keys are the new values. Values are the old values to change.

            Returns:
                value: The new value.
            """
            for key, values in dict_map.items():
                if value in values:
                    return key
            return value

        # Consolidates exchanges and lowercases exchange & asset_type columns
        df["exchange"] = df["exchange"].apply(
            map_to_key, args=(consolidations,)
        ).str.lower()
        df["asset_type"] = df["asset_type"].str.lower()

        # Sets date columns to date format (comes from Tiingo as string)
        df["start_date"] = pd.to_datetime(df["start_date"], format="ISO8601").dt.date
        df["end_date"] = pd.to_datetime(df["end_date"], format="ISO8601").dt.date

        # Calculates tenure using start_date and end_date columns
        def difference_in_years(start, end):
            return [e.year - s.year for e, s in zip(end, start)]

        df["years_listed"] = difference_in_years(df["start_date"], df["end_date"])

        # Maps designation to active/dead
        df["designation"] = df["is_active"].apply(lambda x: "active" if x == True else "delisted")

        df = pd.merge(df, profile_df, how="left", on=["permaticker", "ticker"])
        df.rename(
            columns={
                "statement_last_updated": "last_statement_api_update",
                "daily_last_updated": "last_price_api_update"
            },
            inplace=True
        )
        return df

    def update_metadata(self, df):
        """
        Updates metadata table only where necessary using the existing table and the new data from df.

        Args:
            df: Full, formatted metadata from Tiingo.

        Returns:
            None
        """

        if type(IS_FRIDAY) == int:
            # Limits fundamental statements API calls to once per week when the endpoint does not return data
            metadata_reset1 = '''
                UPDATE
                    prices.metadata 
                SET 
                    no_api_statements = DEFAULT;
                '''
            self.conn.run_query(
                metadata_reset1,
                commit=True
            )
            # Allows tickers that failed price checks to try again (once per week) with some new data
            metadata_reset2 = '''
                UPDATE
                    prices.metadata
                SET 
                    designation = 'active'
                WHERE
                    designation = 'not_enough_data'
                    OR 
                        designation = 'dead_ticker'
                        AND is_active = true;
                '''
            self.conn.run_query(
                metadata_reset2,
                commit=True
            )
        with self.conn.DataFrameToPostgreSQL(self.conn, "metadata", "prices", df) as databridge:
            # Updates attributes that could be subject to change for existing tickers
            metadata_update = '''
                UPDATE
                    prices.metadata 
                SET
                    permaticker = df.permaticker,
                    ticker = df.ticker,
                    sector = df.sector,
                    industry = df.industry,
                    sic_sector = df.sic_sector,
                    sic_industry = df.sic_industry,
                    end_date = df.end_date,
                    is_active = df.is_active,
                    years_listed = df.years_listed,
                    company_website = df.company_website,
                    last_statement_api_update = df.last_statement_api_update,
                    last_price_api_update = df.last_price_api_update
                FROM
                    df
                WHERE
                    prices.metadata.permaticker = df.permaticker;
                '''
            databridge.use_df_as_table(
                metadata_update
            )

            apply_delisted_designation = '''
                UPDATE
                    prices.metadata
                SET
                    designation = df.designation
                FROM
                    df
                WHERE
                    prices.metadata.permaticker = df.permaticker
                    AND prices.metadata.designation = 'active'; 
                '''
            databridge.use_df_as_table(
                apply_delisted_designation
            )

            # Adds rows for new tickers
            columns = ", ".join(df.columns)
            insert_query = f'''
                INSERT INTO prices.metadata ({columns}) 
                    SELECT 
                        * 
                    FROM 
                        df
                    WHERE
                        df.permaticker NOT IN 
                        (SELECT prices.metadata.permaticker FROM prices.metadata);
                '''
            databridge.use_df_as_table(
                insert_query,
                params={"columns": columns}
            )

            # Deletes rows for tickers that are no longer listed in Tiingo
            delete_query = '''
                DELETE FROM prices.metadata
                    WHERE
                        prices.metadata.permaticker NOT IN
                        (SELECT df.permaticker FROM df);
                '''
            databridge.use_df_as_table(
                delete_query
            )

        self.update_etf_metadata()
        return

    def update_etf_metadata(self):
        """
        Attempts to gather details for ETF's from Yahoo Finance if it has not yet been acquired.

        Returns:
            None
        """

        # Gathers all ETFs without category data
        etf_lst_query = '''
            SELECT 
                table_alias
            FROM 
                prices.metadata 
            WHERE 
                asset_type = 'etf' 
                AND 
                    is_active = true 
                    AND etf_info_initialized = false;
            '''
        etf_lst = self.conn.run_query(
            etf_lst_query,
            return_as="tuple"
        )

        # Opens Yahoo Finance API session
        with YFinanceAPI() as api:
            for (ticker,) in etf_lst:
                # Attempts to gather category data
                net_assets, family, category = api.fund_profile(ticker)
                if type(net_assets) == float:
                    cap_category = market_cap_category(net_assets)
                else:
                    cap_category = "unknown"

                # Writes category data to metadata table
                etf_update = '''
                    UPDATE
                        prices.metadata
                    SET
                        etf_info_initialized = True,
                        cap_category = :cap_category,
                        sector = :sector,
                        industry = :industry,
                        sic_sector = :sic_sector,
                        sic_industry = :sic_industry
                    WHERE
                        table_alias = :ticker;
                    '''
                self.conn.run_query(
                    etf_update,
                    params={
                        "ticker": ticker,
                        "cap_category": cap_category,
                        "sector": category,
                        "industry": family,
                        "sic_sector": category,
                        "sic_industry": family
                    },
                    commit=True
                )
        return

    def initialize_metadata(self, df):
        """
        Process for creating metadata table if it does not yet exist.

        Args:
            df: Full, formatted metadata from Tiingo.

        Returns:
            None
        """

        # Creates metadata table
        create_metadata = '''
            CREATE TABLE prices.metadata(
                permaticker VARCHAR, 
                ticker VARCHAR,
                use_permaticker BOOLEAN,
                \"name\" VARCHAR, 
                exchange VARCHAR NOT NULL, 
                asset_type VARCHAR NOT NULL,
                cap_category VARCHAR DEFAULT('unknown'),
                sector VARCHAR DEFAULT('unknown'),
                industry VARCHAR DEFAULT('unknown'),
                sic_sector VARCHAR DEFAULT('unknown'), 
                sic_industry VARCHAR DEFAULT('unknown'),
                company_website VARCHAR,
                is_active BOOLEAN,
                start_date DATE, 
                end_date DATE,  
                years_listed BIGINT,
                designation VARCHAR,
                newest_db_prices DATE DEFAULT('1990-01-01'), 
                newest_db_statements DATE DEFAULT('1990-01-01'),
                last_statement_api_update DATE, 
                last_price_api_update DATE,   
                prices_initialized BOOLEAN DEFAULT(CAST('f' AS BOOLEAN)),  
                fundamentals_initialized BOOLEAN DEFAULT(CAST('f' AS BOOLEAN)),
                etf_info_initialized BOOLEAN DEFAULT(CAST('f' AS BOOLEAN)),
                recent_backtest BOOLEAN DEFAULT(CAST('f' AS BOOLEAN)),
                last_processed DATE DEFAULT('1990-01-01'),
                PRIMARY KEY(permaticker)
            );
            '''
        self.conn.run_query(
            create_metadata,
            commit=True
        )

        columns = ", ".join(df.columns)
        with self.conn.DataFrameToPostgreSQL(self.conn, "metadata", "prices", df) as databridge:
            # Populates metadata table
            insert_metadata = '''
                INSERT INTO prices.metadata (:columns)
                    SELECT * FROM df;
                '''
            databridge.use_df_as_table(
                insert_metadata,
                params={"columns": columns},
            )

            # Corrects behavior where last_statement_api_update is re-cast as a VARCHAR, not DATE on update/insert
            alter_metadata = '''
                ALTER TABLE prices.metadata
                    ALTER last_statement_api_update TYPE DATE;
                '''
            databridge.use_df_as_table(
                alter_metadata
            )

            self.update_etf_metadata()
            return
