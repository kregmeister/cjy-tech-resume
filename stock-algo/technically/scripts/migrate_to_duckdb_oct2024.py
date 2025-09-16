#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:07:21 2024

@author: cjymain
"""

from technically.utils.logger_init import log_setup
from technically.utils.sql import multi_db_connection

import pandas as pd


class SQLiteToDuckDB:
    def __init__(self, sqlite_path, duckdb_path, log):
        """
        Parameters
        ----------
        sqlite_path : Full file path to existing SQLite database location.
        duckdb_path : Full file path to desired DuckDB database location.
        log : Full file path to existing/desired log location.

        """
        self.sqlite_path = sqlite_path
        self.duckdb_path = duckdb_path
        self.log = log
        
    def metadata_indexes(self, sqlitedb, duckdb):
        inspector = sqlitedb.inspect(sqlitedb.engine)
        
        # Get indexes
        indexes = inspector.get_indexes("metadata")
        
        # Create indexes
        for idx in indexes:
            cols = ", ".join(idx['column_names'])
            unique = "UNIQUE " if idx['unique'] else ""
            duckdb.write_query(f"CREATE {unique}INDEX {idx['name']} ON metadata ({cols});")

    def migrate_database(self):
        with multi_db_connection(self.sqlite_path, self.duckdb_path, self.log, db_type1="sqlite") as (sqlite, duck):
            # Obtain metadata of SQLite db tables
            md = sqlite.meta()
            md.reflect(bind=sqlite.engine)
            
            # Migrate each table
            for table_name in md.tables:
                self.log.info(f"Migrating table: {table_name}")
                
                try:
                    # Define table schema from exising
                    table = md.tables[table_name]
                    create_stmt = str(duck.create_table(table))
                    
                    if table_name == "metadata":
                        create_stmt = create_stmt.replace(
                            """"latestData" NUMERIC""", """"latestData" DATE"""
                        ).replace(
                            """"lastStatementCheck" NUMERIC""", """"lastStatementCheck" DATE"""
                        )
                        # Create table
                        duck.write_query(create_stmt)
                        self.metadata_indexes(sqlite, duck)
                    else:
                        # Change data types for smooth transfer
                        create_stmt = create_stmt.replace(
                            "volume INTEGER", "volume BIGINT"
                        ).replace(
                            """"marketCap" INTEGER""", """"marketCap" FLOAT"""
                        ).replace(
                            """"enterpriseVal" INTEGER""", """"enterpriseVal" FLOAT"""
                        )
                        # Create table
                        duck.write_query(create_stmt)
                    
                    # Get SQLite data as DataFrame
                    df = sqlite.query_to_df(
                        f"SELECT * FROM '{table_name}';"
                    )
                    # Write DataFrame to DuckDB
                    df.to_sql(table_name, index=False, con=duck.engine,
                              if_exists="append", chunksize=1000, method="multi")
                    
                    self.log.info(f"{table_name} copied to DuckDB")
                    
                    duck.cursor.commit()
                
                except Exception as e:
                    log.error(e)
                    continue
                    
            self.log.info("Migration completed successfully!")
                

base_path = "/home/craig99/.technically"
log = log_setup("DEBUG", base_path + "/logs/dev/sqlite_to_duck.log")

#SQLiteToDuckDB(base_path + "/sql/prices.sqlite", base_path + "/sql/prices.duck", log).migrate_database()
