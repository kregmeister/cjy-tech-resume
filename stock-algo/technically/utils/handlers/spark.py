#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:15:43 2025

@author: cjymain
"""

from pyspark.sql import SparkSession
import os
import sys
import traceback

from technically.utils.log import get_logger


class PySparkSession:
    """
    Manages spark session creation and data extraction.
    """

    def __init__(self):
        # Tells pyspark to point to TC venv for imports
        venv = sys.executable
        os.environ["PYSPARK_PYTHON"] = venv
        os.environ["PYSPARK_DRIVER_PYTHON"] = venv

    def __enter__(self):
        self.session = SparkSession.builder \
            .appName("sparkClusterTC") \
            .config("spark.driver.memory", "16g") \
            .config("spark.driver.maxResultSize", "8g") \
            .getOrCreate()

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        try:
            if exc_type is not None:
                error_code = "".join(
                    traceback.format_exception(
                        exc_type, exc_value, exc_traceback
                    )
                )
                get_logger().error(error_code)
        except Exception as e:
            get_logger().error(f"PySpark session cleanup error: {str(e)}")
            raise
        finally:
            self.session.stop()