#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 22:25:52 2024

@author: cjymain
"""

import boto3
import json
from time import sleep
import traceback

from technically.utils.log import get_logger


def get_credentials(keys: list):
    """
    Retrieves sensitive credentials from a secure location (AWS Secrets Manager).

    Args:
        keys: List of strings representing the AWS Secrets Manager key names.

    Returns:
        response: List of strings representing the returned AWS Secrets Manager values.
    """

    retries = 0
    while True:
        secret_name = "TechnicallyAuth"
        region_name = "us-east-1"

        # Create a Secrets Manager client
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )

        try:
            get_secret_value_response = client.get_secret_value(
                SecretId=secret_name
            )
        except Exception as e:
            get_logger().error(
                f"Error connecting to AWS Secrets Manager.", extra={
                    "num_retries": retries,
                    "error": traceback.format_exc()
                }
            )
            if retries == 10:
                raise e
            # Exponential backoff
            retries += 1
            sleep(retries ** 2)
            continue

        secret = json.loads(get_secret_value_response["SecretString"])

        response = [value for key, value in secret.items() if key in keys]

        if len(response) == 1:
            return response[0]
        else:
            return response


        