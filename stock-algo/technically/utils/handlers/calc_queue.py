#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:35:28 2024

@author: cjymain
"""
import threading
import queue
import traceback

from technically.utils.log import get_logger, timer


class CalculationsOrchestrator:
    """
    Calculates and stores technicals in near-live time as price data is consumed.
    """

    def __init__(self, worker_class, indicator_success_rates, threads=2, max_idle_time=90):
        self.worker_class = worker_class  # Must have specific parameters
        self.indicator_success_rates = indicator_success_rates

        self.ticker_queue = queue.Queue()
        self.num_threads = threads
        self.threads = []
        self.stop_event = threading.Event()
        self.max_idle_time = max_idle_time

        self.start_workers()

    def start_workers(self):
        """
        Starts worker threads for the queue.
        """
        for i in range(self.num_threads):
            t = threading.Thread(
                target=self.worker,
                name=f'{self.worker_class.__name__}-{i + 1}'
            )
            t.start()
            self.threads.append(t)

    @timer()
    def worker(self):
        """
        Main thread worker function.
        """
        thread_name = threading.current_thread().name
        while True:
            try:
                ticker, exchange, cap, sector, last_check, initialized = self.ticker_queue.get(timeout=1)
            except queue.Empty:
                if self.stop_event.is_set():
                    if self.ticker_queue.empty():
                        get_logger().info(
                            "Thread stopping. Queue is empty.", extra={
                                "thread_name": thread_name
                            }
                        )
                        break
                    else:
                        continue  # Queue is not empty, continue processing
                else:
                    continue  # Continue waiting for items
            except Exception:
                get_logger().critical(
                    "Unexpected error in worker thread.", extra={
                        "thread_name": thread_name,
                        "error": traceback.format_exc()
                    }
                )
                self.stop_workers_signal()
                break

            # Process the item
            self.worker_class(ticker, exchange, cap, sector,
                              last_check, initialized, self.indicator_success_rates).execute()
            self.ticker_queue.task_done()

    def add_ticker(self, ticker: list):
        """
        Adds ticker to the queue to be processed.
        """
        self.ticker_queue.put(ticker)

    def stop_workers_signal(self):
        """
        Signal workers to stop.
        """
        self.stop_event.set()
        self.wait_on_workers()

    def wait_on_workers(self):
        """
        Waits until all worker threads have stopped.
        """
        self.ticker_queue.join()
        for t in self.threads:
            t.join()
        get_logger().info(
            "All worker threads have terminated gracefully.", extra={
                "worker_class": self.worker_class.__name__
            }
        )
