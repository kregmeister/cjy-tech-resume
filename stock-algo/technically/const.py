#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 09:35:12 2025

@author: cjymain
"""

import os
from datetime import date

from technically.utils.log import log_setup
from technically.utils.date_checks import which_friday

# Sets the base path for the system
TC_PATH = os.path.expanduser("~") + "/.technically"

# Date is constant throughout run (even after midnight)
CURRENT_DATE = date.today()

# Initializes log
TC_LOG = log_setup(CURRENT_DATE, TC_PATH)

# Other constants that are used throughout
GROUP_BY = "sector"
IS_FRIDAY = which_friday(CURRENT_DATE)  # Int or False

# Mininum number of data points required for a ticker to be included in the program
MIN_PERIODS = 120

# Steps to take to initialize technically data pipeline
INITIALIZATION_NOTICE = (
    "Technically is not ready to be executed. Next installation steps: "
    "1. TiingoAPI subscription required."
    "2. Place package files into a new python3.11 virtual environment. "
    "3. Activate the virtual environment and run 'pip install -r requirements.txt'. "
    "4. Install and configure AWS CLI V2. "
    "5. Store API keys in AWS Secrets Manager. "
    "6. Install PostgreSQL, psql, DuckDB, and Amazon Corretto (or any JDK). "
    "7. Permit user that runs technically to access PostgreSQL. "
    "8. Open default PostgreSQL database and grant all permissions to user. "
    "9. Create a database named 'technically'. "
    "10. Create three schemas in new DB: 'prices', 'fundamentals', & 'models'. "
    "NOTE: You will only see these instructions once. "
    "Some or all of these steps may be automated for convenience in the future. "
    "This package is not a distribution and is intended for personal use only. "
    "These instructions are here for reference and clarity."
)

# Default date dividers to use when bucketing data
DATE_BUCKETS = [
    "1997-01-01",
    "2003-01-01",
    "2009-01-01",
    "2017-01-01",
    "2020-01-01",
]

INCREMENTAL_CALC_COLS = [
    "demand_idx",
    "kalman_close",
    "kalman_trend",
    "close_percent_from_min",
    "close_percent_from_max",
    "atr",
    "aroon",
    "cci",
    "cci_ma",
    "rsi",
    "stdev_percent_of_sma",
    "bollinger_upper_band",
    "bollinger_sma",
    "bollinger_lower_band",
    "bollinger_band_range",
    "kst",
    "kst_sig",
    "macd",
    "macd_sig",
    "macd_hist",
    "wpr",
    "stochastic_k",
    "stochastic_d",
    "psar",
    "adx",
    "dmi_plus",
    "dmi_minus",
    "ema_5",
    "ema_10",
    "ema_20",
    "ema_50",
    "ema_100",
    "ema_250"
]

TREND_PERIODS = {
    "two_week": 10,
    "month": 20,
    "three_month": 60,
    "six_month": 120,
    "year": 250,
    "three_year": 750
}

TREND_STRENGTH_SCORING = {
    "close_diff": 25,
    "macd": 30,
    "dmi_diff": 25,
    "obv": 20
}

# Structured technical indicator signals (assessed by signals.py)
INDICATOR_SIGNALS_CONFIG = {
    "indicator": [
        "demand_idx",
        "wpr",
        "macd",
        "cci",
        "rsi",
        "bollinger_sma",
        "kst",
        "stochastic_k",
        "adx",
        "psar"
    ],
    "bullish_signals": [
        {
            "indicator": "demand_idx",
            "threshold": {
                "name": "demand_idx_threshold",
                "above": "90th",
                "current_trend": 1
            },
            "multi": {
                "name": "demand_idx_cross_zero",
                "crossover": {
                    "above": 0,
                    "current_trend": -1
                },
                "crossover2": {
                    "below": 0,
                    "current_trend": -1
                }
            },
            "divergence_long": {
                "name": "demand_idx_longterm_divergence",
                "cofeature": "close",
                "above": "75th",
                "direction": 1,
                "length": 10,
                "current_trend": -1
            },
            "divergence": {
                "name": "demand_idx_trend_divergence",
                "cofeature": "close",
                "above": "80th",
                "trend": 20,
                "direction": 1,
                "current_trend": -1
            }
        },
        {
            "indicator": "wpr",
            "multi": {
                "name": "wpr_divergence",
                "threshold": {
                    "below": -80
                },
                "divergence": {
                    "cofeature": "close",
                    "above": "75th",
                    "current_trend": -1
                }
            },
            "threshold": {
                "name": "wpr_3_period_threshold",
                "below": -80,
                "length": 3,
                "current_trend": -1
            }
        },
        {
            "indicator": "macd",
            "multi": {
                "name": "macd_price_divergence",
                "threshold": {
                    "below": "10th"
                },
                "divergence": {
                    "feature": "macd_hist",
                    "cofeature": "close",
                    "above": "85th",
                    "trend": 19,
                    "direction": 1,
                    "current_trend": -1
                }
            },
            "multi2": {
                "name": "macd_crossover_sigline",
                "threshold": {
                    "below": "40th"
                },
                "crossover": {
                    "above": "macd_sig"
                }
            },
            "threshold": {
                "name": "macd_5_period_low",
                "below": "5th",
                "length": 5
            },
            "crossover": {
                "name": "macd_cross_above_zero",
                "above": 0
            }
        },
        {
            "indicator": "cci",
            "multi": {
                "name": "cci_trend_reversal",
                "threshold_trend": {
                    "trend": 5,
                    "above": 0
                },
                "threshold": {
                    "below": -50
                },
                "crossover": {
                    "above": "cci_ma"
                }
            },
            "multi2": {
                "name": "cci_divergence",
                "threshold": {
                    "trend": 10,
                    "above": "60th"
                },
                "threshold_close": {
                    "feature": "close",
                    "trend": 10,
                    "below": "40th"
                },
                "divergence": {
                    "cofeature": "close",
                    "above": "80th",
                    "trend": 10,
                    "direction": 1
                }
            }
        },
        {
            "indicator": "rsi",
            "multi": {
                "name": "rsi_divergence",
                "threshold": {
                    "trend": 14,
                    "above": "60th"
                },
                "threshold_close": {
                    "feature": "close",
                    "trend": 14,
                    "below": "40th"
                },
                "divergence": {
                    "cofeature": "close",
                    "above": "80th",
                    "trend": 14,
                    "direction": 1
                }
            },
            "failure_swing": {
                "name": "rsi_failure_swing",
                "bottom": 14,
                "thresh": 30,
                "current_trend": -1
           },
            "threshold": {
                "name": "rsi_threshold",
                "below": 30,
                "length": 2
           }
        },
        {
            "indicator": "bollinger_sma",
            "threshold_squeeze": {
                "name": "bollinger_squeeze",
                "feature": "bollinger_band_range",
                "zscored": 250,
                "below": "10th",
                "current_trend": -1
            },
            "threshold": {
                "name": "bollinger_threshold",
                "feature": "close",
                "below": "bollinger_lower_band"
            }
        },
        {
            "indicator": "kst",
            "multi": {
                "name": "kst_crossover_sigline",
                "crossover": {
                    "above": "kst_sig"
                },
                "threshold": {
                    "below": "20th"
                }
            },
            "crossover": {
                "name": "kst_cross_above_zero",
                "above": 0
            }
        },
        {
            "indicator": "stochastic_k",
            "multi": {
                "name": "stochastic_crossover_d",
                "crossover": {
                    "above": "stochastic_d"
                },
                "threshold": {
                    "below": 20
                }
            },
            "divergence": {
                "name": "stochastic_price_divergence",
                "cofeature": "close",
                "above": "80th",
                "trend": 20,
                "direction": -1,
                "current_trend": -1
            }
        },
        {
            "indicator": "adx",
            "multi": {
                "name": "adx_threshold_crossover",
                "threshold": {
                    "above": 20
                },
                "crossover": {
                    "feature": "dmi_plus",
                    "above": "dmi_minus"
                }
            },
            "multi2": {
                "name": "adx_trend_after_low",
                "threshold": {
                    "below": 20
                },
                "threshold_adx_trend": {
                    "trend": 10,
                    "above": "75th"
                },
                "threshold_pc_from_min": {
                    "feature": "close_percent_from_min",
                    "below": "25th"
                }
            }
        },
        {
            "indicator": "psar",
            "crossover": {
                "name": "psar_below_close",
                "below": "close"
            }
        }
    ],
    "bearish_signals": [
        {
            "indicator": "demand_idx",
            "divergence": {
                "name": "demand_idx_trend_divergence",
                "cofeature": "close",
                "above": "75th",
                "trend": 20,
                "direction": -1,
                "current_trend": 1
            },
            "multi": {
                "name": "demand_idx_cross_zero",
                "crossover": {
                    "above": 0,
                    "current_trend": 1
                },
                "crossover2": {
                    "below": 0,
                    "current_trend": 1
                }
            },
            "divergence_long": {
                "name": "demand_idx_longterm_divergence",
                "cofeature": "close",
                "above": "75th",
                "zscored": 250,
                "direction": -1,
                "length": 10,
                "current_trend": 1
            }
        },
        {
            "indicator": "wpr",
            "multi": {
                "name": "wpr_divergence",
                "threshold": {
                    "above": -20
                },
                "divergence": {
                    "cofeature": "close",
                    "above": "75th",
                    "current_trend": 1
                }
            },
            "threshold": {
                "name": "wpr_3_period_threshold",
                "above": -20,
                "length": 3,
                "current_trend": 1
            }
        },
        {
            "indicator": "macd",
            "multi": {
                "name": "macd_price_divergence",
                "threshold": {
                    "above": "90th"
                },
                "divergence": {
                    "feature": "macd_hist",
                    "cofeature": "close",
                    "above": "85th",
                    "trend": 19,
                    "direction": -1,
                    "current_trend": 1
                }
            },
            "multi2": {
                "name": "macd_crossover_sigline",
                "threshold": {
                    "above": "60th"
                },
                "crossover": {
                    "below": "macd_sig"
                }
            },
            "threshold": {
                "name": "macd_5_period_high",
                "above": "95th",
                "length": 5
            },
            "crossover": {
                "name": "macd_cross_below_zero",
                "below": 0
            }
        },
        {
            "indicator": "cci",
            "multi": {
                "name": "cci_trend_reversal",
                "threshold": {
                    "trend": 5,
                    "below": 0
                },
                "threshold2": {
                    "above": 50
                },
                "crossover": {
                    "below": "cci_ma"
                }
            },
            "multi2": {
                "name": "cci_divergence",
                "threshold": {
                    "trend": 10,
                    "below": "40th"
                },
                "threshold_close": {
                    "feature": "close",
                    "trend": 10,
                    "above": "60th"
                },
                "divergence": {
                    "cofeature": "close",
                    "above": "75th",
                    "trend": 10,
                    "direction": -1
                }
            }
        },
        {
            "indicator": "rsi",
            "multi": {
                "name": "rsi_divergence",
                "threshold": {
                    "trend": 14,
                    "below": "40th"
                },
                "threshold_close": {
                    "feature": "close",
                    "trend": 14,
                    "above": "60th"
                },
                "divergence": {
                    "cofeature": "close",
                    "above": "80th",
                    "trend": 14,
                    "direction": -1
                }
            },
            "failure_swing": {
                "name": "rsi_failure_swing",
                "top": 14,
                "thresh": 70,
                "current_trend": 1
           },
            "threshold": {
                "name": "rsi_threshold",
                "above": 70,
                "length": 2
            }
        },
        {
            "indicator": "bollinger_sma",
            "threshold_squeeze": {
                "name": "bollinger_squeeze",
                "feature": "bollinger_band_range",
                "zscored": 250,
                "below": "10th",
                "current_trend": 1
            },
            "threshold": {
                "name": "bollinger_threshold",
                "feature": "close",
                "above": "bollinger_upper_band"
            }
        },
        {
            "indicator": "kst",
            "multi": {
                "name": "kst_crossover_sigline",
                "crossover": {
                    "below": "kst_sig"
                },
                "threshold": {
                    "above": "80th"
                }
            },
            "crossover": {
                "name": "kst_cross_below_zero",
                "below": 0
            }
        },
        {
            "indicator": "stochastic_k",
            "multi": {
                "name": "stochastic_crossover_d",
                "crossover": {
                    "below": "stochastic_d"
                },
                "threshold": {
                    "above": 80
                }
            },
            "divergence": {
                "name": "stochastic_price_divergence",
                "cofeature": "close",
                "above": "80th",
                "trend": 20,
                "direction": 1,
                "current_trend": 1
            }
        },
        {
            "indicator": "adx",
            "multi": {
                "name": "adx_threshold_crossover",
                "threshold": {
                    "above": 20
                },
                "crossover": {
                    "feature": "dmi_minus",
                    "above": "dmi_plus"
                }
            },
            "multi2": {
                "name": "adx_trend_after_high",
                "threshold": {
                    "below": 20
                },
                "threshold_adx_trend": {
                    "trend": 10,
                    "below": "25th"
                },
                "threshold_pc_from_max": {
                    "feature": "close_percent_from_max",
                    "below": "25th"
                }
            }
        },
        {
            "indicator": "psar",
            "crossover": {
                "name": "psar_above_close",
                "above": "close"
            }
        }
    ]
}

# Specifies which columns are altered using various feature engineering techniques
ML_FEATURE_ENGINEERING_CONFIG = {
    "technicals": {
        "cols_to_scale": {
            "type": "Robust",
            "features": [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "ema10",
                "ema20",
                "ema50",
                "ema100",
                "ema250",
                "kama",
                "two_week_trend_line",
                "month_trend_line",
                "three_month_trend_line",
                "six_month_trend_line",
                "year_trend_line",
                "three_year_trend_line",
                "prevailing_trend_line",
                "two_week_trend",
                "month_trend",
                "three_month_trend",
                "six_month_trend",
                "year_trend",
                "three_year_trend",
                "prevailing_trend",
                "daily_price_change",
                "daily_price_ceiling",
                "daily_price_floor",
                "kalman_close",
                "close_percent_from_max",
                "close_percent_from_min",
                "stdev_percent_of_sma",
                "bollinger_upper_band",
                "bollinger_sma",
                "bollinger_lower_band",
                "bollinger_band_range",
                "psar",
                "obv"
            ]
        },
        "cols_to_lag": {
            "lag_periods": [
                1,
                2,
                3,
                5
            ],
            "features": [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "kalman_close",
                "ema10",
                "ema20",
                "ema50",
                "ema100",
                "ema250"
            ]
        },
        "cols_to_roll": {
            "windows": [
                5,
                10,
                20
            ],
            "features": [
                "volume",
                "obv",
                "market_cap",
                "enterprise_val"
            ]
        }
    },
    "fundamentals": {
        "cols_to_scale": {
            "type": "Robust",
            "features": [
                "cash_and_eq",
                "debt",
                "equity",
                "retained_earnings",
                "total_assets",
                "total_liabilities",
                "capex",
                "consolidated_income",
                "cost_rev",
                "ebit",
                "ebitda",
                "ebt",
                "gross_profit",
                "netinc",
                "opex",
                "opinc",
                "revenue",
                "rnd",
                "sga",
                "tax_exp"
            ]
        },
        "cols_to_roll": {
            "windows": [
                2,
                4
            ],
            "features": [
                "cash_and_eq",
                "debt",
                "equity",
                "retained_earnings",
                "total_assets",
                "total_liabilities",
                "capex",
                "consolidated_income",
                "cost_rev",
                "ebit",
                "ebitda",
                "ebt",
                "gross_profit",
                "netinc",
                "opex",
                "opinc",
                "revenue",
                "rnd",
                "sga",
                "tax_exp"
            ]
        }
    }
}
