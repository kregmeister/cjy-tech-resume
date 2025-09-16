---TO ALTER TECHNICAL INDICATOR SIGNALS, SEE VARIABLE 'INDICATOR_SIGNALS_CONFIG' IN const.py---

Technical indicator signal example structure:
{
    "indicator": "macd",
    "threshold" : {  # Keyword
        "name": "macd_threshold",  # Signal name
        "above": "90th",  # Condition
        "current_trend": 1
    }
}

Rules for creating technical indicator trade signals in indicator_signals dictionary:
- Keyword identifiers must begin with: 'threshold', 'crossover', 'failure_swing', 'divergence', 'convergence', or 'multi'
- Possible
- A keyword cannot be repeated across an indicator; when a keyword is used more than once, use 'divergence' and 'divergence_long', for example
- Each signal name must be unique across all indicators
- If a signal has multiple conditions, the keyword must begin with 'multi'
- When passing feature/cofeature, it must exist as a column in the dataframe
- Conditions may contain both primary and supplementary conditions:
    ~ Threshold --> Primary: 'above', 'below'. Supplementary: 'feature', 'trend', 'zscored', 'length', 'current_trend'
    ~ Crossover --> Primary: 'above', 'below'. Supplementary: 'feature', 'current_trend'
    ~ Failure_swing --> Primary: 'top', 'bottom'. Supplementary: 'feature', 'current_trend'
    ~ Divergence/Convergence --> Primary: 'above', 'below', 'cofeature'. Supplementary: 'feature', 'trend', 'direction', 'length', 'current_trend'
- Condition descriptions:
    ~ Above/below (int(70) or str('50th')) --> In all instances, specifies one of two directions of data comparisons (i.e. trend of closing price is diverging below trend of bollinger SMA)
    ~ Feature (str('kama')) --> Used to pass a custom column to be used in generating signals; when not specified, the indicator's data is used (i.e. using DMI+ and DMI- rather than ADX to detect ADX crossovers)
    ~ Cofeature (str('macd_hist') --> Only used in divergence/convergence. Is required (i.e. trend of closing price diverges above trend of stochastic indicator)
    ~ Trend (int(10)) --> Converts feature/cofeature scalars to trend of X periods for comparison (i.e. demand index values frequently fluctuate, its 20 period trend is more indicative of its activity relative to closing price)
    ~ Zscored (int(250)) --> Converts feature/cofeature scalars to zscore standardized values for comparison (i.e. closing price and williams percent R are at different scales; should be zscored for proper comparison)
    ~ Length (int(3)) --> A condition must be true for X periods in a row to actuate (i.e. commodity channel index must be below a threshold for 3 straight periods)
    ~ Current_trend (1, 0, or -1) --> The condition must occur during a specific trend (i.e. bullish relative strength index failure swings only actuate during downtrend)
