import pandas as pd

def non_negative_series(series):
    series = series.copy(deep=True)
    series['Returns'] = series['Close'].diff() / series['Close'].shift(1)
    series['rPrices'] = (1 + series['Returns']).cumprod()
    return series

def daily_bars(df):
    df = df.copy()
    grouped = df.groupby(df.index.date)

    bars = grouped.agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'Instrument': 'first'
    })
    bars['start_time'] = grouped.apply(lambda x: x.index.min())
    bars['end_time'] = bars['start_time'].shift(-1)
    bars.iloc[-1, bars.columns.get_loc('end_time')] = df.index.max()
    bars = bars.set_index("start_time")
    bars.index.name = None
    bars["start_time"] = bars.index
    cols = [c for c in bars.columns if c not in ['start_time', 'end_time']]
    bars = bars[[*cols, 'start_time', 'end_time']]
    return bars

def volume_bars(df, bar_size=10000):
    df = df.copy()
    df['CumVolume'] = df['Volume'].cumsum()
    bar_idx = (df['CumVolume'] / bar_size).round().astype(int)

    grouped = df.groupby(bar_idx)

    bars = grouped.agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'Instrument': 'first'
    })
    bars['start_time'] = grouped.apply(lambda x: x.index.min())
    bars['end_time'] = bars['start_time'].shift(-1)
    bars.iloc[-1, bars.columns.get_loc('end_time')] = df.index.max()
    bars = bars.set_index("start_time")
    bars.index.name = None
    bars["start_time"] = bars.index
    cols = [c for c in bars.columns if c not in ['start_time', 'end_time']]
    bars = bars[[*cols, 'start_time', 'end_time']]

    return bars


def dollar_bars(df, bar_size=10000 * 3000):
    df = df.copy()
    df['Dollar Volume'] = df['Volume'] * df['Close']
    df['CumDollarVolume'] = df['Dollar Volume'].cumsum()
    bar_idx = (df['CumDollarVolume'] / bar_size).round().astype(int)

    grouped = df.groupby(bar_idx)

    bars = grouped.agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'Instrument': 'first'
    })
    bars['start_time'] = grouped.apply(lambda x: x.index.min())
    bars['end_time'] = bars['start_time'].shift(-1)
    bars.iloc[-1, bars.columns.get_loc('end_time')] = df.index.max()
    bars = bars.set_index("start_time")
    bars.index.name = None
    bars["start_time"] = bars.index
    cols = [c for c in bars.columns if c not in ['start_time', 'end_time']]
    bars = bars[[*cols, 'start_time', 'end_time']]

    return bars