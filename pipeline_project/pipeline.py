from triple_barrier import getDailyVol, getTEvents, getEvents, getBins
from bars import volume_bars
from seqbootstrap import getIndMatrix, seqBootstrap
from purged_kfold import PurgedKFold
import pandas as pd
from multiprocessing import cpu_count
import tpqoa
import numpy as np

def build_labeled_datasets(df, h_factor=1, trgt_factor=1, minRet=0.0005, 
                      ptSl=1, numDays=5, sampling_method=None,
                      nBootstraps=1, k_folds=3, pctEmbargo=0.01,
                      n_lags=10, label_col="bin"):
    """
    Triple-barrier labeling pipeline with options:
      - 'seqBootstrap'
      - 'PurgedKFold'
      - None (no sampling)

    Also returns a dataset with lagged candle features.
    """

    # --- Keep 2 views of the dataset ---
    bars_full = df.copy()                   # full historical dataset
    valid_idx = bars_full.index[n_lags-1:]
    # --- Volatility & event detection ---
    close = bars_full['Close']
    daily_vol = getDailyVol(close)
    h_threshold = daily_vol.mean() * h_factor
    tEvents = getTEvents(close, h=h_threshold)

    # --- Triple-barrier construction ---
    trgt = daily_vol * trgt_factor
    trgt_reindexed = trgt.reindex(tEvents, method='ffill')
    t1_idx = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
    t1_idx = t1_idx[t1_idx < close.shape[0]]
    t1 = pd.Series(close.index[t1_idx], index=tEvents[:t1_idx.shape[0]])
    numThreads = min(4, cpu_count())
    events = getEvents(close, tEvents, ptSl, trgt_reindexed, minRet, numThreads, t1=t1)
    bins = getBins(events, close)
    events_complete = events.dropna(subset=['t1'])

    # --- Lagged feature construction ---
    # Match bins with candles using merge_asof (safe alignment)
    bins_valid = bins.loc[bins.index.intersection(valid_idx)]
    bins_bars = bins_valid.merge(bars_full, left_index=True, right_index=True, how="left")

    # Build column names
    feat_cols = []
    for lag in range(n_lags, 0, -1):
        for col in bars_full.columns:
            feat_cols.append(f"{col}_t-{lag-1}")

    windows = []

    for _, idx in enumerate(bins_bars.index):
        pos = bars_full.index.get_loc(idx)
        window = bars_full.iloc[pos - n_lags + 1: pos + 1]  # full window
        windows.append(window.values.flatten())
    
    # Final dataset
    Xy = pd.DataFrame(windows, columns=feat_cols, index=bins_bars.index)
    Xy[label_col] = bins_valid[label_col].loc[Xy.index]

    new_index = Xy[f"start_time_t-{n_lags-1}"]
    events_complete = events_complete.loc[events_complete.index.intersection(Xy.index)]
    
    # Create extended version (based on the oldest bar of the lagged window)
    lagged_events = events_complete.copy()
    lagged_events.index = new_index.loc[lagged_events.index].values

    lagged_bins = bins_valid.copy()
    lagged_bins.index = new_index.loc[bins_valid.index].values

    lagged_Xy = Xy.copy()
    lagged_Xy.index = new_index.loc[Xy.index].values

    # --- Sampling ---
    bins_bootstrap, bins_oos = [], []
    
    if sampling_method == 'seqBootstrap':
        indM = getIndMatrix(close.index, lagged_events['t1'])
        for _ in range(nBootstraps):
            phi = seqBootstrap(indM)
            bins_bootstrap.append(lagged_bins.index[phi])
            all_idx = set(range(len(lagged_bins)))
            bins_oos.append(lagged_bins.index[list(all_idx - set(phi))])

    elif sampling_method == 'PurgedKFold':
        pkf = PurgedKFold(n_splits=k_folds, t1=lagged_events['t1'], pctEmbargo=pctEmbargo)
        for train_idx, test_idx in pkf.split(lagged_bins):
            bins_bootstrap.append(lagged_bins.iloc[train_idx].index)
            bins_oos.append(lagged_bins.iloc[test_idx].index)
            # to also purge features + labels, not just labels
    elif sampling_method is None:
        bins_bootstrap, bins_oos = [], []
        
    else:
        raise ValueError("sampling_method must be 'seqBootstrap', 'PurgedKFold' or None")

    datasets = []
    
    for boot_idx, oos_idx in zip(bins_bootstrap, bins_oos):
        # Train/test features and labels
        X_train = lagged_Xy.loc[boot_idx].drop(columns=[label_col])
        y_train = lagged_Xy.loc[boot_idx, label_col]

        X_test = lagged_Xy.loc[oos_idx].drop(columns=[label_col])
        y_test = lagged_Xy.loc[oos_idx, label_col]

        datasets.append({
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        })

    return {
        'Xy': lagged_Xy,
        'datasets': datasets
    }

if __name__ == "__main__":
    # Example usage
    api = tpqoa.tpqoa("../oanda.cfg")
    df = api.get_history(instrument = "EUR_USD", start = "2024-07-01", end = "2024-12-30",
                granularity = "H1", price = "B")
    df = df.rename(columns={
    "o": "Open",
    "h": "High",
    "l": "Low",
    "c": "Close",
    "volume": "Volume"
        })
    df["Instrument"] = "EUR_USD"

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = np.log(df[col])

    bars = volume_bars(df, bar_size=10000)
    results = build_labeled_datasets(bars, sampling_method='PurgedKFold', k_folds=4, minRet=0.0001, n_lags=5, pctEmbargo=0.01, h_factor=1/10)
    print(results["Xy"].head())