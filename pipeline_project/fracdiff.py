'''Target: fractionally difference the series to make it stationary, while preserving as much memory of the original series as possible.'''

'''How: by applying fractional differencing (fracdiff), we find the minimum order d that makes the series stationary according to the ADF
test at a given confidence level. For efficiency we iterate d from 0 to 1 in steps of 0.1, although this step size can be adjusted'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def getWeights_FFD(d, thres=1e-5):
    w = [1.]
    k = 1
    while True:
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def fracDiff_FFD(series, d, thres=1e-5):
    w = getWeights_FFD(d, thres)
    width = len(w) - 1
    df = {}
    for name in series.columns:
        seriesF = series[[name]].ffill().dropna()
        df_ = pd.Series(dtype='float64')
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue
            window_data = seriesF.loc[loc0:loc1].values.flatten()
            df_.at[loc1] = np.dot(w.T, window_data)[0]
        df[name] = df_
    return pd.concat(df, axis=1)

def findThreshold(series, d, seed_threshold, series_size, tol=0.1, max_iter=10):
    n = len(series)
    target_size = int(series_size * n)

    low, high = 1e-8, 1.0
    threshold = seed_threshold

    for _ in range(max_iter):
        df_diff = fracDiff_FFD(series, d, thres=threshold)
        current_size = df_diff.dropna().shape[0]

        if abs(current_size - target_size) <= tol * n:
            break

        if current_size > target_size:
            low = threshold
        else:
            high = threshold
        threshold = (low + high) / 2

    return threshold

def findMinFFD_fromData(df, col='Close', confidence_level=0.95,
                        seed_threshold=0.01, d_step=0.1, series_size=0.9, tol=0.1, max_iter=10):
    best_d = None
    best_thres = None
    valid_keys = ['1%', '5%', '10%']
    closest_key = min(valid_keys, key=lambda x: abs(int(x[:-1])/100 - (1 - confidence_level)))

    for d in np.arange(0, 1 + d_step, d_step):
        thres = findThreshold(df[[col]], d, seed_threshold, series_size=series_size, tol=tol, max_iter=max_iter)
        df_diff = fracDiff_FFD(df[[col]], d, thres=thres)
        common_idx = df.index.intersection(df_diff.index)
        y_valid = df_diff.loc[common_idx, col].dropna()
        if len(y_valid) > 1:
            try:
                adf_result = adfuller(y_valid, maxlag=1, regression='c', autolag=None)
                conf_val = adf_result[4][closest_key]
                if adf_result[0] < conf_val:
                    best_d = d
                    best_thres = thres
                    break
            except Exception as e:
                print(f"ADF falló en d={d}: {e}")
    return (np.round(best_d, 2) if best_d is not None else None,
            best_thres if best_thres is not None else None)

# Test script
if __name__ == "__main__":
    np.random.seed(27)
    dates = pd.date_range('2020-01-01', periods=500)
    data = np.cumsum(np.random.normal(0, 1, 500))
    df = pd.DataFrame(data, index=dates, columns=['Close'])

    min_d = findMinFFD_fromData(df, col='Close', confidence_level=0.95)
    print("Optimal d:", min_d)

    stationary_series = fracDiff_FFD(df[['Close']], min_d, thres=0.01)

    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'], label='Original Series', alpha=0.7)
    plt.plot(stationary_series.index, stationary_series['Close'], label=f'Stationary Series (d={min_d})', alpha=0.9)
    plt.title('Original vs Stationary Series')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


'''The threshold (thres) in fractional differencing controls when small weights are truncated. A lower threshold keeps more weights, 
increasing the window size used for each value. This improves accuracy but reduces the number of valid points in the resulting series, 
since only positions with enough past data can be computed. In short: lower thres → more accurate but shorter series.

To address this trade-off, a bisection method can be used to automatically find the optimal threshold. 
This method adjusts the threshold so that the resulting series preserves a desired proportion of the original data, 
balancing accuracy with series length. In other words, instead of choosing thres arbitrarily, the bisection ensures 
that the fractional differentiation is as precise as possible while maintaining enough valid observations for analysis.'''
