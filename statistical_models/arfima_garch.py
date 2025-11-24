from scipy.stats import t, kstest
import pandas as pd
import tpqoa
import matplotlib.pyplot as plt
from fracdiff import findMinFFD_fromData, fracDiff_FFD, inverse_fracdiff
import pmdarima as pm
import numpy as np
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch
from scipy.stats import jarque_bera, shapiro

def fit_model(y, series_size=0.9, confidence_level_diff=0.95, 
              p_value_arch=0.05, p_value_ljung=0.05, p_value_student=0.05):

    t_params = None

    d_optimal, thres_optimal = findMinFFD_fromData(
        y, col='Close', confidence_level=confidence_level_diff, series_size=series_size
    )
    y_stationary = fracDiff_FFD(y[['Close']], d=d_optimal, thres=thres_optimal).dropna()

    arima = pm.auto_arima(
        y_stationary, d=0,
        start_p=0, max_p=5,
        start_q=0, max_q=5,
        seasonal=False,
        stepwise=True,
        trace=True,
        information_criterion='aic'
    )

    residuals = pd.Series(arima.resid())
    arch_test = het_arch(residuals)

    if arch_test[1] < p_value_arch:

        best_aic = np.inf
        best_fit = None

        for p in range(1, 4):
            for q in range(1, 4):
                try:
                    fit = arch_model(
                        residuals, vol='Garch', p=p, q=q, dist='StudentsT'
                    ).fit(disp='off')

                    if np.isfinite(fit.aic) and fit.aic < best_aic:
                        best_aic = fit.aic
                        best_fit = fit

                except Exception as e:
                    print(f"Error fitting GARCH({p},{q}): {e}")
                    continue

        if best_fit is not None:
            garch = best_fit

            z = garch.resid / (garch.conditional_volatility + 1e-8)

            ljung_z_p = acorr_ljungbox(z, lags=[10], return_df=True)['lb_pvalue'].iloc[-1]
            arch_pvalue_z = het_arch(z)[1]

            df_t, loc_t, scale_t = t.fit(z)
            t_params = (df_t, 0, scale_t)

            ks_stat, ks_pvalue = kstest(z, 't', args=t_params)
            print(f"Kolmogorov–Smirnov test for Student-t: p = {ks_pvalue:.4f}")

            if (ljung_z_p <= p_value_ljung) or (arch_pvalue_z <= p_value_arch) or (ks_pvalue <= p_value_student):
                garch, arima, t_params = None, None, None
        else:
            garch, arima, t_params = None, None, None

    else:
        garch = None

        residuals = pd.Series(arima.resid())
        ljung_resid_p = acorr_ljungbox(residuals, lags=[10], return_df=True)['lb_pvalue'].iloc[-1]

        df_t, loc_t, scale_t = t.fit(residuals)
        t_params = (df_t, loc_t, scale_t)
        ks_stat, ks_pvalue = kstest(residuals, 't', args=t_params)
        print(f"Kolmogorov–Smirnov test for Student-t residuals: p = {ks_pvalue:.4f}")
        if (ljung_resid_p <= p_value_ljung) or (ks_pvalue <= p_value_student):
            arima, t_params = None, None

    return y_stationary, arima, garch, d_optimal, thres_optimal, t_params

