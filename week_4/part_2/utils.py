from sktime.utils.plotting import plot_correlations
from statsmodels.tsa.stattools import adfuller, kpss, acf
import pandas as pd
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
import matplotlib.pyplot as plt

def tsdisplay(y, lags=36, title=None, figsize=(13, 6)):
    """Панель «ряд + ACF + PACF» — сквозной инструмент диагностики семинара."""
    fig, ax = plot_correlations(y, lags=lags, suptitle=title, acf_title="ACF", pacf_title="PACF")
    fig = plt.figure(figsize=figsize)
    return fig


def unit_root_report(series, title="", adf_spec="c", kpss_spec="c", alpha=0.05):
    """Совместный отчёт ADF + KPSS с готовым вердиктом по таблице 2×2 (ноутбук 02)."""
    series = pd.Series(series).dropna()
    adf_stat, adf_p, *_ = adfuller(series, regression=adf_spec, autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kpss_stat, kpss_p, _, _ = kpss(series, regression=kpss_spec, nlags="auto")

    adf_unit_root = adf_p >= alpha
    kpss_stationary = kpss_p >= alpha

    verdict = {
        (True, True): "мало данных — вывода нет",
        (True, False): "нестационарен ⇒ нужна разность",
        (False, True): "стационарен",
        (False, False): "противоречие ⇒ ни то ни сё",
    }[(adf_unit_root, kpss_stationary)]

    return {
        "ряд": title,
        "ADF p": round(adf_p, 4),
        "ADF: ед. корень?": "да" if adf_unit_root else "нет",
        "KPSS p": round(kpss_p, 4),
        "KPSS: стационарен?": "да" if kpss_stationary else "нет",
        "вердикт": verdict,
    }