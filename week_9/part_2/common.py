"""Общая обвязка ноутбуков семинара 9: данные, метрики, графики.

Нужна, чтобы в ноутбуках оставались только вызовы библиотек и пояснения.
"""

import logging
import warnings
from pathlib import Path

# ETNA и sktime предупреждают об опциональных зависимостях прямо при импорте — глушим до него.
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"


def quiet() -> None:
    """Глушит предупреждения и логи библиотек, чтобы в ноутбуке был виден только результат."""
    warnings.filterwarnings("ignore")
    for name in ("etna", "joblib", "statsmodels", "matplotlib", "py.warnings", "cmdstanpy"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        from etna.loggers import tslogger

        tslogger.loggers.clear()
    except ImportError:
        pass


def mape(y_true, y_pred) -> float:
    """MAPE в процентах."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def load_airpassengers() -> pd.Series:
    """Месячные авиаперевозки 1949–1960, тыс. пассажиров; индекс — `PeriodIndex('M')`."""
    raw = pd.read_csv(DATA / "airpassengers.csv", parse_dates=["Month"])
    return pd.Series(
        raw["#Passengers"].to_numpy(dtype=float),
        index=pd.PeriodIndex(raw["Month"], freq="M"),
        name="passengers",
    )


def load_peyton() -> pd.Series:
    """Дневные просмотры страницы Пейтона Мэннинга в Википедии (уже в логарифме); индекс — даты как есть."""
    raw = pd.read_csv(DATA / "example_wp_log_peyton_manning.csv", parse_dates=["ds"])
    return pd.Series(raw["y"].to_numpy(dtype=float), index=pd.DatetimeIndex(raw["ds"]), name="y")


def plot_forecast(y_train, y_test, preds: dict, metric=mape, metric_name="MAPE", last=None, title=None, ax=None):
    """Train / test / прогнозы на одном графике; в легенде — метрика каждого прогноза.

    last — сколько последних точек train показывать (None — весь).
    """
    def ts(idx):
        return idx.to_timestamp() if isinstance(idx, pd.PeriodIndex) else idx

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    shown = y_train if last is None else y_train.iloc[-last:]
    ax.plot(ts(shown.index), shown.to_numpy(), color="black", lw=1, label="train")
    ax.plot(ts(y_test.index), y_test.to_numpy(), color="black", lw=1, ls="--", label="test")
    for name, pred in preds.items():
        pred = pd.Series(np.asarray(pred).ravel(), index=y_test.index)
        ax.plot(ts(pred.index), pred.to_numpy(), lw=1.5, label=f"{name}: {metric_name} {metric(y_test, pred):.2f}")
    ax.legend(loc="upper left", fontsize=9)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax
