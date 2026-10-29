"""Общая обвязка ноутбуков семинара 9: данные, метрики, графики.

Нужна, чтобы в ноутбуках оставались только вызовы библиотек и пояснения.
Данные читаются по ссылкам на GitHub, как в остальных неделях курса: ноутбук запускается
и локально, и в Colab без копии каталога `data`.
"""

import logging
import warnings

# ETNA и sktime предупреждают об опциональных зависимостях прямо при импорте — глушим до него.
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAW = "https://raw.githubusercontent.com/alex-hse-repository/time-series-cu-2026/refs/heads/main/week_9/data"


def read_data(name: str, **kwargs) -> pd.DataFrame:
    """CSV из каталога `week_9/data` репозитория курса."""
    return pd.read_csv(f"{RAW}/{name}", **kwargs)


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
    raw = read_data("airpassengers.csv", parse_dates=["Month"])
    return pd.Series(
        raw["#Passengers"].to_numpy(dtype=float),
        index=pd.PeriodIndex(raw["Month"], freq="M"),
        name="passengers",
    )


def load_peyton() -> pd.Series:
    """Дневные просмотры страницы Пейтона Мэннинга в Википедии (уже в логарифме); индекс — даты как есть."""
    raw = read_data("example_wp_log_peyton_manning.csv", parse_dates=["ds"])
    return pd.Series(raw["y"].to_numpy(dtype=float), index=pd.DatetimeIndex(raw["ds"]), name="y")


def load_bike() -> pd.DataFrame:
    """Почасовые прокаты велосипедов (Kaggle Bike Sharing Demand, 2011–2012); колонка `datetime` — даты."""
    return read_data("bike_train.csv", parse_dates=["datetime"])


def load_m4_hourly() -> tuple[pd.DataFrame, pd.DataFrame]:
    """M4 Hourly: широкие таблицы (ряд × шаг) для train и test; в test — следующие 48 значений каждого ряда."""
    return read_data("Hourly-train.csv", index_col=0), read_data("Hourly-test.csv", index_col=0)


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
