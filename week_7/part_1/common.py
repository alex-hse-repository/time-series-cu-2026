"""Общая обвязка ноутбуков семинара: данные, бэктест, сравнение вариантов.

Нужна, чтобы в ноутбуках оставались только вызовы ETNA и пояснения.
"""

import logging
import warnings
from pathlib import Path

# ETNA предупреждает об опциональных зависимостях прямо при импорте — глушим до него.
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
from etna.datasets import TSDataset  # noqa: E402
from etna.metrics import MAE, WAPE  # noqa: E402
from etna.pipeline import Pipeline  # noqa: E402
from etna.transforms import DateFlagsTransform, FilterFeaturesTransform, OneHotEncoderTransform  # noqa: E402

ROOT = Path(__file__).resolve().parent
METRICS = [WAPE(), MAE()]


def quiet() -> None:
    """Глушит предупреждения и логи библиотек, чтобы в ноутбуке был виден только результат."""
    warnings.filterwarnings("ignore")
    for name in ("etna", "joblib", "statsmodels", "matplotlib", "py.warnings"):
        logging.getLogger(name).setLevel(logging.ERROR)
    from etna.loggers import tslogger

    tslogger.loggers.clear()


def make_ts(df: pd.DataFrame, freq: str = "D", exog: pd.DataFrame | None = None) -> TSDataset:
    """TSDataset из длинной таблицы с колонками `timestamp, segment, target`.

    exog — длинная таблица `timestamp, segment, <признаки>` с known-future признаками;
    может заходить дальше конца истории (это нужно для прогноза).
    """
    wide = TSDataset.to_dataset(df[["timestamp", "segment", "target"]])
    if exog is None:
        return TSDataset(wide, freq=freq)
    return TSDataset(wide, freq=freq, df_exog=TSDataset.to_dataset(exog), known_future="all")


def load_onion(new_year_days: int = 0) -> TSDataset:
    """Дневная выручка от продаж лука, 2018-01-01 … 2020-12-31, один ряд `onion`.

    new_year_days > 0 добавляет known-future индикаторы `new_year_minus_k` для каждого из
    k = 1…new_year_days дней перед Новым годом (см. ноутбук 1, раздел «Праздники»).
    """
    df = pd.read_csv("https://raw.githubusercontent.com/alex-hse-repository/time-series-cu-2026/refs/heads/main/week_7/data/onion_sales.csv", parse_dates=["day"])
    df = df.rename(columns={"day": "timestamp", "rto": "target"}).assign(segment="onion")
    if not new_year_days:
        return make_ts(df)
    dates = pd.date_range(df["timestamp"].min(), df["timestamp"].max() + pd.Timedelta(days=60), freq="D")
    days_left = (pd.to_datetime((dates.year + 1).astype(str) + "-01-01") - dates).days
    exog = pd.DataFrame({f"new_year_minus_{k}": (days_left == k).astype(int) for k in range(1, new_year_days + 1)})
    return make_ts(df, exog=exog.assign(timestamp=dates, segment="onion"))


def backtest(model, transforms, ts: TSDataset, horizon: int, n_folds: int = 3) -> dict:
    """Бэктест пайплайна; возвращает словарь ETNA (`metrics`, `forecasts`, `fold_info`, `pipelines`)."""
    pipeline = Pipeline(model=model, transforms=transforms, horizon=horizon)
    return pipeline.backtest(ts=ts, metrics=METRICS, n_folds=n_folds, joblib_params=dict(verbose=0))


def score(result: dict) -> pd.Series:
    """Средние по сегментам и фолдам метрики бэктеста."""
    return result["metrics"][[m.name for m in METRICS]].mean()


def compare(results: dict[str, dict]) -> pd.DataFrame:
    """Таблица «вариант → метрики» для нескольких бэктестов одного блока."""
    return pd.DataFrame({name: score(res) for name, res in results.items()}).T.round(4)


def dow(prefix: str = "dow") -> list:
    """День недели как 7 бинарных признаков (one-hot) — так его понимает линейная модель.

    DateFlagsTransform выдаёт категориальную колонку 0…6; OneHotEncoder разворачивает её,
    а FilterFeatures убирает исходную колонку, чтобы она не попала в модель числом.
    """
    raw = f"{prefix}_day_number_in_week"
    return [
        DateFlagsTransform(day_number_in_week=True, day_number_in_month=False, is_weekend=False, out_column=prefix),
        OneHotEncoderTransform(in_column=raw, out_column=f"{prefix}_ohe"),
        FilterFeaturesTransform(exclude=[raw]),
    ]
