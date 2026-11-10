"""Общая часть ноутбуков про прерывистый спрос (4, 5, 6): данные M5 CA_1, классы ADI/CV², метрики.

Импорт из ноутбука той же папки:

    from utils import load_m5_ca1, to_sf_format, train_test_split, demand_classes, evaluate_by_class
"""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import bias, mae, rmsse

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Пороги Syntetos–Boylan (2005)
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49
CLASSES = ["smooth", "intermittent", "erratic", "lumpy"]

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def load_m5_ca1(calendar_cols=("date",), data_dir=DATA_DIR):
    """M5, магазин CA_1, long-формат: одна строка = (товар, день).

    `calendar_cols` — какие столбцы `calendar.csv` подтянуть к продажам (`date` добавляется всегда).
    """
    data_dir = Path(data_dir)
    sales = pd.read_csv(data_dir / "sales_train_validation.csv.gz")
    calendar = pd.read_csv(data_dir / "calendar.csv")

    sales_ca1 = sales[sales["store_id"] == "CA_1"]
    day_cols = [c for c in sales_ca1.columns if c.startswith("d_")]

    df = sales_ca1.melt(id_vars=ID_COLS, value_vars=day_cols, var_name="d", value_name="sales")
    cal_cols = ["d", *dict.fromkeys(["date", *calendar_cols])]
    df = df.merge(calendar[cal_cols], on="d", how="left")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["id", "date"]).reset_index(drop=True)


def to_sf_format(df):
    """long-формат M5 → формат Nixtla: `unique_id`, `ds`, `y`."""
    return df[["id", "date", "sales"]].rename(columns={"id": "unique_id", "date": "ds", "sales": "y"})


def train_test_split(df, h, id_col="unique_id", time_col="ds"):
    """Последние `h` точек каждого ряда — тест, остальное — обучение."""
    df = df.sort_values([id_col, time_col])
    grouped = df.groupby(id_col, sort=False)
    test = grouped.tail(h)
    train = df.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def demand_classes(df, id_col="unique_id", time_col="ds", target_col="y"):
    """ADI, CV² и класс спроса для каждого ряда (векторизованно, без цикла по рядам).

    ADI — средний интервал между соседними ненулевыми продажами (как `np.diff` индексов ненулей),
    CV² — квадрат коэффициента вариации ненулевых продаж (ddof=0).
    Крайние случаи как в ручной `classify_demand` из ноутбука 4: нет продаж → lumpy,
    одна продажа → ADI = длина ряда, CV² = 0.
    """
    df = df.sort_values([id_col, time_col])
    # Номер точки внутри ряда — чтобы интервалы считались в периодах, а не в днях календаря.
    df = df.assign(_t=df.groupby(id_col, sort=False).cumcount())
    nonzero = df[df[target_col] > 0]
    gap = nonzero.groupby(id_col, sort=False)["_t"].diff()

    stats = pd.DataFrame(
        {
            "adi": gap.groupby(nonzero[id_col], sort=False).mean(),
            "mean": nonzero.groupby(id_col, sort=False)[target_col].mean(),
            "std": nonzero.groupby(id_col, sort=False)[target_col].std(ddof=0),
        }
    )
    stats["cv2"] = (stats["std"] / stats["mean"]) ** 2

    lengths = df.groupby(id_col, sort=False).size()
    out = pd.DataFrame(index=lengths.index)
    out = out.join(stats[["adi", "cv2"]])
    n_nonzero = nonzero.groupby(id_col, sort=False).size().reindex(out.index, fill_value=0)
    single = n_nonzero == 1
    out.loc[single, "adi"] = lengths[single]
    out.loc[single, "cv2"] = 0.0
    out.loc[n_nonzero == 0, ["adi", "cv2"]] = np.inf

    smooth_cv = out["cv2"] < CV2_THRESHOLD
    frequent = out["adi"] < ADI_THRESHOLD
    out["class"] = np.select(
        [frequent & smooth_cv, ~frequent & smooth_cv, frequent & ~smooth_cv],
        ["smooth", "intermittent", "erratic"],
        default="lumpy",
    )
    out.index.name = id_col
    return out


METRICS = {
    "rmsse": partial(rmsse, seasonality=1),
    "mae": mae,
    "bias": bias,
}


def evaluate_by_class(forecasts, test, train, classes, models=None, metrics=METRICS):
    """Метрики каждого ряда → среднее по классам спроса и по всем рядам.

    forecasts — прогнозы в формате Nixtla (`unique_id`, `ds`, столбец на модель),
    test / train — факт на тесте и обучение (для масштаба RMSSE),
    classes — результат `demand_classes` (нужен столбец `class`).

    RMSSE масштабирует RMSE ошибкой наивного прогноза на обучении (seasonality=1), поэтому
    сравнима между рядами разного масштаба. MAE — в штуках. Bias = прогноз − факт
    (>0 — перепрогноз, <0 — недопрогноз), как в utilsforecast.

    Возвращает таблицу: строки (metric, class), столбцы — модели, плюс число рядов `n_series`.
    """
    if models is None:
        models = [c for c in forecasts.columns if c not in ("unique_id", "ds", "cutoff", "index")]
    df = test[["unique_id", "ds", "y"]].merge(
        forecasts[["unique_id", "ds", *models]], on=["unique_id", "ds"], how="left"
    )
    if df[models].isna().any().any():
        raise ValueError("Не для всех точек теста есть прогноз — проверь `ds` в прогнозах.")

    metric_fns = []
    for name, fn in metrics.items():
        base = fn.func if isinstance(fn, partial) else fn
        if base.__name__ != name:
            raise ValueError(f"Ключ {name!r} должен совпадать с именем функции {base.__name__!r}")
        metric_fns.append(fn)

    per_series = evaluate(df, metrics=metric_fns, models=models, train_df=train[["unique_id", "ds", "y"]])
    per_series = per_series.join(classes["class"], on="unique_id")

    by_class = per_series.groupby(["metric", "class"])[models].mean()
    overall = per_series.groupby("metric")[models].mean()
    overall.index = pd.MultiIndex.from_product([overall.index, ["all"]], names=["metric", "class"])

    result = pd.concat([by_class, overall]).sort_index()
    order = [c for c in [*CLASSES, "all"] if c in result.index.get_level_values("class")]
    result = result.reindex(
        pd.MultiIndex.from_product([list(metrics), order], names=["metric", "class"])
    )
    counts = classes["class"].value_counts().reindex(order)
    counts["all"] = len(classes)
    result["n_series"] = result.index.get_level_values("class").map(counts).astype(int)
    return result


def metric_table(result, metric):
    """Срез `evaluate_by_class` по одной метрике: класс × модель."""
    return result.loc[metric].drop(columns="n_series")


def highlight_best(table, by_abs=False, fmt="{:.3f}", color="lightgreen"):
    """Подсветить лучшую модель в каждой строке (меньше — лучше; для bias — меньше по модулю)."""
    key = table.abs() if by_abs else table
    best = key.eq(key.min(axis=1), axis=0)
    return table.style.format(fmt).apply(
        lambda _: np.where(best, f"background-color: {color}", ""), axis=None
    )
