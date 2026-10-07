# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: "Python (ts-cu-2026 \xB7 week 6)"
#     language: python
#     name: ts-cu-2026-week6
# ---

# %%
import sys
import os

# Проверяем, запущен ли ноутбук в среде Google Colab
if "google.colab" in sys.modules:
    # !pip install uv -q
    # !wget https://raw.githubusercontent.com/alex-hse-repository/time-series-cu-2026/refs/heads/main/week_6/pyproject.toml -q
    # !uv pip install -r pyproject.toml --system -q
    os.kill(os.getpid(), 9)  # force runtime restart

# %% [markdown]
# # Задача поиска аномалий

# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (23, 9)

import warnings

warnings.filterwarnings("ignore")

# For adtk
import matplotlib.style as mplstyle

if "seaborn-v0_8-whitegrid" in mplstyle.library:
    mplstyle.library["seaborn-whitegrid"] = mplstyle.library["seaborn-v0_8-whitegrid"]

np.random.seed(42)

# %% [markdown]
# # Точечные аномалии

# %% [markdown]
# ## 0. Utils
#
# Тут наборчик вспомогательных функций для оценки метрик и отрисовки графиков

# %%
from etna.datasets import TSDataset
from etna.analysis import plot_anomalies as plot_anomalies_etna
from adtk.visualization import plot
from adtk.metrics import recall, precision, f1_score


# %%
def convert_to_etna(df):
    df_etna = df_etna = pd.melt(
        pd.DataFrame(df).reset_index(),
        id_vars="timestamp",
        value_vars=[f"segment_{i}" for i in range(1, 8)],
        var_name="segment",
        value_name="target",
    )
    return df_etna


def anomaly_dict_to_mask(df, anomaly_dict):
    df_mask = df.copy()
    df_mask[:] = 0
    for segment, indexes in anomaly_dict.items():
        df_mask.loc[indexes, segment] = 1
    return df_mask


def eval_etna(df, anomaly_dict):
    anomalies_mask = anomaly_dict_to_mask(df, anomaly_dict)

    plot_anomalies_etna(ts, anomaly_dict)
    metrics, metrics_agg = eval_metrics(df_anomaly, anomalies_mask, metrics_list=[precision, recall, f1_score])
    print(metrics_agg)
    return metrics, metrics_agg, anomalies_mask


def eval_metrics(y_true, y_pred, metrics_list):
    metrics = {}
    metrics_agg = {}
    for metric in metrics_list:
        name = metric.__name__
        metrics[name] = metric(y_true, y_pred)
        metrics_agg[name] = pd.Series(metric(y_true, y_pred)).mean()

    return metrics, metrics_agg


def plot_anomalies(series, detector):
    try:
        anomalies = detector.fit_detect(series)
    except:
        anomalies = detector.detect(series)

    metrics, metrics_agg = eval_metrics(df_anomaly, anomalies, metrics_list=[precision, recall, f1_score])
    plot(
        series,
        anomaly=anomalies,
        ts_linewidth=1,
        ts_markersize=3,
        anomaly_markersize=5,
        anomaly_color="red",
        anomaly_tag="marker",
    )
    print(metrics_agg)
    return metrics, metrics_agg, anomalies


# %% [markdown]
# ## 1. Генерация данных
#
# На лекции мы обсуждали, что для подбора и настроки алгоритма поиска аномалий под вашу задачу бывает полезно поэксперементировать на синтетических данных -- для них известна разметка, на которой можно замерять качество. На семинаре мы как раз попробуем позапускать наши алгоритмы на такой слоеной синтетике:
#
# - Белый шум + точечные выбросы
# - Белый шум + линейный тренд + точечные выбросы
# - Белый шум + сезонность + точечные выбросы
# - Белый шум + линейный тренд + сезонность + точечные выбросы

# %%
# Количество точек по времени
N = 600

# Масштаб компонент
TREND = 1 / 60
SEASONAL = 3
ANOMALY = 6

# Периоды сезонностей
SEASONALITY_1 = 7
SEASONALITY_2 = 14

timestamps = np.arange(0, N)

noise_component = np.random.normal(size=N)
trend_component = timestamps * TREND
seasonal_component = np.sin(1 / SEASONALITY_1 * 2 * np.pi * timestamps) * SEASONAL
anomaly_component = np.random.choice([-1, 0, 1], p=[0.02, 0.96, 0.02], size=N) * ANOMALY

seasonal_component_2 = np.sin(1 / SEASONALITY_2 * 2 * np.pi * timestamps) * SEASONAL
anomaly_component_2 = (
    (timestamps % SEASONALITY_2 == 0) * np.random.choice([-1, 0, 1], p=[0.03, 0.94, 0.03], size=N) * ANOMALY
)

anomaly_mask = (anomaly_component != 0).astype(int)
anomaly_mask_2 = (anomaly_component_2 != 0).astype(int)


df = pd.DataFrame(
    {
        "timestamp": pd.date_range(start="2000-01-01", periods=N, freq="D"),
        "segment_1": noise_component + anomaly_component,
        "segment_2": noise_component + trend_component + anomaly_component,
        "segment_3": noise_component + seasonal_component + anomaly_component,
        "segment_4": noise_component + trend_component + seasonal_component + anomaly_component,
        "segment_5": noise_component + anomaly_component_2,
        "segment_6": noise_component + trend_component + anomaly_component_2,
        "segment_7": noise_component + trend_component + seasonal_component_2 + anomaly_component_2,
    }
)

df_anomaly = pd.DataFrame(
    {
        "timestamp": pd.date_range(start="2000-01-01", periods=N, freq="D"),
        "segment_1": anomaly_mask,
        "segment_2": anomaly_mask,
        "segment_3": anomaly_mask,
        "segment_4": anomaly_mask,
        "segment_5": anomaly_mask_2,
        "segment_6": anomaly_mask_2,
        "segment_7": anomaly_mask_2,
    }
)

df = df.set_index("timestamp")
df_anomaly = df_anomaly.set_index("timestamp")
df["segment_7"].plot()

# %%
df_anomaly["segment_7"].plot()

# %%
df_etna = convert_to_etna(df)
ts = TSDataset(df=df_etna, freq="D")
ts.plot()

# %%
#df.reset_index().to_csv("data/data.csv", index=False)
#df_anomaly.reset_index().to_csv("data/anomaly.csv", index=False)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## 2. Rule-Based подходы
#
# Начнем с самых простых подходов

# %% [markdown]
# ### 2.1 Global statistics
# https://adtk.readthedocs.io/en/stable/notebooks/demo.html#
#
# Идея: константные пороги на все время
# - ThresholdAD -- нужен свой трешхолд под каждый ряд + фигово работает в случае нестационарности
# - QuantileAD/InterQuartileRangeAD -- адаптивный трешхолд, однако все также плохо для тренда/сезонности

# %%
from adtk.detector import ThresholdAD, QuantileAD, InterQuartileRangeAD

# %%
threshold_ad = ThresholdAD(low=-2, high=2)
metrics_thr, metrics_agg_thr, anomalies_thr = plot_anomalies(df, threshold_ad)

# %%
quantile_ad = QuantileAD(high=0.99, low=0.01)
metrics_qt, metrics_agg_qt, anomalies_qt = plot_anomalies(df, quantile_ad)

# %%
iqr_ad = InterQuartileRangeAD(c=1.5)
metrics_iqr, metrics_agg_iqr, anomalies_iqr = plot_anomalies(df, iqr_ad)

# %% [markdown]
# ### 2.2 Sliding statistics
#
# https://github.com/etna-team/etna/blob/master/examples/204-outliers.ipynb
#
# Идея: пороги, зависящие от времени, строящиеся на статистиках в окне
# - Гораздо лучше отрабатывают на нестационарных данных

# %%
from etna.analysis import get_anomalies_median, get_anomalies_density

# %%
anomaly_dict = get_anomalies_median(ts, window_size=100)
metrics_med_100, metrics_agg_med_100, anomalies_med_100 = eval_etna(df, anomaly_dict)

# %%
anomaly_dict = get_anomalies_median(ts, window_size=20)
metrics_med_20, metrics_agg_med_20, anomalies_med_20 = eval_etna(df, anomaly_dict)

# %%
anomaly_dict = get_anomalies_density(ts, window_size=20, distance_coef=1, n_neighbors=4)
metrics_density, metrics_agg_density, anomalies_density = eval_etna(df, anomaly_dict)

# %% [markdown]
# ### 2.3 Sliding statistics(интерактивная визуализация)
#
# Идея: параметры для предразметки можно подбирать визуально

# %%
from etna.analysis import plot_anomalies_interactive

# %%
segment = "segment_1"
method = get_anomalies_median
params_bounds = {"window_size": (40, 70, 1), "alpha": (0.1, 4, 0.25)}
plot_anomalies_interactive(ts=ts, segment=segment, method=method, params_bounds=params_bounds)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## 3. Forecasting Model
#
# Следующий метод основан на использовании предсказательных моделей
#
# Идея: если модель плохо предсказывает точку(точка выходит за доверительный интервал например), возможно она аномальная. Однако есть и другая альтернатива -- у вас просто плохая модель

# %% [markdown]
# ### 3.1 Запуск метода
#
# Будем искать аномалии с помощью модели Prophet
# - Удалось найти все аномальные точки, при этом FP не так уж и много

# %%
from etna.analysis import get_anomalies_prediction_interval
from etna.models import ProphetModel

# %%
anomaly_dict = get_anomalies_prediction_interval(ts, model=ProphetModel, interval_width=0.95)
metrics_forecast, metrics_agg_forecast, anomalies_forecast = eval_etna(df, anomaly_dict)

# %% [markdown]
# ### 3.2 Offline vs Online 
#
# На лекции мы обсуждали что задачу поиска аномалий можно встретить в 2 постановках:
# 1. Offline -- когда изначально известны все данные, в рамках которых необходимо обнаружить аномалии
# 2. Online -- аномалии необходимо обноруживать в приходящем потоке данных
#
#

# %%
from etna.pipeline import Pipeline
from etna.analysis import plot_forecast

# %%
pipeline = Pipeline(model=ProphetModel(), transforms=[], horizon=72)
pipeline.fit(ts)

# %%
# Offline
forecast = pipeline.predict(ts, prediction_interval=True)
plot_forecast(forecast_ts=forecast, train_ts=ts, prediction_intervals=True)

# %%
# Online
forecast = pipeline.forecast(prediction_interval=True)
plot_forecast(forecast_ts=forecast, train_ts=ts, prediction_intervals=True)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## 4. Сведение к табличкам
#
# Следующая идея это сведение к табличным данным -- там уже придумано куча методов

# %% [markdown]
# ### 4.1 AutoEncoder
#
# Построим простейший автоэнкодер, в качестве признаков для точки будем использовать лаги

# %%
from pyod.models.auto_encoder import AutoEncoder

N_LAGS = 14

# %%
anomalies_autoencoder = df_anomaly.copy()
anomalies_autoencoder_prob = df_anomaly.copy()

for target in df.columns:
    # Построим признаки
    X, y = df[[target]], df_anomaly[target]
    for i in range(1, N_LAGS + 1):
        X[f"lag_{i}"] = X[target].shift(i)
    X = X.dropna()

    # Обучим модель
    clf = AutoEncoder()
    clf.fit(X)

    # Выберем аномальные точки
    anomalies_autoencoder[target] = (
        [
            0 for _ in range(1, N_LAGS + 1)
        ]  # Для первых точечк ряда известны не все признаки, поэтому там скор аномальности мы оценить не можем
        + list(clf.predict(X))
    )
    anomalies_autoencoder_prob[target] = (
        [
            0 for _ in range(1, N_LAGS + 1)
        ]  # Для первых точечк ряда известны не все признаки, поэтому там скор аномальности мы оценить не можем
        + list(clf.predict_proba(X)[:, 1])
    )

# %%
metrics, metrics_agg_autoencoder = eval_metrics(
    df_anomaly, anomalies_autoencoder, metrics_list=[precision, recall, f1_score]
)
plot(
    df,
    anomaly=anomalies_autoencoder,
    ts_linewidth=1,
    ts_markersize=3,
    anomaly_markersize=5,
    anomaly_color="red",
    anomaly_tag="marker",
)
print(metrics_agg_autoencoder)

# %%
anomalies_autoencoder_prob.plot(y="segment_1")

# %%
df.plot(y="segment_1")

# %% [markdown]
# ### 4.2 Isolation Forest 
#
# На семинаре построим базовую версию IF, которая ищет аномалии на 1 признаке -- таргете. В таком случае мы получаем 1d решающие поверхности -- по картинкам видно что этого не достаточно
#
# В качестве ДЗ предлагается попробовать улучшить ситуацию и добавить дургие признаки

# %%
from etna.analysis import get_anomalies_isolation_forest

# %%
anomaly_dict = get_anomalies_isolation_forest(ts)
metrics_if, metrics_agg_if, anomalies_if = eval_etna(df, anomaly_dict)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## 5. Ensembling
#
# Бывает полезно смешать ответы нескольких методов

# %%
from adtk.aggregator import OrAggregator, AndAggregator

# %%
anomalies = [
    anomalies_autoencoder,
    anomalies_if,
    anomalies_forecast,
    anomalies_thr,
    anomalies_qt,
    anomalies_iqr,
    anomalies_med_100,
    anomalies_med_20,
    anomalies_density,
]
metrics = [
    metrics_agg_autoencoder,
    metrics_agg_if,
    metrics_agg_forecast,
    metrics_agg_thr,
    metrics_agg_qt,
    metrics_agg_iqr,
    metrics_agg_med_100,
    metrics_agg_med_20,
    metrics_agg_density,
]
names = ["anomalies_autoencoder", "if", "if_det", "forecast", "thr", "qt", "iqr", "med_100", "med_20", "density"]

# %%
anomalies_thr_or = df.copy()
for target in df.columns:
    anomaly = {name: anomaly[target] for name, anomaly in zip(names, anomalies)}
    anomalies_thr_or[target] = OrAggregator().aggregate(anomaly)

# %%
metrics, metrics_agg_or = eval_metrics(df_anomaly, anomalies_thr_or, metrics_list=[precision, recall, f1_score])
plot(
    df,
    anomaly=anomalies_thr_or,
    ts_linewidth=1,
    ts_markersize=3,
    anomaly_markersize=5,
    anomaly_color="red",
    anomaly_tag="marker",
)
print(metrics_agg_or)

# %%
anomalies_thr_and = df.copy()
for target in df.columns:
    anomaly = {name: anomaly[target] for name, anomaly in zip(names, anomalies)}
    anomalies_thr_and[target] = AndAggregator().aggregate(anomaly)

# %%
metrics, metrics_agg_and = eval_metrics(df_anomaly, anomalies_thr_and, metrics_list=[precision, recall, f1_score])
plot(
    df,
    anomaly=anomalies_thr_and,
    ts_linewidth=1,
    ts_markersize=3,
    anomaly_markersize=5,
    anomaly_color="red",
    anomaly_tag="marker",
)
print(metrics_agg_and)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## 6. All-in-all
#
# Теперь посмотрим на итогововые результаты и выберем лучше методы для наших данных по разным метрикам

# %%
anomalies = [
    anomalies_autoencoder,
    anomalies_if,
    anomalies_forecast,
    anomalies_thr,
    anomalies_qt,
    anomalies_iqr,
    anomalies_med_100,
    anomalies_med_20,
    anomalies_density,
]
metrics = [
    metrics_agg_autoencoder,
    metrics_agg_if,
    metrics_agg_forecast,
    metrics_agg_thr,
    metrics_agg_qt,
    metrics_agg_iqr,
    metrics_agg_med_100,
    metrics_agg_med_20,
    metrics_agg_density,
]
names = ["autoencoder", "if", "forecast", "thr", "qt", "iqr", "med_100", "med_20", "density"]

metrics += [metrics_agg_or, metrics_agg_and]
names += ["or", "and"]

pd.DataFrame(metrics, index=names).sort_values("f1_score", ascending=False)

# %%
pd.DataFrame(metrics, index=names).sort_values("recall", ascending=False)

# %%
pd.DataFrame(metrics, index=names).sort_values("precision", ascending=False)

# %% [markdown]
# # Точки смены поведения
#
# Кратко рассмтотрим варианты для онаружения точек смены поведения

# %% [markdown]
# ## 1. Ruptures
#
# [ruptures](https://centre-borelli.github.io/ruptures-docs/) -- библиотека для обнаружения точек смены поведения, все методы в ней состоят из двух компонент
# 1. Cost function -- она определяет характер ихменения поведения
# 2. Search method -- стратегия поиска точек
#
# Также стратегия поиска точек позволяет искать как конкретное количество(n_bkps), так и задавать некоторый критерий остановки поиска 

# %%
import matplotlib.pyplot as plt
import ruptures as rpt

n_samples, dim, sigma = 1000, 3, 4
n_bkps = 4
signal, bkps = rpt.pw_constant(n_samples, dim, n_bkps, noise_std=sigma)

model = "l2"  # cost function -- "l1", "rbf", "linear", "normal", "ar",...
algo = rpt.Binseg(model=model).fit(signal)  # search method
result = algo.predict(n_bkps=n_bkps)

rpt.display(
    signal,
    bkps,
    result,  # Вертикальные черты
)
plt.show()

# %% [markdown]
# Тут не указываем точное количетсво точек

# %%
result = algo.predict(pen=100)

rpt.display(signal, bkps, result)
plt.show()

# %%
result = algo.predict(pen=1000)

rpt.display(signal, bkps, result)
plt.show()

# %% [markdown]
# ## 2. Скользящие статистики
#
# Основная идея: проходимся по ряду скользящим окошком и для центральной точки подсчитываем discrepancy function 

# %%
from adtk.detector import LevelShiftAD

# %%
df_seg = pd.DataFrame(
    {
        "timestamp": pd.date_range(start="2000-01-01", freq="D", periods=signal.shape[0]),
        "segment_1": signal[:, 0],
        "segment_2": signal[:, 1],
        "segment_3": signal[:, 2],
    }
)
df_seg = df_seg.set_index("timestamp")

# %%
level_shift_ad = LevelShiftAD(c=2.0, side="both", window=30)
anomalies = level_shift_ad.fit_detect(df_seg)

pd.Series.iteritems = pd.Series.items
plot(df_seg, anomaly=anomalies, anomaly_color="red");

# %% [markdown]
# ## 3.3 Prophet
#
# https://facebook.github.io/prophet/docs/trend_changepoints.html#automatic-changepoint-detection-in-prophet
#
# Поиск точек смены поведения работает примерно так:
# 1. Разрабсываем равномерно n_changepoints по первым changepoint_range точкам ряда
# 2. Обучаем модель с априорным распределением Лапласа для коэффициентов для каждой точки (аля Lasso регрессия)
# 3. Точки, для которых не занулились коэффициенты и есть точки смены поведения
#
# Через changepoint_prior_scale можно управлять консервативностью выбора количества значимых точек

# %%
df_prophet = df_seg["segment_1"].reset_index().rename(columns={"timestamp": "ds", "segment_1": "y"})
df_prophet.head()

# %%
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

m = Prophet(n_changepoints=25, changepoint_range=0.8, changepoint_prior_scale=0.5)
forecast = m.fit(df_prophet).predict(df_prophet)

fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)

# %%
m = Prophet(n_changepoints=25, changepoint_range=0.8, changepoint_prior_scale=0.001)
forecast = m.fit(df_prophet).predict(df_prophet)

fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)

# %%
m = Prophet(n_changepoints=25, changepoint_range=0.8, changepoint_prior_scale=100)
forecast = m.fit(df_prophet).predict(df_prophet)

fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)

# %%
