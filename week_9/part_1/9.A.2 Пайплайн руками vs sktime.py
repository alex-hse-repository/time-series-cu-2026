# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: "Python (ts-cu-2026 \xB7 week 9)"
#     language: python
#     name: ts-cu-2026-week9
# ---

# %% [markdown]
# # 9.A.2. Пайплайн руками vs sktime (Peyton Manning)
#
# Один и тот же прогноз бустингом построим двумя способами:
#
# 1. **руками** — pandas `shift`/`rolling` + `CatBoostRegressor`;
# 2. **в sktime** — `make_reduction` + `WindowSummarizer` + `DateTimeFeatures`.
#
# Признаки одинаковые, поэтому прогнозы должны почти совпасть. Интересно другое: где в каждом
# варианте легко ошибиться и сколько кода нужно, чтобы не ошибиться.
#
# Данные — дневные просмотры страницы Пейтона Мэннинга в Википедии (уже в логарифме),
# 2007-12 … 2016-01. Горизонт H = 90 дней.

# %%
import common  # первым: глушит предупреждения библиотек при импорте

common.quiet()

import time  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402
from sktime.forecasting.compose import (  # noqa: E402
    ForecastingPipeline,
    TransformedTargetForecaster,
    make_reduction,
)
from sktime.forecasting.naive import NaiveForecaster  # noqa: E402
from sktime.split import temporal_train_test_split  # noqa: E402
from sktime.transformations.series.date import DateTimeFeatures  # noqa: E402
from sktime.transformations.series.impute import Imputer  # noqa: E402
from sktime.transformations.summarize import WindowSummarizer  # noqa: E402

from common import mae, plot_forecast  # noqa: E402

H = 90
LAGS = [90, 120, 150, 180]  # все лаги >= H: на любом шаге горизонта они известны
WINDOWS = [30, 90]  # скользящие средние, сдвинутые на H


def make_catboost():
    return CatBoostRegressor(iterations=800, learning_rate=0.05, depth=4, random_seed=0, verbose=False)


# %% [markdown]
# ## 0. Постановка и пропуски дат

# %%
raw = common.load_peyton()
full_index = pd.date_range(raw.index.min(), raw.index.max(), freq="D")
missing = full_index.difference(raw.index)
print(f"{len(raw)} наблюдений, {len(full_index)} дней в календаре, пропущено дат: {len(missing)}")
print("первые пропуски:", [d.date().isoformat() for d in missing[:5]])

# %% [markdown]
# Пропущенные даты — не `NaN` в столбце, а **отсутствующие строки**. Это опасно для лагов:
# `shift(90)` сдвигает на 90 **строк**, а не на 90 дней. После пропуска «лаг 90» молча
# превращается в «лаг 91, 92, …».

# %%
naive_shift = raw.shift(90)
true_lag = raw.reindex(raw.index - pd.Timedelta(days=90)).set_axis(raw.index)
wrong = (naive_shift - true_lag).abs() > 0  # NaN (нет значения 90 дней назад) сюда не попадает
print(f"строк, где shift(90) взял не то значение: {wrong.sum()} из {len(raw)}")

# %% [markdown]
# Поэтому первый шаг в обоих вариантах — регулярная сетка `asfreq("D")` и заполнение пропусков:
# руками — `interpolate()`, в sktime — трансформация `Imputer`.
#
# Ряд — многосезонный (неделя, год — сезон NFL), с всплесками в плей-офф.

# %%
y = raw.asfreq("D").interpolate()
y_train, y_test = y.iloc[:-H], y.iloc[-H:]

fig, axes = plt.subplots(1, 2, figsize=(14, 3.5), gridspec_kw={"width_ratios": [3, 1]})
axes[0].plot(y.index, y, lw=0.6, color="black")
axes[0].axvline(y_test.index[0], color="red", ls="--")
axes[0].set_title("Весь ряд (красная линия — начало теста)")
week = y.iloc[-H - 56 : -H]
axes[1].plot(week.index, week, marker=".", color="black")
axes[1].set_title("8 недель перед тестом")
axes[1].tick_params(axis="x", rotation=45)
for ax in axes:
    ax.grid(alpha=0.3)

# %% [markdown]
# Бейзлайн — недельный наивный прогноз: «повторить последнюю неделю».

# %%
fh = np.arange(1, H + 1)
naive = NaiveForecaster(strategy="last", sp=7).fit(y_train, fh=fh).predict()
results = {"naive (последняя неделя)": {"MAE": mae(y_test, naive)}}
plot_forecast(y_train, y_test, {"naive": naive}, metric=mae, metric_name="MAE", last=365);

# %% [markdown]
# ## 1. Руками: pandas + CatBoost
#
# Признаки:
# - лаги 90, 120, 150, 180 дней;
# - скользящие средние за 30 и 90 дней, **сдвинутые на H**: `shift(H).rolling(w)`;
# - календарь: день недели, месяц, день года.
#
# Главное правило против утечки (лонгрид §2.4): **до** построения признаков заменить target
# на тесте на `NaN`. Тогда любой признак, который случайно «подсмотрел» тест, окажется `NaN`,
# и это сразу будет видно.

# %%
start = time.perf_counter()

df = pd.DataFrame({"y": y})
df.loc[y_test.index, "y"] = np.nan  # маскируем тест

for lag in LAGS:
    df[f"lag_{lag}"] = df["y"].shift(lag)
for w in WINDOWS:
    df[f"mean_{w}"] = df["y"].shift(H).rolling(w).mean()
df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month
df["day_of_year"] = df.index.dayofyear

features = [c for c in df.columns if c != "y"]
train = df.loc[y_train.index].dropna()  # первые 180 дней без лагов отбрасываем
manual_model = make_catboost().fit(train[features], train["y"])
manual = pd.Series(manual_model.predict(df.loc[y_test.index, features]), index=y_test.index)

time_manual = time.perf_counter() - start
print(f"NaN в признаках теста: {df.loc[y_test.index, features].isna().sum().sum()} — утечки нет")
print(f"MAE = {mae(y_test, manual):.3f}, {time_manual:.1f} с")
results["CatBoost руками"] = {"MAE": mae(y_test, manual)}

# %% [markdown]
# Посмотрим, как маска ловит ошибку. Типичная опечатка — сдвиг на один день меньше горизонта:
# `shift(H - 1)`. На последнем дне теста такой признак берёт значение из первого дня теста,
# то есть из будущего, — и благодаря маске там оказывается `NaN`.

# %%
leaky = df["y"].shift(H - 1).rolling(30).mean()
print(f"NaN в признаке с shift(H - 1) на тесте: {leaky.loc[y_test.index].isna().sum()}")

# %% [markdown]
# Без маски этот признак тихо подсмотрел бы истинное значение, и MAE на тесте был бы лучше
# реального.
#
# ## 2. sktime: тот же пайплайн из готовых блоков
#
# - `Imputer(method="linear")` — заполняет пропуски, как `interpolate()`;
# - `DateTimeFeatures` — календарь из индекса (через пустую `X`, см. ловушку в 9.A.1);
# - `make_reduction(..., transformers=[WindowSummarizer(...)])` — лаги и окна из самого `y`.
#
# Синтаксис `WindowSummarizer`: `"lag": [90, 120, …]` — лаги; для остальных функций окно
# задаётся парой `[lag, window_length]`: `"mean": [[90, 30]]` — среднее за 30 дней,
# заканчивающееся 90 дней назад. Это ровно `shift(90).rolling(30).mean()`.
#
# `pooling="global"` обязателен: `make_reduction` с `transformers` работает только в этом режиме,
# для одного ряда это ничего не меняет.

# %%
y_raw = raw.asfreq("D")  # пропуски остаются NaN — их заполнит Imputer внутри пайплайна
y_raw_train, _ = temporal_train_test_split(y_raw, test_size=H)
X = pd.DataFrame(index=y_raw.index)
X_train, X_test = X.loc[y_raw_train.index], X.loc[y_test.index]

start = time.perf_counter()
forecaster = TransformedTargetForecaster(
    [
        Imputer(method="linear"),
        ForecastingPipeline(
            [
                DateTimeFeatures(manual_selection=["day_of_week", "month_of_year", "day_of_year"]),
                make_reduction(
                    make_catboost(),
                    strategy="recursive",
                    window_length=None,
                    pooling="global",
                    transformers=[
                        WindowSummarizer(lag_feature={"lag": LAGS, "mean": [[H, w] for w in WINDOWS]}, n_jobs=1)
                    ],
                ),
            ]
        ),
    ]
)
forecaster.fit(y_raw_train, X=X_train, fh=fh)
sk = forecaster.predict(X=X_test)
time_sktime = time.perf_counter() - start

print(f"MAE = {mae(y_test, sk):.3f}, {time_sktime:.1f} с")
print(f"корреляция прогнозов руками и sktime: {np.corrcoef(manual, sk)[0, 1]:.3f}")
results["CatBoost sktime"] = {"MAE": mae(y_test, sk)}

# %% [markdown]
# Стратегия `recursive` здесь формальность: модель учится на шаг вперёд и на прогнозе подставляет
# свои прогнозы вместо неизвестных `y`. Но все лаги ≥ 90 = H, поэтому ни на одном шаге горизонта
# собственные прогнозы в признаки не попадают — это тот же «прямой» прогноз одной моделью, что и
# руками.

# %%
plot_forecast(y_train, y_test, {"руками": manual, "sktime": sk, "naive": naive}, metric=mae,
              metric_name="MAE", last=365);

# %% [markdown]
# ## 3. Сравнение реализаций
#
# | | руками | sktime |
# |---|---|---|
# | строк кода пайплайна | ~15 | ~15 |
# | пропуски дат | `asfreq` + `interpolate` — легко забыть | `Imputer` — тоже надо не забыть, но он в пайплайне |
# | утечка через окна | один неверный `shift` — и тест утёк | `[lag, window]` проверяется библиотекой |
# | прогноз на новых данных | всю обработку повторить руками | `update` / `predict` того же объекта |
# | бэктест | писать цикл самому | `evaluate` + сплиттер |
# | прозрачность | видна каждая колонка | `WindowSummarizer` надо уметь читать |
#
# По числу строк выигрыша нет. Выигрыш sktime — в том, что пайплайн — **один объект** с `fit/predict`:
# его можно отдать в бэктест, подбор параметров или прод, и признаки на новых данных построятся так
# же, как на train.

# %%
pd.DataFrame(results).T.round(3)

# %% [markdown]
# ## 4. Честный вывод: бейзлайн обязателен
#
# Обе реализации проигрывают недельному наивному прогнозу. Причина — в постановке: при H = 90 все
# признаки из прошлого ряда старше 90 дней, **свежий уровень модель не видит**. А наивный прогноз
# видит именно его — последнюю неделю.
#
# Попробуем одно изменение, которое даёт модели свежий уровень: **прямая стратегия** (по модели на
# каждый шаг горизонта, подробно — блок 3). Модель для шага `h` обучается на последних 14 днях
# перед прогнозом, поэтому для первых шагов свежие лаги доступны. 90 моделей — дорого, так что
# уменьшим число деревьев до 200.

# %%
start = time.perf_counter()
direct = TransformedTargetForecaster(
    [
        Imputer(method="linear"),
        ForecastingPipeline(
            [
                DateTimeFeatures(manual_selection=["day_of_week", "month_of_year", "day_of_year"]),
                make_reduction(
                    CatBoostRegressor(iterations=200, learning_rate=0.05, depth=4, random_seed=0, verbose=False),
                    strategy="direct",
                    window_length=14,
                ),
            ]
        ),
    ]
)
direct.fit(y_raw_train, X=X_train, fh=fh)
direct_pred = direct.predict(X=X_test)
print(f"direct: MAE = {mae(y_test, direct_pred):.3f}, {time.perf_counter() - start:.1f} с")
results["CatBoost direct, окно 14"] = {"MAE": mae(y_test, direct_pred)}

plot_forecast(y_train, y_test, {"direct": direct_pred, "naive": naive}, metric=mae, metric_name="MAE", last=365)
pd.DataFrame(results).T.round(3)

# %% [markdown]
# Прямая стратегия заметно лучше, но наивный прогноз всё ещё не побеждён. Это нормальный
# результат, а не провал:
#
# 1. **Реализации эквивалентны** — руками и в sktime получаются одни и те же прогнозы (корреляция
#    ≈ 0.99); выбор — вопрос удобства и защиты от ошибок, а не качества.
# 2. **Бейзлайн обязателен**: без него «MAE 0.47» выглядит приемлемо, а на деле модель хуже, чем
#    «повторить последнюю неделю».
# 3. **Длинный горизонт отнимает у модели свежую информацию**. Что с этим делать — выбирать
#    стратегию прогнозирования (блок 3).
