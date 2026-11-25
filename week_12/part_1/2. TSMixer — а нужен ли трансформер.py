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
#     display_name: "Python (ts-cu-2026 \xB7 week 12)"
#     language: python
#     name: ts-cu-2026-week12
# ---

# %%
import os
import sys

# Проверяем, запущен ли ноутбук в среде Google Colab
if "google.colab" in sys.modules:
    # !pip install uv -q
    # !wget https://raw.githubusercontent.com/alex-hse-repository/time-series-cu-2026/refs/heads/main/week_12/pyproject.toml -q
    # !uv pip install -r pyproject.toml --system -q
    os.kill(os.getpid(), 9)  # force runtime restart

# %% [markdown]
# # 12.2. TSMixer: а нужен ли трансформер
#
# Патчинг (PatchTST) вернул трансформер в игру, но выигрыш дало не внимание, а то, что подаётся на вход.
# Сколько нелинейности нужно сверх одной линейной матрицы и нужно ли смешивать каналы?
#
# TSMixer (Chen et al., 2023) отвечает «почти никакой». В модели нет ни внимания, ни свёрток — только
# полносвязные слои, которые по очереди перемешивают окно по времени и по каналам. Тот же приём, что
# в MLP-Mixer для изображений: два направления смешивания вместо придуманного механизма связей.
#
# Разбираем:
# - окно как матрица `L × C` — впервые панель подаётся целиком;
# - temporal и feature mixing, и почему без них TSMixer становится линейной моделью;
# - опыт: испортить шесть каналов и посмотреть, сдвинется ли прогноз седьмого;
# - `TSMixerx`: ковариаты как дополнительные каналы.

# %%
import logging
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasetsforecast.long_horizon import LongHorizon
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from neuralforecast import NeuralForecast
from neuralforecast.models import DLinear, PatchTST, TSMixer, TSMixerx
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mape

warnings.simplefilter("ignore")
for name in ["pytorch_lightning", "lightning", "lightning.pytorch", "lightning_fabric"]:
    logging.getLogger(name).setLevel(logging.ERROR)
plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["axes.grid"] = True

SEED = 1
ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
TRAINER_KW = dict(accelerator=ACCELERATOR, enable_progress_bar=True, enable_model_summary=False, logger=False)


# %%
def score(cv):
    """MAE и MAPE каждой модели, усреднённые по окнам кросс-валидации."""
    models = [c for c in cv.columns if c not in ("unique_id", "ds", "cutoff", "y")]
    res = evaluate(cv, metrics=[mae, mape], models=models)
    return res.groupby("metric")[models].mean().T.round({"mae": 3, "mape": 4}).sort_values("mae")


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def box(ax, x, y, w, h, text, fc="#eef3fb", fs=9):
    """Прямоугольник со скруглёнными углами и подписью — для схем архитектур."""
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#3b5bdb", lw=1.2)
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.35)


def arrow(ax, xy_from, xy_to, color="#495057"):
    ax.add_patch(
        FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>", mutation_scale=11, lw=1.1, color=color, shrinkA=0, shrinkB=0)
    )


# %% [markdown]
# ## Данные и постановка
#
# ETTm2 — семь рядов электрического трансформатора (температура масла `OT` и шесть нагрузок), шаг 15 минут.
# Горизонт `h=96` (сутки), `input_size=336` (3.5 суток), три окна кросс-валидации и фиксированный бюджет
# обучения — `100` шагов для линейных моделей и `300` для нейросетевых.

# %%
Y_df, _, _ = LongHorizon.load(directory="../data", group="ETTm2")
Y_df["ds"] = pd.to_datetime(Y_df["ds"])

H, L, N_WINDOWS, LR = 96, 336, 3, 1e-3
LINEAR_STEPS, NEURAL_STEPS = 100, 300
N_SERIES = Y_df["unique_id"].nunique()
DROP = 0.1  # библиотечное значение по умолчанию 0.9 на таком бюджете не даёт модели ничего выучить

common = dict(h=H, input_size=L, learning_rate=LR, random_seed=SEED, **TRAINER_KW)
linear_kw = {**common, "max_steps": LINEAR_STEPS}
neural_kw = {**common, "max_steps": NEURAL_STEPS}

# %% [markdown]
# ## 1. Окно как матрица
#
# До сих пор каждый ряд жил своей жизнью: модель получала вектор длины `L` и не знала, что рядом есть ещё
# шесть векторов. TSMixer получает их сразу — матрицей `L × C`, где строки это моменты времени, а столбцы
# это каналы.

# %%
tail = Y_df[Y_df["ds"] >= Y_df["ds"].max() - pd.Timedelta(minutes=15 * (L - 1))]
matrix = tail.pivot(index="ds", columns="unique_id", values="y")
print("матрица окна:", matrix.shape)

# %%
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.imshow(matrix.T.to_numpy(), aspect="auto", cmap="viridis")
ax.set(yticks=range(N_SERIES), yticklabels=matrix.columns, xlabel="шаг внутри окна (15 минут)", ylabel="канал")
ax.set_title("Последние 3.5 суток: окно как матрица");
ax.grid(False)

# %% [markdown]
# Каждая строка этой картинки раньше была отдельным рядом, а теперь стала строкой общей матрицы. Вся разница
# между моделями дальше — в том, разрешено ли им смотреть поперёк строк.

# %% [markdown]
# ## 2. Устройство TSMixer
#
# Блок состоит из двух MLP, каждый перемешивает свою ось:
#
# - **temporal mixing** — линейный слой `L → L` внутри каждого канала: модель смотрит вдоль времени;
# - **feature mixing** — MLP `C → ff_dim → C` внутри каждого момента: модель смешивает каналы.
#
# Формально, для входа $X \in \mathbb{R}^{L \times C}$:
#
# $$U = X + \mathrm{ReLU}\big(W_t\,\mathrm{BN}(X) + b_t\big), \qquad
#   V = U + W_2\,\mathrm{ReLU}\big(W_1\,\mathrm{BN}(U) + b_1\big) + b_2 .$$
#
# $W_t$ действует вдоль времени (по столбцам), $W_1, W_2$ — вдоль каналов (по строкам); нормировка стоит перед
# каждой половиной, dropout опущен. Обе операции не меняют размерность, поэтому блоки можно ставить друг
# на друга. После `n_block` блоков стоит линейная проекция `L → H`, общая для всех каналов, а вход и выход
# обёрнуты в RevIN — нормировку каждого окна (`revin=True` по умолчанию).

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for ax in axes:
    ax.imshow(matrix.T.to_numpy(), aspect="auto", cmap="viridis")
    ax.set(yticks=range(N_SERIES), yticklabels=matrix.columns, xlabel="шаг внутри окна", ylabel="канал")
    ax.grid(False)

axes[0].add_patch(FancyArrowPatch((30, 3), (300, 3), arrowstyle="<|-|>", mutation_scale=14, lw=2.2, color="#ff8787"))
axes[0].set_title("temporal mixing: вдоль времени, внутри канала")

axes[1].add_patch(FancyArrowPatch((168, 0.4), (168, 6.6), arrowstyle="<|-|>", mutation_scale=14, lw=2.2, color="#ff8787"))
axes[1].set_title("feature mixing: поперёк каналов, в каждом моменте")

plt.tight_layout();

# %% [markdown]
# Посмотрим на размерности в собранной модели.

# %%
demo = TSMixer(h=H, input_size=L, n_series=N_SERIES, dropout=DROP, **TRAINER_KW)

for i, layer in enumerate(demo.mixing_layers, 1):
    time_shape = tuple(layer.temporal_mixer.temporal_lin.weight.shape)
    chan_in = tuple(layer.feature_mixer.feature_lin_1.weight.shape)
    chan_out = tuple(layer.feature_mixer.feature_lin_2.weight.shape)
    print(f"блок {i}: время {time_shape[0]}→{time_shape[1]}, каналы {chan_in[1]}→{chan_in[0]}→{chan_out[0]}")

print("выходная проекция:", tuple(demo.out.weight.shape), "= L → H, общая для всех каналов")
print("параметров:", n_params(demo))

# %% [markdown]
# Внутри блока размерности не меняются: временная матрица `336 × 336` живёт внутри каждого канала отдельно,
# канальная `7 → 64 → 7` — внутри каждого из 336 моментов. И только в конце стоит общая проекция в горизонт.
#
# Отсюда прямое следствие: **если убрать перемешивающие блоки, останется линейная модель** — одна матрица
# `L → H` на нормированном окне, применённая к каждому каналу отдельно (по сути NLinear). DLinear — та же
# идея с разложением на тренд и остаток. TSMixer — это линейная модель плюс два слоя нелинейного
# перемешивания, и вопрос в том, сколько эти слои добавляют.

# %% [markdown]
# ## 3. Модели сравнения
#
# DLinear — линейная планка, почти вырожденный TSMixer. PatchTST — канально-независимый трансформер,
# точка отсчёта «как выглядит модель без смешивания каналов».

# %%
t0 = time.time()
nf_dl = NeuralForecast(models=[DLinear(alias="DLinear", **linear_kw)], freq="15min")
cv_dl = nf_dl.cross_validation(df=Y_df, n_windows=N_WINDOWS, step_size=H)
time_dl = time.time() - t0
score(cv_dl)

# %%
t0 = time.time()
nf_pt = NeuralForecast(models=[PatchTST(alias="PatchTST", **neural_kw)], freq="15min")
cv_pt = nf_pt.cross_validation(df=Y_df, n_windows=N_WINDOWS, step_size=H)
time_pt = time.time() - t0
score(cv_pt)

# %% [markdown]
# ## 4. TSMixer
#
# Та же постановка и те же окна. Отличие одно: `n_series=7`, модель получает матрицу `L × C` целиком.

# %%
t0 = time.time()
nf_tm = NeuralForecast(models=[TSMixer(alias="TSMixer", n_series=N_SERIES, dropout=DROP, **neural_kw)], freq="15min")
cv_tm = nf_tm.cross_validation(df=Y_df, n_windows=N_WINDOWS, step_size=H)
time_tm = time.time() - t0
score(cv_tm)

# %%
summary = pd.concat([score(cv_dl), score(cv_pt), score(cv_tm)]).sort_values("mae")
summary["параметров"] = summary.index.map(
    {"DLinear": n_params(nf_dl.models[0]), "PatchTST": n_params(nf_pt.models[0]), "TSMixer": n_params(nf_tm.models[0])}
)
summary["обучение, с"] = summary.index.map({"DLinear": time_dl, "PatchTST": time_pt, "TSMixer": time_tm})
summary.round(2)

# %% [markdown]
# Порядок такой: DLinear (0.188) впереди, TSMixer (0.199) второй, PatchTST (0.206) третий. Два слоя нелинейного
# перемешивания и в четыре с лишним раза больше параметров, чем у DLinear, — а разница в сотую, около 6%.
# Канально-независимый PatchTST при втрое большем числе параметров ещё чуть хуже. Значимы ли эти разницы,
# по одному сиду не сказать: нужны три сида.
#
# Время обучения в лоб сравнивать нельзя: у TSMixer по умолчанию `windows_batch_size=32` многомерных окон
# по 7 каналов, то есть 224 отрезка ряда за шаг, у остальных — 1024 отрезка. За шаг он видит впятеро меньше
# данных, поэтому быстрее PatchTST он отчасти просто потому, что делает меньше работы.

# %% [markdown]
# ## 5. Смешивает ли модель каналы
#
# Свойство надо измерить, а не принять на веру. Испортим историю шести рядов, оставив `OT` нетронутым,
# и посмотрим, сдвинется ли прогноз для `OT`.
#
# У канально-независимой модели сдвига быть не может: у неё нет общего веса между каналами. У TSMixer он
# обязан быть — feature mixing смотрит поперёк столбцов на каждом шаге.

# %%
cutoff = cv_tm["cutoff"].max()
history = Y_df[Y_df["ds"] <= cutoff]

ruined = history.copy()
other = ruined["unique_id"] != "OT"
ruined.loc[other, "y"] = np.random.default_rng(0).permutation(ruined.loc[other, "y"].to_numpy())


def shift(nf, alias):
    """Насколько сдвигается прогноз OT, если историю остальных шести рядов перемешать."""
    clean = nf.predict(df=history).query("unique_id == 'OT'")[["ds", alias]]
    bad = nf.predict(df=ruined).query("unique_id == 'OT'")[["ds", alias]]
    both = clean.merge(bad, on="ds", suffixes=("_чисто", "_испорчено"))
    return (both[f"{alias}_чисто"] - both[f"{alias}_испорчено"]).abs().max()


pd.Series(
    {"PatchTST, каналы независимы": shift(nf_pt, "PatchTST"), "TSMixer, каналы смешаны": shift(nf_tm, "TSMixer")},
    name="макс. сдвиг прогноза OT",
).to_frame().round(3)

# %% [markdown]
# Ноль у PatchTST и ненулевой сдвиг у TSMixer. Свойство измерено: одна модель не видит другие каналы, вторая
# видит и реагирует.
#
# Масштаб реакции стоит сравнить с ошибкой. Максимальный по горизонту сдвиг — 0.039, а средняя ошибка TSMixer —
# около 0.2: даже в худшей точке испорченные каналы двигают прогноз на пятую часть типичной ошибки, хотя
# их история перемешана полностью. Смешивание разрешено архитектурой, но опирается модель на чужие каналы
# слабо.
#
# Семь рядов ETTm2 — датчики одного объекта, измеряющие разные величины. Предсказательная связь между ними
# слабая. В статье на семи наборах средняя разница между полным TSMixer и версией без смешивания каналов
# не достигает процента: смешивание окупается там, где каналы связаны по смыслу.

# %% [markdown]
# ## 6. TSMixerx: ковариаты как ещё несколько каналов
#
# В TSMixer экзогенных признаков нет. `TSMixerx` добавляет их не отдельной ветвью, а тем же способом, каким
# устроено всё остальное: известные в будущем признаки становятся дополнительными каналами той же матрицы.
# Feature mixing после этого смешивает `7 + число признаков` каналов.
#
# Проверим на календарных признаках, которые для нагрузки трансформатора должны быть полезны: час суток
# и день недели.

# %%
exog_df = Y_df.assign(hour=Y_df["ds"].dt.hour, weekday=Y_df["ds"].dt.weekday)

t0 = time.time()
nf_tmx = NeuralForecast(
    models=[
        TSMixerx(
            alias="TSMixerx + календарь",
            n_series=N_SERIES,
            futr_exog_list=["hour", "weekday"],
            dropout=DROP,
            **neural_kw,
        )
    ],
    freq="15min",
)
cv_tmx = nf_tmx.cross_validation(df=exog_df, n_windows=N_WINDOWS, step_size=H)
time_tmx = time.time() - t0

pd.concat([score(cv_tm), score(cv_tmx)]).round(3)

# %% [markdown]
# Признаки не помогли: MAE не изменился (0.199 → 0.198), а MAPE даже вырос (0.256 → 0.276). Читать это надо
# аккуратно, потому что неверный вывод здесь возможен в обе стороны.
#
# Неверно заключить, что календарь не нужен вообще. Объяснений два. Первое — данные: окно в 3.5 суток уже
# содержит несколько суточных циклов, и час суток модель восстанавливает по самой истории. Второе — устройство
# модели: `TSMixerx` не отбирает признаки, а добавляет их каналами, и два лишних канала удорожают задачу.
#
# Отсюда вывод про архитектуру. У `TSMixerx` признак — ещё один канал той же матрицы: ни отдельной сети под
# каждый тип входа, ни отбора переменных. Пока признаков десятки, этого хватает; когда их сотни и они
# разнородны, нужен отбор — а это устройство другого класса моделей.

# %% [markdown]
# ## 7. Где смешивание каналов окупается
#
# На ETT его почти нет, и это свойство задачи, а не случайность набора: каналы измеряют разные величины,
# и модель прекрасно живёт, обрабатывая их по отдельности.
#
# Другое дело розница. В M5 нужно предсказать продажи тысяч товаров с общей структурой спроса — акции, праздники,
# сезонность категории, — и информация о соседних товарах полезна напрямую. Числа там расходятся резко:
# канально-независимые модели держатся около 1.0 WRMSSE, то есть на уровне наивного прогноза, а TSMixer
# показывает 0.737.
#
# Оговорка: это числа из чужих таблиц, победитель M5 показал 0.520, и напрямую сравнивать их с нашими нельзя —
# разные наборы, метрики и бюджеты настройки. Вывод не «TSMixer лучше», а «смешивание каналов стоит того,
# когда каналы связаны по смыслу».

# %% [markdown]
# ### Для самостоятельного изучения
#
# - **TMix-Only.** Убрать feature mixing и оставить только временное: в статье на семи наборах и четырёх
#   горизонтах средняя разница с полным TSMixer не достигает процента. Отдельной ручки в neuralforecast нет,
#   но приблизиться можно, обучив TSMixer с `n_series=1` на каждом ряде отдельно: при одном канале feature
#   mixing смешивать нечего, он становится поточечной функцией одного значения.
# - **`n_block` и `ff_dim`.** Два блока и 64 — значения по умолчанию в neuralforecast. Сколько блоков нужно на нашем ряде,
#   решается сеткой, а не чтением.
# - **`dropout`.** Мы взяли 0.1 вместо 0.9. Разница покажет, насколько результат зависит от регуляризации.
# - **Три сида.** Разница между TSMixer и DLinear может оказаться меньше разброса по инициализациям.
# - **`TimeMixer` и `TiDE`.** TimeMixer тоже `MULTIVARIATE = True`, но смешивает не каналы, а масштабы
#   сезонности. TiDE — encoder-decoder без внимания, где декодер отвечает за реакцию на ковариаты.

# %% [markdown]
# ## 8. Итоги
#
# 1. **Точечная токенизация — главная поломка трансформера на рядах.** Но лечит она не столько точность,
#    сколько стоимость: патч уменьшает карту внимания на порядок, а качество на гладком ETTm2 не меняет.
# 2. **Ни внимание, ни смешивание каналов не дают выигрыша сами по себе.** TSMixer обошёл PatchTST на сотые
#    (0.199 против 0.206), а DLinear впереди обоих (0.188). Смешивание каналов модель использует слабо,
#    календарь как дополнительные каналы не помог. Архитектура решает, что модель физически может делать;
#    данные решают, что она из этого возьмёт.
# 3. **MLP без внимания конкурентоспособен:** он обходит трансформер, когда данных мало и нелинейности
#    в задаче почти нет.
#
# Главное: TSMixer и линейная модель — не два разных подхода, а одна модель и она же с двумя дополнительными
# слоями.
#
# Дальше — другой класс задач: не «один ряд, известное прошлое, точечный прогноз», а разнотипные признаки
# разной доступности и нужный интервал. Там на первый план выходит система подготовки входов, и её удобно
# разбирать на TFT.

# %%
