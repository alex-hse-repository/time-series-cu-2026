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
#     display_name: "Python (ts-cu-2026-dev \xB7 week 6)"
#     language: python
#     name: ts-cu-2026-week6-dev
# ---

# %% [markdown]
# # Неделя 6. Методы декомпозиции: классическая, STL, MSTL
#
# Декомпозиция раскладывает ряд на тренд, сезонность и остаток. Разложение не
# единственно: это результат работы сглаживателя, и при другом окне получится другой тренд.
#
# Разбираем три метода; каждый следующий исправляет недостаток предыдущего:
# - **классическая** — скользящее среднее и средние по сезону;
# - **STL** — сезонность может меняться, нет NaN на краях, есть устойчивость к выбросам;
# - **MSTL** — несколько сезонностей сразу.
#
# Для каждого метода: идея → гиперпараметры → разложение реального ряда.

# %%
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsforecast.feature_engineering import mstl_decomposition
from statsforecast.models import MSTL as SFMSTL
from statsmodels.datasets import elec_equip
from statsmodels.graphics.tsaplots import month_plot, plot_acf
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.filters.filtertools import convolution_filter
from statsmodels.tsa.seasonal import MSTL, STL, seasonal_decompose
from statsmodels.tsa.tsatools import freq_to_period

warnings.simplefilter("ignore")
plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["axes.grid"] = True


# %%
def plot_decomps(results, title=None, figsize=(12, 7)):
    """Наложить несколько разложений statsmodels на одну фигуру."""
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    axes[0].plot(next(iter(results.values())).observed, color="black", lw=1)
    axes[0].set_ylabel("observed")
    for name, res in results.items():
        for ax, comp in zip(axes[1:], ["trend", "seasonal", "resid"]):
            ax.plot(getattr(res, comp), lw=1, label=name)
            ax.set_ylabel(comp)
    axes[1].legend(loc="upper left")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


# %% [markdown]
# ## Данные
#
# - **AirPassengers** — месячный, мультипликативная сезонность.
# - **elec_equip** — месячный индекс производства электрооборудования (еврозона), есть резкие провалы.
# - **PJME** — почасовая нагрузка энергосистемы, сезонности 24 / 168 / 8766 часов.
#   В исходнике есть 4 дубля и 30 пропущенных часов (перевод часов), STL не принимает NaN.
# - **Синтетика** — ряд с известными компонентами: можно измерить ошибку разложения.

# %%
air = pd.read_csv("../data/air_passengers.csv", index_col="Month", parse_dates=True)["#Passengers"].asfreq("MS")
log_air = np.log(air)

elec = elec_equip.load().data.iloc[:, 0].asfreq("MS")

pjme = (
    pd.read_csv("../data/PJME_hourly.csv", parse_dates=["Datetime"])
    .groupby("Datetime")["PJME_MW"].mean()  # дубли
    .asfreq("h")
    .interpolate()  # пропуски
    .loc["2015-08-03":"2018-08-02"]  # последние 3 года
)
len(air), len(elec), len(pjme)


# %%
def make_synthetic(n=240, outlier=None, seed=42):
    """Тренд с изломом + сезонность с меняющимися амплитудой и фазой + шум."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 10 + 0.05 * t + 3 * np.tanh((t - 150) / 10)
    drift = np.sin(np.pi * t / n)  # 0 → 1 → 0
    seasonal = (2 + 3 * drift) * np.sin(2 * np.pi * t / 12 + 2 * drift)
    y = trend + seasonal + rng.normal(0, 2, n)
    df = pd.DataFrame(
        {"y": y, "trend": trend, "seasonal": seasonal},
        index=pd.date_range("2000-01", periods=n, freq="MS"),
    )
    if outlier is not None:
        df.loc[outlier, "y"] += 40
    return df


synth = make_synthetic()
synth.plot(subplots=True, figsize=(12, 6), title="Синтетика: ряд и истинные компоненты");

# %% [markdown]
# ## 1. Что такое декомпозиция
#
# - Аддитивная: $y_t = T_t + S_t + R_t$.
# - Мультипликативная: $y_t = T_t \cdot S_t \cdot R_t$; после $\log$ становится аддитивной.
#
# Сезонность — паттерн с **известным** периодом $m$. Его берут из частоты данных
# или по пикам ACF.

# %%
print("period по частоте:", freq_to_period(air.index.freq))
plot_acf(log_air.diff().dropna(), lags=48, title="ACF приростов log(AirPassengers): пики на 12, 24, 36");

# %% [markdown]
# ## 2. Классическая декомпозиция — `seasonal_decompose`
#
# ### Идея
#
# 1. **Тренд** — центрированное скользящее среднее. Для чётного $m$ это $2\times m$-MA
#    с весами $[\tfrac12, 1, \dots, 1, \tfrac12]/m$: окно центрировано, каждый месяц входит с одинаковым весом.
# 2. **Сезонность** — среднее детрендированного ряда по каждой фазе (все январи, все февраля, ...),
#    центрированное к нулю.
# 3. **Остаток** — $y - T - S$.
#
# Повторим шаги средствами statsmodels и сверим с `seasonal_decompose`.

# %%
m = 12
trend = convolution_filter(log_air, np.r_[0.5, np.ones(m - 1), 0.5] / m)
detrended = log_air - trend
profile = detrended.groupby(detrended.index.month).mean()
profile -= profile.mean()
seasonal = pd.Series(profile.loc[log_air.index.month].values, index=log_air.index)
resid = log_air - trend - seasonal

res_classic = seasonal_decompose(log_air, period=m)
print("trend совпадает:   ", np.allclose(trend, res_classic.trend, equal_nan=True))
print("seasonal совпадает:", np.allclose(seasonal, res_classic.seasonal))
print("resid совпадает:   ", np.allclose(resid, res_classic.resid, equal_nan=True))

# %% [markdown]
#
# Сезонная компонента — это всего $m$ чисел, повторённых по кругу:

# %%
profile.plot.bar(title="Сезонный профиль log(AirPassengers)", figsize=(8, 3));

# %%
res_classic.plot();

# %% [markdown]
# ### Гиперпараметры
#
# **`model`.** Аддитивная модель на исходном ряде оставляет в остатке «воронку»:
# амплитуда сезонности растёт вместе с уровнем. Мультипликативная её убирает
# (и эквивалентна аддитивной на $\log y$).

# %%
fig, axes = plt.subplots(3, 1, figsize=(12, 6))
seasonal_decompose(air, model="additive").resid.plot(ax=axes[0], title="additive: resid", marker=".")
seasonal_decompose(log_air, model="additive").resid.plot(ax=axes[1], title="additive(log): resid", marker=".")
seasonal_decompose(air, model="multiplicative").resid.plot(ax=axes[2], title="multiplicative: resid", marker=".")
fig.tight_layout()

# %% [markdown]
# **`period`.** Неверный период — сезонность усредняется по «чужим» фазам и
# вырождается, а настоящий цикл уходит в остаток.

# %%
plot_decomps({f"period={p}": seasonal_decompose(log_air, period=p) for p in [12, 6, 11]});

# %% [markdown]
# **`filt`.** Свои веса фильтра тренда. Окно $2\times 24$ даёт более гладкий тренд,
# но NaN на краях становится вдвое больше.

# %%
filt_24 = np.r_[0.5, np.ones(23), 0.5] / 24
plot_decomps({
    "2x12-MA (default)": res_classic,
    "2x24-MA": seasonal_decompose(log_air, period=12, filt=filt_24),
});

# %% [markdown]
# **`two_sided=False`.** Фильтр только по прошлым значениям: тренд запаздывает
# примерно на $m/2$ шагов, и это запаздывание искажает сезонность.

# %%
plot_decomps({
    "two_sided=True": res_classic,
    "two_sided=False": seasonal_decompose(log_air, period=12, two_sided=False),
});

# %% [markdown]
# **`extrapolate_trend`.** По умолчанию на краях $m/2$ значений NaN. `"freq"`
# достраивает тренд линейной регрессией по ближайшему периоду.

# %%
ax = log_air["1958":].plot(color="black", lw=1, label="log y")
res_classic.trend["1958":].plot(ax=ax, lw=2, label="extrapolate_trend=0")
seasonal_decompose(log_air, period=12, extrapolate_trend="freq").trend["1958":].plot(
    ax=ax, ls="--", label='extrapolate_trend="freq"'
)
ax.legend();

# %% [markdown]
# ### Ограничения
#
# **Сезонность не меняется.** Проверим на синтетике: у истинной сезонности
# амплитуда растёт и фаза сдвигается. `month_plot` рисует каждый месяц по годам —
# у классики это горизонтальные линии, вся эволюция уходит в остаток.

# %%
res_synth = seasonal_decompose(synth["y"], period=12)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
month_plot(synth["seasonal"].to_period("M"), ax=axes[0])
axes[0].set_title("истинная сезонность")
month_plot(res_synth.seasonal.to_period("M"), ax=axes[1])
axes[1].set_title("классическая декомпозиция")
fig.tight_layout()

# %% [markdown]
# **Нет устойчивости к выбросам.** Добавим один выброс +40 в марте 2010.
# Мартовская сезонность сдвигается **во всех годах**, тренд получает горб шириной $m$.

# %%
synth_out = make_synthetic(outlier="2010-03-01")
res_out = seasonal_decompose(synth_out["y"], period=12)

fig, axes = plt.subplots(1, 2, figsize=(12, 3))
(res_out.seasonal - res_synth.seasonal)[:12].set_axis(range(1, 13)).plot.bar(
    ax=axes[0], title="сдвиг сезонного профиля из-за выброса"
)
(res_out.trend - res_synth.trend).plot(ax=axes[1], title="сдвиг тренда из-за выброса")
fig.tight_layout()

# %%
plot_decomps({
    "outlier": res_out,
    "orig": res_synth,
});

# %%
res_out.seasonal[:48].plot(label="orig")
res_synth.seasonal[:48].plot(label="outlier")
plt.legend()

# %% [markdown]
# Итого: NaN на краях, постоянная сезонность, чувствительность к выбросам. STL решает все три проблемы.
#
# ## 3. STL — `STL`
#
# ### Строительный блок: LOESS
#
# LOESS — локальная регрессия: в каждой точке строится взвешенная регрессия
# (степень 0 или 1) по соседям в окне, веса убывают с расстоянием (tricube).
# Главный гиперпараметр — ширина окна: маленькое повторяет шум, большое пересглаживает.

# %%
t = np.arange(len(synth))
ax = synth["y"].plot(color="gray", lw=1, label="y")
for frac in [0.03, 0.15, 0.5]:
    pd.Series(lowess(synth["y"], t, frac=frac, return_sorted=False), index=synth.index).plot(ax=ax, label=f"frac={frac}")
ax.legend(title="доля точек в окне");

# %% [markdown]
# ### Идея
#
# Все гиперпараметры STL — длины окон и степени LOESS для разных компонент.
#
# **Внутренний цикл** (`inner_iter` раз):
# 1. Детрендирование: $y - T$.
# 2. Каждый подряд (все январи, все февраля, ...) сглаживается LOESS с окном `seasonal`.
#    В классике подряд заменялся одним средним, здесь он может медленно меняться.
# 3. Low-pass фильтр (MA $m$ → MA $m$ → MA 3 → LOESS `low_pass`) вычитается из результата:
#    убираем то, что относится к тренду. Получаем $S$.
# 4. Десезонализация $y - S$ и LOESS с окном `trend` → новый $T$.
#
# **Внешний цикл** (`outer_iter` раз, только при `robust=True`): по остаткам
# считаются веса; точки с большим остатком получают вес около 0, и внутренний цикл повторяется.
#
# Сравним сезонность по подрядам с классикой:

# %%
res_stl = STL(log_air, period=12).fit()

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
month_plot(res_classic.seasonal.to_period("M"), ax=axes[0])
axes[0].set_title("классика")
month_plot(res_stl.seasonal.to_period("M"), ax=axes[1])
axes[1].set_title("STL, seasonal=7")
fig.tight_layout()

# %% [markdown]
# ### Гиперпараметры
#
# **`seasonal`** — окно сглаживания подрядов (нечётное, ≥ 7). Маленькое: сезонность
# гибкая, но забирает шум. Большое: сезонность почти постоянная.

# %%
plot_decomps({f"seasonal={w}": STL(synth["y"], period=12, seasonal=w).fit() for w in [7, 35, 999]});

# %% [markdown]
# На синтетике можно измерить ошибку $\hat S$ против истинной. Слишком маленькое
# окно ловит шум, большое не успевает за изменением сезонности и приходит к уровню классики.

# %%
windows = [7, 9, 11, 15, 21, 31, 51, 101, 999]
rmse = [np.sqrt(((STL(synth["y"], period=12, seasonal=w).fit().seasonal - synth["seasonal"]) ** 2).mean()) for w in windows]
rmse_classic = np.sqrt(((res_synth.seasonal - synth["seasonal"]) ** 2).mean())

ax = pd.Series(rmse, index=[str(w) for w in windows]).plot(marker="o", label="STL")
ax.axhline(rmse_classic, color="gray", ls="--", label="классика")
ax.set(xlabel="seasonal", ylabel="RMSE сезонности", title="Ошибка сезонной компоненты на синтетике")
ax.legend();

# %% [markdown]
# При большом окне STL приближается к классике, но точно совпадает с ней только
# при `seasonal_deg=0`. При `seasonal_deg=1` (по умолчанию) LOESS даже в
# бесконечном окне подгоняет в каждом подряде **прямую** — линейно меняющуюся сезонность:

# %%
for deg in [0, 1]:
    s = STL(log_air, period=12, seasonal=999, seasonal_deg=deg).fit().seasonal
    print(f"seasonal_deg={deg}: max |S_STL - S_classic| =", (s - res_classic.seasonal).abs().max().round(4))
print("размах сезонности:", (res_classic.seasonal.max() - res_classic.seasonal.min()).round(4))

# %% [markdown]
# **`trend`** — окно тренда. По умолчанию — наименьшее нечётное
# $\ge 1.5m / (1 - 1.5/\text{seasonal})$; все итоговые значения окон видны в `.config`.
# Чем больше окно, тем жёстче тренд: излом уходит в остаток.

# %%
print(STL(synth["y"], period=12).config)
plot_decomps({f"trend={w}": STL(synth["y"], period=12, trend=w).fit() for w in [23, 61, 151]});

# %% [markdown]
# **`robust`** — внешний цикл с весами. Без него провалы elec_equip размазываются
# по тренду и сезонности, с ним остаются в остатке.

# %%
res_elec = STL(elec, period=12, robust=True).fit()
plot_decomps({
    "robust=False": STL(elec, period=12).fit(),
    "robust=True": res_elec,
});

# %%
res_elec.weights.plot(title="Веса robust STL: около 0 — наблюдение считается выбросом", marker=".");

# %% [markdown]
# **`seasonal_deg`, `trend_deg`** — степень LOESS (0 — локальная константа, 1 — локальная прямая).
# На реальных рядах обычно влияет слабо.

# %%
plot_decomps({
    "deg=1 (default)": res_elec,
    "deg=0": STL(elec, period=12, robust=True, seasonal_deg=0, trend_deg=0).fit(),
});

# %% [markdown]
# **`seasonal_jump`, `trend_jump`, `low_pass_jump`** — LOESS считается в каждой
# `jump`-й точке, между ними линейная интерполяция. Разумный `jump` — 10–20% от
# соответствующего окна. На 3 годах PJME с `period=168` окна тренда и low-pass —
# 321 и 169 точек, а `seasonal=7`, поэтому `seasonal_jump` не трогаем:

# %%
rows, base = [], None
for jump in [1, 5, 25]:
    start = time.perf_counter()
    r = STL(pjme, period=168, trend_jump=jump, low_pass_jump=jump).fit()
    elapsed = time.perf_counter() - start
    base = r if base is None else base
    rows.append({
        "jump": jump,
        "время, с": elapsed,
        "RMSE(ΔS) / std(S)": np.sqrt(((r.seasonal - base.seasonal) ** 2).mean()) / base.seasonal.std(),
        "RMSE(ΔT) / std(T)": np.sqrt(((r.trend - base.trend) ** 2).mean()) / base.trend.std(),
    })
pd.DataFrame(rows).set_index("jump").round(4)

# %% [markdown]
# Остальное:
# - `low_pass` — окно low-pass фильтра, по умолчанию наименьшее нечётное $> m$; трогать почти не нужно.
# - `inner_iter`, `outer_iter` — аргументы `.fit()`; по умолчанию 5/0 без `robust` и 2/15 с ним.
#
# ### Разложение реальных рядов
#
# AirPassengers: STL на $\log y$ против классики — нет NaN на краях, сезонность меняется.

# %%
plot_decomps({"classic": res_classic, "STL": res_stl});

# %%
res_elec.plot();

# %% [markdown]
# ## 4. MSTL — `MSTL`
#
# ### Зачем
#
# У почасовой нагрузки есть дневной (24) и недельный (168) циклы. STL берёт один
# период. С `period=24` недельный цикл попадает в тренд:

# %%
weeks = slice("2018-06-04", "2018-06-24")
ax = pjme[weeks].plot(color="gray", lw=1, label="y")
STL(pjme, period=24).fit().trend[weeks].plot(ax=ax, label="тренд STL(period=24)")
ax.legend();

# %% [markdown]
# С `period=168` оба цикла сливаются в одну компоненту, и каждый из 168 часов
# недели оценивается по одной точке в неделю — данных на фазу в 7 раз меньше.
#
# ### Идея
#
# 1. Опционально Box-Cox (`lmbda`).
# 2. Периоды сортируются по возрастанию; STL с периодом $m_1$ → $S^{(1)}$, вычитаем;
#    STL с $m_2$ на остатке → $S^{(2)}$ и т.д.
# 3. `iterate` раз: каждую $S^{(i)}$ возвращаем в ряд и переоцениваем STL с периодом $m_i$.
# 4. Тренд — из последнего STL, остаток $= y - T - \sum_i S^{(i)}$.

# %%
res_mstl = MSTL(pjme, periods=[24, 168]).fit()
res_mstl.plot();

# %% [markdown]
# ### Гиперпараметры
#
# **`periods`.** На 3 годах есть ещё годовая сезонность (8766 ч). Без неё лето/зима
# сидят в тренде. Нужно не меньше двух полных периодов каждой длины.
#
# STL с периодом 8766 медленный: окна тренда и low-pass — десятки тысяч точек
# (около 2 минут). `trend_jump` и `low_pass_jump` через `stl_kwargs` сокращают
# время до нескольких секунд; компоненты при этом меняются на 1–2% от своего std.

# %%
start = time.perf_counter()
res_mstl_y = MSTL(
    pjme, periods=[24, 168, 8766], stl_kwargs={"trend_jump": 20, "low_pass_jump": 20}
).fit()
print(f"{time.perf_counter() - start:.1f} с")

ax = pjme.resample("D").mean().plot(color="gray", lw=1, label="y (среднее за день)")
res_mstl.trend.plot(ax=ax, label="тренд, periods=[24, 168]")
res_mstl_y.trend.plot(ax=ax, label="тренд, periods=[24, 168, 8766]")
ax.legend();

# %% [markdown]
# Лето и зима ушли из тренда. Но за 3 года у каждого часа года всего 3 наблюдения,
# поэтому годовая компонента забирает и погодные колебания (см. разложение ниже).
#
# **`windows`** — окна `seasonal` для каждого периода. По умолчанию $7 + 4k$ (11, 15):
# зимой у дневного профиля два пика (утро и вечер), летом — один днём (кондиционеры).
# Большие окна делают сезонность почти постоянной, и профили сливаются.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
for ax, w in zip(axes, [None, [2001, 2001]]):
    s24 = MSTL(pjme, periods=[24, 168], windows=w).fit().seasonal["seasonal_24"]
    for month, name in [(1, "январь"), (7, "июль")]:
        s = s24[s24.index.month == month]
        s.groupby(s.index.hour).mean().plot(ax=ax, marker=".", label=name)
    ax.set(title=f"windows={w or 'default'}", xlabel="час")
    ax.legend()
fig.tight_layout()

# %% [markdown]
# **`iterate`** — число уточнений. Изменения быстро затухают, по умолчанию 2.

# %%
ref = MSTL(pjme, periods=[24, 168], iterate=5).fit().seasonal
pd.DataFrame({
    it: np.sqrt(((MSTL(pjme, periods=[24, 168], iterate=it).fit().seasonal - ref) ** 2).mean())
    for it in [1, 2, 3]
}).T.rename_axis("iterate").round(1)

# %% [markdown]
# **`lmbda`** — Box-Cox перед разложением (`"auto"` подбирает $\lambda$).
# Без него амплитуда дневного цикла растёт вместе с уровнем нагрузки —
# мультипликативный эффект в аддитивной модели.

# %%
rows = {}
for lmbda in [None, "auto"]:
    mstl = MSTL(pjme, periods=[24, 168], lmbda=lmbda)
    r = mstl.fit()
    amp = r.seasonal["seasonal_24"].resample("D").agg(lambda x: x.max() - x.min())
    rows[str(lmbda)] = {
        "λ": getattr(mstl, "est_lmbda", None),
        "corr(амплитуда за день, тренд)": np.corrcoef(amp, r.trend.resample("D").mean())[0, 1],
    }
pd.DataFrame(rows).T

# %% [markdown]
# **`stl_kwargs`** — пробрасываются в каждый STL: `trend`, `robust`, `seasonal_deg`, `*_jump`
# и т.д. Влияют так же, как в разделе 3.
#
# ### Разложение
#
# Полное разложение с годовой сезонностью:

# %%
res_mstl_y.plot()
plt.gcf().set_size_inches(12, 10);

# %% [markdown]
# Одна неделя крупным планом:

# %%
week = slice("2018-01-08", "2018-01-14")
pd.concat([pjme.rename("y"), res_mstl_y.trend, res_mstl_y.seasonal[["seasonal_24", "seasonal_168"]]], axis=1)[week].plot(
    subplots=True, figsize=(12, 7)
);

# %% [markdown]
# Дневная сезонность меняется в течение года: heatmap «час × день» за последний год.
# Летом — пик днём, зимой — утренний и вечерний.

# %%
s24 = res_mstl_y.seasonal["seasonal_24"]["2017-08-01":]
heat = s24.groupby([s24.index.hour, s24.index.date]).first().unstack()

fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", origin="lower")
xt = [i for i, d in enumerate(heat.columns) if d.day == 1]
ax.set(xticks=xt, xticklabels=[heat.columns[i].strftime("%Y-%m") for i in xt], xlabel="дата", ylabel="час")
ax.grid(False)
fig.colorbar(im, label="seasonal_24, МВт");

# %% [markdown]
# Недельный профиль: будни против выходных.

# %%
s168 = res_mstl_y.seasonal["seasonal_168"]
weekly = s168.groupby([s168.index.dayofweek, s168.index.hour]).mean()
ax = weekly.reset_index(drop=True).plot(title="seasonal_168, средний профиль недели")
ax.set_xticks(range(0, 168, 24), ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]);

# %% [markdown]
# ### MSTL в statsforecast
#
# `statsforecast.models.MSTL` — то же разложение. Отличия:
# - в STL по умолчанию передаётся `seasonal_deg=0` (как в R `forecast::mstl`);
# - нет Box-Cox;
# - есть `trend_forecaster` — модель для прогноза тренда (часть 2).
#
# С одинаковыми параметрами результаты совпадают:

# %%
sf_comp = SFMSTL(season_length=[24, 168]).fit(pjme.values).model_
sm_comp = MSTL(pjme, periods=[24, 168], stl_kwargs={"seasonal_deg": 0}).fit()
print("trend:      ", np.abs(sf_comp["trend"].values - sm_comp.trend.values).max())
print("seasonal24: ", np.abs(sf_comp["seasonal24"].values - sm_comp.seasonal["seasonal_24"].values).max())
sf_comp.head()

# %% [markdown]
# В формате statsforecast (`unique_id, ds, y`) разложение даёт `mstl_decomposition`:
# компоненты на истории и их продолжение на горизонт `h` (пригодится как признаки во 2-й части).

# %%
df = pjme.rename("y").rename_axis("ds").reset_index().assign(unique_id="PJME")
train_df, future_df = mstl_decomposition(df, SFMSTL(season_length=[24, 168]), freq="h", h=24)
train_df.tail(3)

# %%
future_df.head(3)

# %% [markdown]
# ## 5. Итоги
#
# | | Классическая | STL | MSTL |
# |---|---|---|---|
# | Тренд | $2\times m$-MA | LOESS (`trend`) | LOESS из последнего STL |
# | Сезонность | среднее по фазе, постоянная | LOESS по подрядам, меняется | несколько STL по очереди |
# | Края ряда | NaN | есть | есть |
# | Выбросы | сдвигают сезонность во всех периодах | `robust=True` | `stl_kwargs={"robust": True}` |
# | Мультипликативность | `model="multiplicative"` | через $\log$ / Box-Cox | `lmbda` |
# | Несколько сезонностей | нет | нет | да |
# | Главные гиперпараметры | `period`, `model` | `seasonal`, `trend`, `robust` | `periods`, `windows`, `lmbda` |
#
# **Проверка разложения:** в ACF остатка не должно быть пиков на лагах $m, 2m, \dots$.
# Если пик есть — сезонность не извлечена. Пример: MSTL только с периодом 24 оставляет в остатке недельный цикл.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
plot_acf(MSTL(pjme, periods=[24]).fit().resid, lags=400, ax=axes[0], title="остаток, periods=[24]", markersize=2)
plot_acf(res_mstl.resid, lags=400, ax=axes[1], title="остаток, periods=[24, 168]", markersize=2)
fig.tight_layout()

# %% [markdown]
# Пики на 168 и 336 исчезли. Оставшаяся плавная автокорреляция — погода:
# декомпозиция не обязана делать остаток белым шумом.
