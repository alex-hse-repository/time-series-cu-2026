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
#     display_name: "Python (ts-cu-2026 \xB7 week 10)"
#     language: python
#     name: ts-cu-2026-week10
# ---

# %% [markdown]
# # 10.C.3. Практика на M5 CA_1: иерархия с прерывистым низом
#
# **Сквозной вопрос блока:** прогнозы уровней не складываются — как сделать их согласованными и при этом точнее?
#
# В этом ноутбуке:
# - иерархия магазина CA_1: total → категория (3) → отдел (7) → товар (3049); агрегаты ровные, низ прерывистый;
# - разные базовые модели по уровням: ETS для агрегатов, Croston SBA / TSB для товаров;
# - BU vs TD vs MiddleOut vs MinT shrink: какой уровень делать базовым;
# - когда агрегирование помогает прогнозировать прерывистый низ — разбивка по классам спроса;
# - коротко ETNA: иерархический `TSDataset`, `HierarchicalPipeline`, экзогенные признаки на нижнем уровне.

# %%
import warnings
warnings.filterwarnings('ignore')

from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsforecast import StatsForecast
from statsforecast.models import AutoETS, CrostonSBA, TSB
from hierarchicalforecast.utils import aggregate
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, TopDown, MiddleOut, MinTrace
from hierarchicalforecast.evaluation import evaluate
from utilsforecast.losses import rmsse, bias

plt.rcParams['figure.figsize'] = (12, 4)

H = 28          # горизонт, как в M5
INSAMPLE = 364  # последние дни обучения для долей и ковариации остатков

# %% [markdown]
# ## 0. Данные
#
# M5, магазин CA_1, дневные продажи 3049 товаров. Иерархия задаётся полями `cat_id` (3 категории)
# и `dept_id` (7 отделов).

# %%
sales = pd.read_csv('../data/sales_train_validation.csv.gz')
calendar = pd.read_csv('../data/calendar.csv', parse_dates=['date'])

sales_ca1 = sales[sales['store_id'] == 'CA_1']
day_cols = [c for c in sales_ca1.columns if c.startswith('d_')]

bottom = (sales_ca1.melt(id_vars=['item_id', 'dept_id', 'cat_id'], value_vars=day_cols,
                         var_name='d', value_name='y')
          .merge(calendar[['d', 'date']], on='d')
          .rename(columns={'date': 'ds'}))
bottom['total'] = 'total'

spec = [
    ['total'],
    ['total', 'cat_id'],
    ['total', 'cat_id', 'dept_id'],
    ['total', 'cat_id', 'dept_id', 'item_id'],
]
Y_df, S_df, tags = aggregate(bottom[['ds', 'y', 'total', 'cat_id', 'dept_id', 'item_id']], spec)

print({level: len(ids) for level, ids in tags.items()})
print(f'Период: {Y_df["ds"].min().date()} — {Y_df["ds"].max().date()}')

# %% [markdown]
# ## 1. Агрегаты ровные, низ прерывистый
#
# Доля нулевых дней по уровням иерархии.

# %%
zero_share = Y_df.assign(is_zero=Y_df['y'] == 0).groupby('unique_id')['is_zero'].mean()
pd.DataFrame({
    'рядов': {level: len(ids) for level, ids in tags.items()},
    'средняя доля нулей': {level: zero_share.loc[ids].mean() for level, ids in tags.items()},
}).round(3)

# %%
examples = ['total', 'total/FOODS', 'total/FOODS/FOODS_3', 'total/FOODS/FOODS_3/FOODS_3_001']
fig, axes = plt.subplots(len(examples), 1, figsize=(13, 8), sharex=True)
for ax, uid in zip(axes, examples):
    s = Y_df[(Y_df['unique_id'] == uid) & (Y_df['ds'] >= '2015-07-01')]
    ax.plot(s['ds'], s['y'], linewidth=0.8)
    ax.set_title(uid, fontsize=10)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# На агрегатах — недельная сезонность и тренд, на них хорошо работает ETS. У товаров — нули и редкие продажи,
# ETS тут бесполезен, нужен Croston/TSB. Значит, **базовые модели должны быть разными на разных уровнях**,
# а согласование соберёт их в единую иерархию.

# %% [markdown]
# ## 2. Базовые прогнозы по уровням
#
# - агрегаты (11 рядов): `AutoETS(season_length=7)`;
# - товары (3049 рядов): `CrostonSBA` и `TSB` — два варианта нижнего уровня.
#
# Прогнозы на обучении (`fitted=True`) нужны для долей top-down и ковариации остатков MinT.

# %%
cutoff = Y_df['ds'].max() - pd.Timedelta(days=H)
Y_train = Y_df[Y_df['ds'] <= cutoff]
Y_test = Y_df[Y_df['ds'] > cutoff]

is_item = Y_train['unique_id'].isin(tags['total/cat_id/dept_id/item_id'])

sf_agg = StatsForecast(models=[AutoETS(season_length=7)], freq='D', n_jobs=1)
fc_agg = sf_agg.forecast(df=Y_train[~is_item], h=H, fitted=True)
fit_agg = sf_agg.forecast_fitted_values()

sf_item = StatsForecast(models=[CrostonSBA(), TSB(alpha_d=0.1, alpha_p=0.1)], freq='D', n_jobs=1)
fc_item = sf_item.forecast(df=Y_train[is_item], h=H, fitted=True)
fit_item = sf_item.forecast_fitted_values()

# %% [markdown]
# Собираем два набора базовых прогнозов: «ETS + SBA» и «ETS + TSB». На агрегатах они совпадают.

# %%
def combine(fc_agg, fc_item):
    agg = fc_agg.assign(SBA=fc_agg['AutoETS'], TSB=fc_agg['AutoETS']).drop(columns='AutoETS')
    return pd.concat([agg, fc_item.rename(columns={'CrostonSBA': 'SBA'})], ignore_index=True)


Y_hat = combine(fc_agg, fc_item)
Y_fitted = combine(fit_agg, fit_item)
Y_fitted = Y_fitted[Y_fitted['ds'] > cutoff - pd.Timedelta(days=INSAMPLE)]   # последний год обучения

Y_hat.head()

# %% [markdown]
# ## 3. BU vs TD vs MiddleOut vs MinT
#
# - **BU** — базовый уровень товары (Croston/TSB), агрегаты — сумма;
# - **TD** — базовый уровень total (ETS), доли товаров — `proportion_averages` за последний год;
# - **MiddleOut** — базовый уровень отделы (ETS), вниз — доли, вверх — сумма;
# - **MinT shrink** — все уровни сразу.
#
# Ковариация остатков MinT для 3060 рядов по 364 дням вырождена — без shrinkage не обойтись.

# %%
reconcilers = [
    BottomUp(),
    TopDown(method='proportion_averages'),
    MiddleOut(middle_level='total/cat_id/dept_id', top_down_method='proportion_averages'),
    MinTrace(method='mint_shrink'),
]
hrec = HierarchicalReconciliation(reconcilers=reconcilers)
Y_rec = hrec.reconcile(Y_hat_df=Y_hat, Y_df=Y_fitted, S_df=S_df, tags=tags)


def short_name(col):
    return (col.replace('/BottomUp', ' BU')
               .replace('/TopDown_method-proportion_averages', ' TD')
               .replace('/MiddleOut_middle_level-total/cat_id/dept_id_top_down_method-proportion_averages', ' MiddleOut')
               .replace('/MinTrace_method-mint_shrink', ' MinT'))


Y_rec = Y_rec.rename(columns=short_name)
models = [c for c in Y_rec.columns if c not in ('unique_id', 'ds')]
models

# %% [markdown]
# TD и MiddleOut раздают вниз прогноз ETS, поэтому от варианта нижней модели (SBA или TSB) не зависят —
# оставим по одной копии.

# %%
models = [m for m in models if m not in ('TSB TD', 'TSB MiddleOut')]
Y_rec = Y_rec.rename(columns={'SBA TD': 'TD', 'SBA MiddleOut': 'MiddleOut', 'SBA': 'Base SBA', 'TSB': 'Base TSB'})
models = ['Base SBA', 'Base TSB', 'SBA BU', 'TSB BU', 'TD', 'MiddleOut', 'SBA MinT', 'TSB MinT']

print('Отрицательных прогнозов MinT:', {m: int((Y_rec[m] < 0).sum()) for m in ['SBA MinT', 'TSB MinT']})

# %% [markdown]
# MinT поправляет и товары — часть прогнозов уходит в минус. Для спроса обрежем их нулём
# (`nonnegative=True` на 3049 нижних рядах — квадратичная задача, слишком медленно для семинара).
# После обрезки иерархия слегка рассогласуется — честная цена простого решения.

# %%
Y_rec[['SBA MinT', 'TSB MinT']] = Y_rec[['SBA MinT', 'TSB MinT']].clip(lower=0)

# %% [markdown]
# ### Метрики по уровням
#
# Как и в блоке про прерывистый спрос — RMSSE (масштаб — наивный прогноз на обучении) и bias
# (среднее «прогноз − факт»). Для масштаба RMSSE обрезаем ведущие нули: до начала продаж товара наивная
# ошибка нулевая и искусственно уменьшает знаменатель.

# %%
first_sale = Y_train[Y_train['y'] > 0].groupby('unique_id')['ds'].min()
Y_train_trim = Y_train[Y_train['ds'] >= Y_train['unique_id'].map(first_sale)]

Y_eval = Y_rec.merge(Y_test, on=['unique_id', 'ds'])
evaluation = evaluate(Y_eval, metrics=[partial(rmsse, seasonality=1), bias],
                      tags=tags, models=models, train_df=Y_train_trim)
table_rmsse = evaluation[evaluation['metric'] == "rmsse"].drop(columns='metric').set_index('level')
table_bias = evaluation[evaluation['metric'] == "bias"].drop(columns='metric').set_index('level')

# %%
table_rmsse.style.format('{:.3f}').highlight_min(axis=1, color='lightgreen')

# %%
table_bias

# %%
table_bias.abs().style.format('{:.3f}').highlight_min(axis=1, color='lightgreen')

# %% [markdown]
# ### Наблюдения
#
# - **BU плох на агрегатах:** RMSSE 1.11 на total против 0.405 у ETS. Сумма тысяч плоских прогнозов
#   Croston/TSB не знает о недельной сезонности. Схема «Croston + bottom-up» из M5 хороша только для самих товаров.
# - **TD** (ETS на total + доли за последний год) — лучший на категориях (0.508) и отделах (0.702 против 0.984
#   у прямого ETS по отделам). На товарах (0.729) — между SBA и TSB.
# - **MinT shrink с TSB** — лучший на товарах (0.706) и улучшает отделы (0.822), но портит total и категории
#   (≈ 0.47 против 0.405): шумные остатки 3049 товаров тянут агрегаты. Плюс обрезка отрицательных
#   (у TSB MinT их тысячи) слегка рассогласует иерархию.
# - **TSB лучше SBA** на товарах (0.715 против 0.740): TSB обновляет вероятность продажи каждый день.
# - **Bias** (прогноз − факт): ETS на агрегатах занижает (≈ −130 шт./день на total), и TD / MiddleOut / MinT
#   наследуют это занижение; BU, наоборот, немного завышает.
# - Строка Overall почти целиком определяется товарами (3049 из 3060 рядов) — смотреть надо по уровням.
#
# **Какой уровень делать базовым:** для отчёта по агрегатам — верхний (TD / прямой ETS), для заказа
# товаров — нижний с TSB, а MinT даёт лучший компромисс для нижних уровней.

# %% [markdown]
# ## 4. Когда агрегирование помогает прерывистому низу
#
# Разобьём товары по классам спроса (ADI / CV² на обучении после первой продажи) и сравним точность
# на нижнем уровне: собственная модель товара (BU = базовый Croston/TSB) против долей от агрегата.

# %%
items_train = Y_train_trim[Y_train_trim['unique_id'].isin(tags['total/cat_id/dept_id/item_id'])]
nonzero = items_train[items_train['y'] > 0].sort_values(['unique_id', 'ds'])

adi = nonzero.groupby('unique_id')['ds'].diff().dt.days.groupby(nonzero['unique_id']).mean()
cv2 = nonzero.groupby('unique_id')['y'].agg(lambda v: (v.std(ddof=0) / v.mean()) ** 2)

demand_class = pd.Series(
    np.select([(adi < 1.32) & (cv2 < 0.49), (adi >= 1.32) & (cv2 < 0.49), (adi < 1.32) & (cv2 >= 0.49)],
              ['smooth', 'intermittent', 'erratic'], default='lumpy'),
    index=adi.index, name='class')
demand_class.value_counts()

# %%
item_eval = Y_eval[Y_eval['unique_id'].isin(tags['total/cat_id/dept_id/item_id'])]
item_scores = rmsse(item_eval, models=models, seasonality=1,
                    train_df=items_train).set_index('unique_id').join(demand_class)

by_class = item_scores.groupby('class')[models].mean()
by_class['товаров'] = item_scores['class'].value_counts()
by_class.style.format('{:.3f}').highlight_min(axis=1, color='lightgreen')

# %% [markdown]
# ### Наблюдения
#
# - **smooth и erratic** (частые продажи, ADI < 1.32): доли от агрегата не хуже собственной модели —
#   TD 0.764 против 0.776 у TSB (smooth), 0.652 против 0.654 (erratic). Доля такого товара в отделе стабильна,
#   а прогноз агрегата знает о сезонности.
# - **intermittent и lumpy** (редкие продажи, это 2731 товар из 3049): собственный TSB лучше долей —
#   0.721 против 0.738 и 0.669 против 0.690. Средняя доля редкого товара плохо описывает его ближайшие продажи.
# - MinT с TSB — лучший во всех четырёх классах: он смешивает обе информации, а не выбирает одну.
#
# **Правило:** агрегирование помогает прерывистому низу, когда продажи частые; для редких товаров нужна
# собственная модель (TSB), а лучше — согласование, которое использует оба уровня.

# %% [markdown]
# ## 5. Коротко: иерархия в ETNA
#
# В ETNA иерархия живёт внутри `TSDataset`, а согласование встроено в пайплайн:
#
# - `TSDataset.to_hierarchical_dataset` строит структуру по колонкам уровней (total добавляется сам);
# - `HierarchicalPipeline` = модель на исходном уровне `source_level` + согласование на целевой `target_level`;
# - `BottomUpReconciliator` — снизу вверх; `TopDownReconciliator` — сверху вниз с долями
#   AHP (среднее долей) или PHA (доля средних) за последние `period` точек.
#
# Бонус ETNA — экзогенные признаки и глобальная модель на нижнем уровне. Возьмём иерархию
# категория → отдел (7 нижних рядов, последние 2 года), CatBoost с лагами ≥ H и признаками
# SNAP / событий, и сравним прогнозы на уровне категорий.

# %%
from etna.datasets import TSDataset
from etna.models import CatBoostMultiSegmentModel
from etna.pipeline import Pipeline, HierarchicalPipeline
from etna.reconciliation import BottomUpReconciliator, TopDownReconciliator
from etna.transforms import LagTransform, DateFlagsTransform
from etna.metrics import MAE, SMAPE

start = cutoff + pd.Timedelta(days=H) - pd.Timedelta(days=730)
dept_df = (bottom[bottom['ds'] > start]
           .groupby(['ds', 'cat_id', 'dept_id'], as_index=False)['y'].sum()
           .rename(columns={'ds': 'timestamp', 'y': 'target'}))

hier_df, hier_structure = TSDataset.to_hierarchical_dataset(df=dept_df, level_columns=['cat_id', 'dept_id'], sep='/')

calendar_exog = calendar.loc[calendar['date'] > start, ['date', 'snap_CA', 'event_name_1']]
calendar_exog = pd.DataFrame({
    'timestamp': calendar_exog['date'],
    'snap_CA': calendar_exog['snap_CA'],
    'has_event': calendar_exog['event_name_1'].notna().astype(int),
})


def exog_for(segments):
    """Календарь один на весь магазин — размножаем его по сегментам (wide-формат ETNA)."""
    return TSDataset.to_dataset(calendar_exog.merge(pd.DataFrame({'segment': segments}), how='cross'))


dept_segments = hier_df.columns.get_level_values('segment').unique()
ts = TSDataset(df=hier_df, df_exog=exog_for(dept_segments), freq='D', known_future='all',
               hierarchical_structure=hier_structure)
print(ts.level_names(), ts.current_df_level, list(dept_segments))

# %%
def make_transforms():
    return [
        LagTransform(in_column='target', lags=list(range(H, H + 7)), out_column='lag'),
        DateFlagsTransform(day_number_in_week=True, day_number_in_month=False, is_weekend=False, out_column='flag'),
    ]


def make_model():
    return CatBoostMultiSegmentModel(iterations=300, depth=4, random_seed=42)


etna_pipelines = {
    'Прямой прогноз категорий': Pipeline(model=make_model(), transforms=make_transforms(), horizon=H),
    'BU из отделов': HierarchicalPipeline(
        reconciliator=BottomUpReconciliator(target_level='cat_id', source_level='dept_id'),
        model=make_model(), transforms=make_transforms(), horizon=H),
    'TD из total (AHP)': HierarchicalPipeline(
        reconciliator=TopDownReconciliator(target_level='cat_id', source_level='total', period=91, method='AHP'),
        model=make_model(), transforms=make_transforms(), horizon=H),
    'TD из total (PHA)': HierarchicalPipeline(
        reconciliator=TopDownReconciliator(target_level='cat_id', source_level='total', period=91, method='PHA'),
        model=make_model(), transforms=make_transforms(), horizon=H),
}

# %% [markdown]
# Экзогенные признаки заданы на нижнем уровне (отделы) и вверх не агрегируются: при `source_level='total'`
# ETNA теряет колонки признаков, но помнит их в списке регрессоров — и модель падает. Поэтому TD-пайплайнам
# дадим иерархический датасет без признаков (они работают только на лагах и дне недели).
# Чтобы прямая модель категорий была в равных условиях с BU, соберём для неё отдельный датасет с тем же календарём.

# %%
ts_no_exog = TSDataset(df=hier_df, freq='D', hierarchical_structure=hier_structure)

cat_df = TSDataset.to_dataset(
    dept_df.groupby(['timestamp', 'cat_id'], as_index=False)['target'].sum().rename(columns={'cat_id': 'segment'}))
ts_cat = TSDataset(df=cat_df, df_exog=exog_for(cat_df.columns.get_level_values('segment').unique()),
                   freq='D', known_future='all')

datasets = {
    'Прямой прогноз категорий': ts_cat,
    'BU из отделов': ts,
    'TD из total (AHP)': ts_no_exog,
    'TD из total (PHA)': ts_no_exog,
}

etna_results = {}
for name, pipe in etna_pipelines.items():
    metrics = pipe.backtest(ts=datasets[name], metrics=[MAE(), SMAPE()], n_folds=3, aggregate_metrics=True)['metrics']
    etna_results[name] = metrics.set_index('segment')

etna_table = pd.concat(etna_results, names=['метод']).reset_index()
etna_table.pivot(index='segment', columns='метод', values='SMAPE').round(2)

# %% [markdown]
# ### Наблюдения
#
# - **FOODS:** лучший — BU из отделов (SMAPE 6.43 против 7.06 у прямого прогноза и ≈ 7.9 у TD). Признак SNAP
#   (продовольственные талоны) влияет именно на еду, а модель отделов видит его на каждом отделе отдельно.
# - **HOBBIES:** лучший — прямой прогноз категории (13.25); BU и TD хуже на 0.6–1.3 п.п.
# - **HOUSEHOLD:** лучшие — TD из total (≈ 7.6), хотя у них нет календарных признаков вовсе.
# - AHP и PHA почти не различаются: за 91 день доли категорий стабильны.
# - Единого победителя снова нет, а различия — около процентного пункта на трёх фолдах. Выбор уровня
#   для базовой модели — гиперпараметр, который проверяем на бэктесте, как и любой другой.

# %% [markdown]
# ## Итог
#
# | Ситуация | Что делать |
# |---|---|
# | Агрегаты ровные, низ прерывистый | Разные базовые модели по уровням: ETS наверху, Croston/TSB внизу |
# | Нужна точность на агрегатах | Не BU: сумма тысяч Croston-прогнозов теряет сезонность; TD или прямой прогноз |
# | Нужна точность на товарах | TSB лучше SBA; доли от агрегата помогают только товарам с частыми продажами |
# | Нужен компромисс по всем уровням | MinT shrink + обрезка отрицательных: лучший внизу, но проигрывает наверху |
# | Есть экзогенные признаки на нижнем уровне | Глобальная модель (ETNA `HierarchicalPipeline`) + согласование |
