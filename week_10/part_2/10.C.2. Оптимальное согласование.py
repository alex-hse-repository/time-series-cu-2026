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
#     display_name: "Python (ts-cu-2026 \xB7 week 10(dev))"
#     language: python
#     name: ts-cu-2026-week10-dev
# ---

# %% [markdown]
# # 10.C.2. Оптимальное согласование
#
# **Сквозной вопрос блока:** прогнозы уровней не складываются — как сделать их согласованными и при этом точнее?
#
# В этом ноутбуке:
# - согласование как проекция: игрушечный пример total = a + b, OLS руками;
# - взвешенные варианты: WLS struct, WLS var, MinT; почему `mint_cov` не годится при коротких рядах, а `mint_shrink` — да;
# - несмещённость: условие $SGS = S$;
# - сравнение всех методов по уровням;
# - отрицательные согласованные прогнозы и `nonnegative=True`;
# - вероятностное согласование: интервалы `normality` / `bootstrap` / `permbu` и их покрытие;
# - что дальше: временные и кросс-временные иерархии.

# %%
import warnings
warnings.filterwarnings('ignore')

from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsforecast import StatsForecast
from statsforecast.models import AutoETS
from hierarchicalforecast.utils import aggregate
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, TopDown, MinTrace
from hierarchicalforecast.evaluation import evaluate
from utilsforecast.losses import mase, coverage

plt.rcParams['figure.figsize'] = (12, 4)

H = 8  # горизонт: 8 кварталов

# %% [markdown]
# ## 0. Данные и базовые прогнозы
#
# TourismSmall: 56 нижних рядов `штат-цель-зона`, иерархия total → цель → штат×цель → city/noncity (89 рядов),
# 36 кварталов. Последние 8 кварталов — тест, базовая модель — AutoETS по каждому ряду.

# %%
wide = pd.read_csv('../data/hierarchical/TourismSmall/data.csv', index_col=0, parse_dates=True)
bottom_cols = [c for c in wide.columns if c.count('-') == 2]

bottom = (wide[bottom_cols]
          .rename_axis('ds').reset_index()
          .melt('ds', var_name='series', value_name='y'))
bottom[['state', 'purpose', 'area']] = bottom['series'].str.split('-', expand=True)
bottom['total'] = 'total'

spec = [
    ['total'],
    ['total', 'purpose'],
    ['total', 'purpose', 'state'],
    ['total', 'purpose', 'state', 'area'],
]
Y_df, S_df, tags = aggregate(bottom[['ds', 'y', 'total', 'purpose', 'state', 'area']], spec)

S = S_df.drop(columns='unique_id').to_numpy()
n_series, n_bottom = S.shape
ids = S_df['unique_id'].to_numpy()


def to_matrix(df, col):
    """Long-формат → матрица (ряды в порядке S × моменты времени)."""
    return df.pivot(index='unique_id', columns='ds', values=col).loc[ids].to_numpy()


cutoff = Y_df['ds'].sort_values().unique()[-H - 1]
Y_train = Y_df[Y_df['ds'] <= cutoff]
Y_test = Y_df[Y_df['ds'] > cutoff]

# %%
sf = StatsForecast(models=[AutoETS(season_length=4)], freq='QE', n_jobs=1)
Y_hat = sf.forecast(df=Y_train, h=H, fitted=True, level=[80, 95])
Y_fitted = sf.forecast_fitted_values()

Y_hat_mat = to_matrix(Y_hat, 'AutoETS')                                       # 89 × 8
residuals = to_matrix(Y_fitted, 'y') - to_matrix(Y_fitted, 'AutoETS')         # 89 × 28 — ошибки на обучении

print(f'S: {S.shape}, базовые прогнозы: {Y_hat_mat.shape}, остатки: {residuals.shape}')

# %% [markdown]
# ## 1. Согласование как проекция
#
# Согласованные векторы $y = S b$ образуют подпространство размерности $n_{bottom}$ (столбцы $S$).
# Базовые прогнозы $\hat y$ в него обычно не попадают. Самое естественное — спроецировать $\hat y$ на это
# подпространство: $\tilde y = S G \hat y$, $G = (S^\top S)^{-1} S^\top$ — обычный МНК (**OLS**).
#
# Игрушечный пример: total = a + b, базовые прогнозы total = 10, a = 4, b = 5 (4 + 5 ≠ 10).

# %%
S_toy = np.array([[1, 1],
                  [1, 0],
                  [0, 1]])
y_hat_toy = np.array([10., 4., 5.])

G_ols_toy = np.linalg.inv(S_toy.T @ S_toy) @ S_toy.T
y_tilde_toy = S_toy @ G_ols_toy @ y_hat_toy

print('согласованный прогноз (total, a, b):', y_tilde_toy.round(3))
print('total = a + b:', np.isclose(y_tilde_toy[0], y_tilde_toy[1] + y_tilde_toy[2]))

P = S_toy @ G_ols_toy
print('P = SG — проектор (P·P = P):', np.allclose(P @ P, P))
print('поправка ортогональна плоскости (Sᵀ(ŷ − ỹ) = 0):', np.allclose(S_toy.T @ (y_hat_toy - y_tilde_toy), 0))

# %% [markdown]
# Расхождение 1 распределилось поровну на три ряда (по 1/3): total уменьшился, a и b выросли.
# Геометрически — перпендикуляр из $\hat y$ на плоскость согласованных прогнозов.

# %%
a_grid, b_grid = np.meshgrid(np.linspace(2, 7, 10), np.linspace(2, 7, 10))

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(projection='3d')
ax.plot_surface(a_grid, b_grid, a_grid + b_grid, alpha=0.25)
ax.scatter(*y_hat_toy[[1, 2, 0]], color='red', s=60, label='базовый ŷ')
ax.scatter(*y_tilde_toy[[1, 2, 0]], color='green', s=60, label='согласованный ỹ (OLS)')
ax.plot(*np.c_[y_hat_toy, y_tilde_toy][[1, 2, 0]], 'k--')
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('total')
ax.set_title('Плоскость total = a + b')
ax.legend()
plt.show()

# %% [markdown]
# Та же формула на TourismSmall — и сверка с `MinTrace(method='ols')`.

# %%
def reconcile_manual(W):
    """G = (Sᵀ W⁻¹ S)⁻¹ Sᵀ W⁻¹ и согласованный прогноз S·G·ŷ."""
    W_inv = np.linalg.inv(W)
    G = np.linalg.solve(S.T @ W_inv @ S, S.T @ W_inv)
    return G, S @ G @ Y_hat_mat


G_ols, Y_ols_manual = reconcile_manual(np.eye(n_series))

Y_ols_lib = HierarchicalReconciliation([MinTrace(method='ols')]).reconcile(
    Y_hat_df=Y_hat[['unique_id', 'ds', 'AutoETS']], S_df=S_df, tags=tags)
print('OLS руками = библиотека:', np.allclose(Y_ols_manual, to_matrix(Y_ols_lib, 'AutoETS/MinTrace_method-ols')))

# %% [markdown]
# ## 2. Взвешенное согласование: WLS и MinT
#
# OLS считает все ряды одинаково надёжными. Но total прогнозируется куда точнее, чем `nt-oth-noncity`.
# Общий вид — обобщённый МНК с матрицей весов $W$ (ковариацией ошибок базовых прогнозов):
#
# $$G = (S^\top W^{-1} S)^{-1} S^\top W^{-1}$$
#
# | метод | $W$ |
# |---|---|
# | OLS | $I$ |
# | WLS struct | $\mathrm{diag}(S\mathbf{1})$ — число нижних рядов в агрегате |
# | WLS var | $\mathrm{diag}(\hat\sigma^2_i)$ — дисперсии остатков на обучении |
# | MinT (`mint_cov`) | полная выборочная ковариация остатков |
# | MinT (`mint_shrink`) | $\lambda\,\mathrm{diag}(\hat W) + (1-\lambda)\hat W$ — сжатие к диагонали |
#
# MinT (Wickramasuriya, Athanasopoulos, Hyndman, 2019) минимизирует суммарную дисперсию ошибок согласованных
# прогнозов среди всех несмещённых методов.

# %%
W_struct = np.diag(S.sum(axis=1))
W_var = np.diag((residuals ** 2).mean(axis=1))

_, Y_struct_manual = reconcile_manual(W_struct)
_, Y_var_manual = reconcile_manual(W_var)

Y_wls_lib = HierarchicalReconciliation([MinTrace(method='wls_struct'), MinTrace(method='wls_var')]).reconcile(
    Y_hat_df=Y_hat[['unique_id', 'ds', 'AutoETS']], Y_df=Y_fitted[['unique_id', 'ds', 'y', 'AutoETS']],
    S_df=S_df, tags=tags)
print('WLS struct руками = библиотека:', np.allclose(Y_struct_manual, to_matrix(Y_wls_lib, 'AutoETS/MinTrace_method-wls_struct')))
print('WLS var руками = библиотека:   ', np.allclose(Y_var_manual, to_matrix(Y_wls_lib, 'AutoETS/MinTrace_method-wls_var'), rtol=1e-4))

# %% [markdown]
# ### MinT при коротких рядах
#
# Для полной ковариации нужно оценить 89·90/2 ≈ 4000 параметров по 28 векторам остатков.
# Выборочная ковариация имеет ранг не выше 27 — она вырождена, и $W^{-1}$ в формуле не существует.

# %%
W_cov = np.cov(residuals)

lam_check = MinTrace(method='mint_shrink')
lam_check.fit(S=S, y_hat=Y_hat_mat, y_insample=to_matrix(Y_fitted, 'y'), y_hat_insample=to_matrix(Y_fitted, 'AutoETS'))
W_shrink = lam_check.W

print(f'размер W: {W_cov.shape}, наблюдений: {residuals.shape[1]}')
print(f'ранг выборочной ковариации: {np.linalg.matrix_rank(W_cov)}')
print(f'число обусловленности: mint_cov {np.linalg.cond(W_cov):.1e}, mint_shrink {np.linalg.cond(W_shrink):.1e}')

# %% [markdown]
# hierarchicalforecast не обращает $W$ напрямую (решает эквивалентную систему размера «число агрегатов»),
# поэтому `mint_cov` формально считается. Но результат опирается на вырожденную, плохо оценённую матрицу —
# посмотрим ниже, как это сказывается на точности. Shrinkage (Schäfer–Strimmer) подтягивает внедиагональные
# элементы к нулю и делает $W$ невырожденной — это стандартный выбор на практике.

# %% [markdown]
# ## 3. Несмещённость: $SGS = S$
#
# Если базовые прогнозы несмещённые ($E\hat y = S\beta$), согласованные тоже несмещённые, когда
# $E\tilde y = SGS\beta = S\beta$, то есть при $SGS = S$. Для любой $G = (S^\top W^{-1}S)^{-1}S^\top W^{-1}$
# это выполнено автоматически ($GS = I$). Проверим, в том числе для top-down.

# %%
train_mat = to_matrix(Y_train, 'y')
p_avg = (train_mat[-n_bottom:] / train_mat[0]).mean(axis=1)

G_matrices = {
    'BU': np.hstack([np.zeros((n_bottom, n_series - n_bottom)), np.eye(n_bottom)]),
    'TD avg_prop': np.hstack([p_avg[:, None], np.zeros((n_bottom, n_series - 1))]),
    'OLS': G_ols,
    'WLS struct': reconcile_manual(W_struct)[0],
    'WLS var': reconcile_manual(W_var)[0],
    'MinT shrink': reconcile_manual(W_shrink)[0],
}
for name, G in G_matrices.items():
    print(f'{name:12s} SGS = S: {np.allclose(S @ G @ S, S)}')

# %% [markdown]
# TD нарушает условие: даже идеальные несмещённые нижние прогнозы он заменяет долями от total.

# %% [markdown]
# ## 4. Сравнение всех методов по уровням

# %%
reconcilers = [
    BottomUp(),
    TopDown(method='forecast_proportions'),
    MinTrace(method='ols'),
    MinTrace(method='wls_struct'),
    MinTrace(method='wls_var'),
    MinTrace(method='mint_cov'),
    MinTrace(method='mint_shrink'),
]
names = {
    'AutoETS': 'Base',
    'AutoETS/BottomUp': 'BU',
    'AutoETS/TopDown_method-forecast_proportions': 'TD fcst_prop',
    'AutoETS/MinTrace_method-ols': 'OLS',
    'AutoETS/MinTrace_method-wls_struct': 'WLS struct',
    'AutoETS/MinTrace_method-wls_var': 'WLS var',
    'AutoETS/MinTrace_method-mint_cov': 'MinT cov',
    'AutoETS/MinTrace_method-mint_shrink': 'MinT shrink',
}

hrec = HierarchicalReconciliation(reconcilers=reconcilers)
Y_rec = hrec.reconcile(
    Y_hat_df=Y_hat[['unique_id', 'ds', 'AutoETS']], Y_df=Y_fitted[['unique_id', 'ds', 'y', 'AutoETS']],
    S_df=S_df, tags=tags,
).rename(columns=names)

Y_eval = Y_rec.merge(Y_test, on=['unique_id', 'ds'])
evaluation = evaluate(Y_eval, metrics=[partial(mase, seasonality=4)], tags=tags, train_df=Y_train)

table = evaluation.drop(columns='metric').set_index('level')
table.style.format('{:.3f}').highlight_min(axis=1, color='lightgreen')

# %% [markdown]
# ### Наблюдения
#
# - **`mint_cov` ломается:** MASE в сотнях на всех уровнях. Ковариация 89 × 89 по 28 точкам вырождена
#   (число обусловленности ~10²⁰), веса бессмысленны — и ниже мы увидим сотни отрицательных прогнозов.
#   Shrinkage полностью убирает проблему.
# - Остальные согласованные методы в целом (Overall) не хуже базового прогноза: согласованность не стоит точности,
#   а обычно даже немного её добавляет (OLS 0.879, TD fcst_prop 0.883, BU 0.887 против 0.908 у Base).
# - Лучшие по уровням разные: на total — BU, на уровне целей — TD `forecast_proportions` и OLS,
#   на нижнем уровне — OLS.
# - OLS / WLS / MinT shrink отличаются на 1–2% — это уровень шума для одного окна из 8 кварталов.
#   Преимущество MinT (минимальная дисперсия) — теоретическое и проявляется при надёжной оценке ковариации,
#   а у нас всего 28 векторов остатков.

# %% [markdown]
# ## 5. Отрицательные согласованные прогнозы
#
# OLS и MinT перераспределяют расхождение между рядами — маленький нижний ряд может получить отрицательную
# поправку больше собственного прогноза. Для ночей в отелях это бессмысленно. (Пара отрицательных значений
# есть уже у базового ETS.)

# %%
negatives = (Y_rec[list(names.values())] < 0).sum()
negatives[negatives > 0] if negatives.any() else negatives

# %% [markdown]
# `nonnegative=True` решает задачу квадратичного программирования: тот же критерий MinT, но с ограничением
# $\tilde b \ge 0$ на нижние ряды (агрегаты тогда неотрицательны автоматически).

# %%
Y_nonneg = HierarchicalReconciliation([MinTrace(method='ols'), MinTrace(method='ols', nonnegative=True)]).reconcile(
    Y_hat_df=Y_hat[['unique_id', 'ds', 'AutoETS']], S_df=S_df, tags=tags)

ols_cols = ['AutoETS/MinTrace_method-ols', 'AutoETS/MinTrace_method-ols_nonnegative-True']
print('Отрицательных прогнозов:')
print((Y_nonneg[ols_cols] < 0).sum())

# %% [markdown]
# ## 6. Вероятностное согласование
#
# Согласовывать можно не только точечные прогнозы, но и интервалы. Нужно, чтобы базовые прогнозы шли с
# интервалами (`level=` в `forecast`), а в `reconcile` указать `level` и способ `intervals_method`:
#
# | `intervals_method` | идея |
# |---|---|
# | `normality` | ошибки нормальны: $\tilde\Sigma = SGWG^\top S^\top$ |
# | `bootstrap` | бутстреп остатков на обучении, каждая выборка согласуется |
# | `permbu` | эмпирические маргинальные распределения + перестановки, сохраняющие ранговую зависимость рядов; собираются снизу вверх |

# %%
prob_frames = []
for method in ['normality', 'bootstrap', 'permbu']:
    rec = HierarchicalReconciliation([BottomUp(), MinTrace(method='mint_shrink')]).reconcile(
        Y_hat_df=Y_hat, Y_df=Y_fitted, S_df=S_df, tags=tags,
        level=[80, 95], intervals_method=method, seed=0,
    )
    rec = rec[[c for c in rec.columns if c in ('unique_id', 'ds') or '/' in c]]   # без базовых колонок
    rec.columns = [c.replace('AutoETS/BottomUp', f'BU {method}')
                    .replace('AutoETS/MinTrace_method-mint_shrink', f'MinT {method}') for c in rec.columns]
    prob_frames.append(rec.set_index(['unique_id', 'ds']))

Y_prob = (pd.concat(prob_frames, axis=1).reset_index()
          .merge(Y_hat.rename(columns=lambda c: c.replace('AutoETS', 'Base')), on=['unique_id', 'ds']))

# %%
Y_prob_eval = Y_prob.merge(Y_test, on=['unique_id', 'ds'])
prob_models = [c for c in Y_prob.columns if c not in ('unique_id', 'ds') and '-lo-' not in c and '-hi-' not in c]

cov = evaluate(Y_prob_eval, metrics=[coverage], tags=tags, models=prob_models, level=[80, 95])
cov_table = cov.set_index(['level', 'metric']).T
cov_table.style.format('{:.2f}')

# %% [markdown]
# ### Наблюдения
#
# - **Нижний уровень** (56 рядов × 8 кварталов = 448 точек): `normality` и `permbu` дают покрытие близко
#   к номиналу (≈ 0.81 / 0.93), `bootstrap` заметно не дотягивает (≈ 0.68 / 0.82): остатки на обучении
#   слишком оптимистичны — та же проблема, что и у бутстрепа одиночного ряда.
# - **Верхние уровни** покрыты хуже, но там мало точек: на total всего 8, один промах — это 12.5 п.п.
#   Базовые интервалы на total покрывают столько же (0.50).
# - MinT сужает интервалы агрегатов (дисперсия согласованного прогноза меньше), поэтому при ошибке модели
#   на верху недопокрытие у него сильнее, чем у BU (`normality`: 0.25 против 0.50 на total при 80%).
# - Вывод: согласование интервалов не чинит плохую базовую модель; покрытие проверяем по уровням,
#   а на верхних уровнях одного окна для такой проверки мало.

# %% [markdown]
# ## 7. За рамками: временные и кросс-временные иерархии
#
# - **Временная иерархия:** один ряд на разных частотах (месяц → квартал → год) — тоже сумма, тоже $y = Sb$.
#   В hierarchicalforecast её строит `aggregate_temporal`, согласование — те же `BottomUp` / `MinTrace`
#   (`reconcile(..., temporal=True)`). Годовой прогноз становится равен сумме месячных.
# - **Кросс-временное согласование:** одновременно по структуре (регионы) и по времени (частоты).
#   Матрица $S$ — произведение Кронекера структурной и временной; `get_cross_temporal_tags` собирает теги.
# - Идея та же, что и с ADIDA для прерывистого спроса: прогноз на агрегированной частоте точнее,
#   а согласование передаёт эту точность обратно на исходную частоту.

# %% [markdown]
# ## Итог
#
# | Метод | $W$ | Что нужно | Когда брать |
# |---|---|---|---|
# | OLS | $I$ | только $S$ | быстрый старт, нет остатков |
# | WLS struct | число нижних рядов | только $S$ | нет надёжных остатков, разный масштаб уровней |
# | WLS var | дисперсии остатков | in-sample прогнозы | короткие ряды, много рядов |
# | MinT cov | полная ковариация | наблюдений ≫ рядов | почти никогда на практике |
# | MinT shrink | сжатая ковариация | in-sample прогнозы | выбор по умолчанию |
#
# - Все методы — $\tilde y = S G \hat y$ с $G = (S^\top W^{-1}S)^{-1}S^\top W^{-1}$; они несмещённые ($SGS = S$),
#   в отличие от top-down.
# - Для неотрицательных величин — `nonnegative=True`.
# - Интервалы согласуются через `level` + `intervals_method`; покрытие проверяем так же, как в любом
#   вероятностном прогнозе — на отложенных данных и по уровням.

# %%
