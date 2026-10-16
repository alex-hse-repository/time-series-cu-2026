# Занятие 7. Прогнозирование временных рядов с помощью ML

Материалы седьмого занятия курса по прогнозированию временных рядов. Занятие про сведение прогнозирования к задаче регрессии: как честно поставить задачу без утечки будущего, какие признаки можно построить из календаря и истории ряда, как бороться с большим числом признаков и как обучать одну модель сразу на много рядов.

- **Часть 1** — постановка задачи и бэктест, утечка будущего (лаг меньше горизонта), тренд и детрендинг, кодирование сезонности (dummy, Фурье, RBF, сплайны), праздники и события; лаговые признаки и выбор лагов по ACF/PACF, рекурсивная стратегия, скользящие и расширяющиеся окна; регуляризация (Ridge / Lasso / ElasticNet) и отбор признаков деревом, групповые признаки для одной модели на много рядов в ETNA.
- **Часть 2** — end-to-end пример прогноза веб-трафика с ридж-регрессией в skforecast: анализ сезонности и автокорреляций, базовая и авторегрессионная модели, экзогенные факторы, важность признаков через SHAP и прогнозные интервалы.

---

## Ноутбуки

### Часть 1. Признаки для ML-моделей

| № | Ноутбук | Colab |
|---|---------|-------|
| 1 | [Постановка и календарные признаки](part_1/1.%20%D0%9F%D0%BE%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BA%D0%B0%20%D0%B8%20%D0%BA%D0%B0%D0%BB%D0%B5%D0%BD%D0%B4%D0%B0%D1%80%D0%BD%D1%8B%D0%B5%20%D0%BF%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%D0%B8.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_7/part_1/1.%20%D0%9F%D0%BE%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BA%D0%B0%20%D0%B8%20%D0%BA%D0%B0%D0%BB%D0%B5%D0%BD%D0%B4%D0%B0%D1%80%D0%BD%D1%8B%D0%B5%20%D0%BF%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%D0%B8.ipynb) |
| 2 | [Лаги и окна](part_1/2.%20%D0%9B%D0%B0%D0%B3%D0%B8%20%D0%B8%20%D0%BE%D0%BA%D0%BD%D0%B0.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_7/part_1/2.%20%D0%9B%D0%B0%D0%B3%D0%B8%20%D0%B8%20%D0%BE%D0%BA%D0%BD%D0%B0.ipynb) |
| 3 | [Много признаков и много рядов](part_1/3.%20%D0%9C%D0%BD%D0%BE%D0%B3%D0%BE%20%D0%BF%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%D0%BE%D0%B2%20%D0%B8%20%D0%BC%D0%BD%D0%BE%D0%B3%D0%BE%20%D1%80%D1%8F%D0%B4%D0%BE%D0%B2.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_7/part_1/3.%20%D0%9C%D0%BD%D0%BE%D0%B3%D0%BE%20%D0%BF%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%D0%BE%D0%B2%20%D0%B8%20%D0%BC%D0%BD%D0%BE%D0%B3%D0%BE%20%D1%80%D1%8F%D0%B4%D0%BE%D0%B2.ipynb) |

### Часть 2. Практический кейс

| № | Ноутбук | Colab |
|---|---------|-------|
| 1 | [Прогнозирование веб-трафика с помощью ридж-регрессии](part_2/1.%20Forecasting%20web%20traffic.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_7/part_2/1.%20Forecasting%20web%20traffic.ipynb) |
