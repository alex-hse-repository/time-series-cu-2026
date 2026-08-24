# Занятие 2. Базовые методы прогнозирования и оценка качества

Материалы второго занятия курса по прогнозированию временных рядов. Занятие про полный цикл базового прогноза: построить простую модель, привести таргет в удобный вид, получить интервал — и честно всё это измерить.

- **Часть 1** — базовые методы (mean, naive, seasonal naive, drift), преобразования таргета (пропуски, Бокс–Кокс, ограничения на диапазон, обратное преобразование) и интервалы прогнозирования вместе с метриками их качества.
- **Часть 2** — как проверять модель: стратегии кросс-валидации для временных рядов, анализ остатков и «зоопарк» метрик точечного прогноза.

Сквозная логика ноутбуков: сначала всё считается **руками** на numpy/pandas, затем то же самое показывается в трёх фреймворках — `statsforecast` (Nixtla), `sktime` и `etna` — и сводится в сравнительную таблицу.

Ноутбуки можно запускать двумя способами: **локально** (нужен Python 3.11+ и [uv](https://docs.astral.sh/uv/)) или **в Google Colab** — по ссылкам из таблицы, без установки чего-либо. Инструкция по запуску — в [README репозитория](../README.md), для этого занятия `week_N` = `week_2`.

---

## Ноутбуки

### Часть 1. Базовые методы, преобразования и интервалы

| № | Ноутбук | Colab |
|---|---------|-------|
| 1 | [Базовые методы прогнозирования](part_1/01.%20%D0%91%D0%B0%D0%B7%D0%BE%D0%B2%D1%8B%D0%B5%20%D0%BC%D0%B5%D1%82%D0%BE%D0%B4%D1%8B%20%D0%BF%D1%80%D0%BE%D0%B3%D0%BD%D0%BE%D0%B7%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D1%8F.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_2/part_1/01.%20%D0%91%D0%B0%D0%B7%D0%BE%D0%B2%D1%8B%D0%B5%20%D0%BC%D0%B5%D1%82%D0%BE%D0%B4%D1%8B%20%D0%BF%D1%80%D0%BE%D0%B3%D0%BD%D0%BE%D0%B7%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D1%8F.ipynb) |
| 2 | [Преобразование таргета](part_1/02.%20%D0%9F%D1%80%D0%B5%D0%BE%D0%B1%D1%80%D0%B0%D0%B7%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D1%82%D0%B0%D1%80%D0%B3%D0%B5%D1%82%D0%B0.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_2/part_1/02.%20%D0%9F%D1%80%D0%B5%D0%BE%D0%B1%D1%80%D0%B0%D0%B7%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D1%82%D0%B0%D1%80%D0%B3%D0%B5%D1%82%D0%B0.ipynb) |
| 3 | [Интервалы прогнозирования и оценка их качества](part_1/03.%20%D0%98%D0%BD%D1%82%D0%B5%D1%80%D0%B2%D0%B0%D0%BB%D1%8B%20%D0%BF%D1%80%D0%BE%D0%B3%D0%BD%D0%BE%D0%B7%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D1%8F%20%D0%B8%20%D0%BE%D1%86%D0%B5%D0%BD%D0%BA%D0%B0%20%D0%B8%D1%85%20%D0%BA%D0%B0%D1%87%D0%B5%D1%81%D1%82%D0%B2%D0%B0.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_2/part_1/03.%20%D0%98%D0%BD%D1%82%D0%B5%D1%80%D0%B2%D0%B0%D0%BB%D1%8B%20%D0%BF%D1%80%D0%BE%D0%B3%D0%BD%D0%BE%D0%B7%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D1%8F%20%D0%B8%20%D0%BE%D1%86%D0%B5%D0%BD%D0%BA%D0%B0%20%D0%B8%D1%85%20%D0%BA%D0%B0%D1%87%D0%B5%D1%81%D1%82%D0%B2%D0%B0.ipynb) |

### Часть 2. Валидация, остатки и метрики

| № | Ноутбук | Colab |
|---|---------|-------|
| 1 | [Стратегии кросс-валидации](part_2/01.%20%D0%A1%D1%82%D1%80%D0%B0%D1%82%D0%B5%D0%B3%D0%B8%D0%B8%20%D0%BA%D1%80%D0%BE%D1%81%D1%81-%D0%B2%D0%B0%D0%BB%D0%B8%D0%B4%D0%B0%D1%86%D0%B8%D0%B8.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_2/part_2/01.%20%D0%A1%D1%82%D1%80%D0%B0%D1%82%D0%B5%D0%B3%D0%B8%D0%B8%20%D0%BA%D1%80%D0%BE%D1%81%D1%81-%D0%B2%D0%B0%D0%BB%D0%B8%D0%B4%D0%B0%D1%86%D0%B8%D0%B8.ipynb) |
| 2 | [Анализ остатков](part_2/02.%20%D0%90%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7%20%D0%BE%D1%81%D1%82%D0%B0%D1%82%D0%BA%D0%BE%D0%B2.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_2/part_2/02.%20%D0%90%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7%20%D0%BE%D1%81%D1%82%D0%B0%D1%82%D0%BA%D0%BE%D0%B2.ipynb) |
| 3 | [Метрики качества прогноза](part_2/03.%20%D0%9C%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%20%D0%BA%D0%B0%D1%87%D0%B5%D1%81%D1%82%D0%B2%D0%B0%20%D0%BF%D1%80%D0%BE%D0%B3%D0%BD%D0%BE%D0%B7%D0%B0.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_2/part_2/03.%20%D0%9C%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%20%D0%BA%D0%B0%D1%87%D0%B5%D1%81%D1%82%D0%B2%D0%B0%20%D0%BF%D1%80%D0%BE%D0%B3%D0%BD%D0%BE%D0%B7%D0%B0.ipynb) |
