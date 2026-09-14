# Занятие 5. SARIMA и регрессоры

Материалы пятого занятия курса по прогнозированию временных рядов. Занятие про расширение ARIMA на сезонные ряды и добавление внешних переменных: как устроена модель SARIMA, как подбирать её порядки, как работает механика прогноза, и как правильно включать регрессоры для улучшения предсказаний.

- **Часть 1** — внутреннее устройство процесса fit, четыре стратегии подбора порядков (по шагам, grid search, auto_arima, AutoARIMA), практика автоматического подбора в разных библиотеках.
- **Часть 2** — механика прогноза в SARIMAX, как работают сезонные компоненты (S), внешние переменные (X) и аналитические доверительные интервалы, где брать регрессоры на будущее, end-to-end примеры на трёх реальных кейсах по единому чек-листу.

---

## Ноутбуки

### Часть 1. Подбор порядков SARIMA

| № | Ноутбук | Colab |
|---|---------|-------|
| 1 | [Что происходит внутри fit](part_1/01.%20%D0%A7%D1%82%D0%BE%20%D0%BF%D1%80%D0%BE%D0%B8%D1%81%D1%85%D0%BE%D0%B4%D0%B8%D1%82%20%D0%B2%D0%BD%D1%83%D1%82%D1%80%D0%B8%20fit.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_5/part_1/01.%20%D0%A7%D1%82%D0%BE%20%D0%BF%D1%80%D0%BE%D0%B8%D1%81%D1%85%D0%BE%D0%B4%D0%B8%D1%82%20%D0%B2%D0%BD%D1%83%D1%82%D1%80%D0%B8%20fit.ipynb) |
| 2 | [Подбор порядков — четыре стратегии](part_1/02.%20%D0%9F%D0%BE%D0%B4%D0%B1%D0%BE%D1%80%20%D0%BF%D0%BE%D1%80%D1%8F%D0%B4%D0%BA%D0%BE%D0%B2%20%E2%80%94%20%D1%87%D0%B5%D1%82%D1%8B%D1%80%D0%B5%20%D1%81%D1%82%D1%80%D0%B0%D1%82%D0%B5%D0%B3%D0%B8%D0%B8.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_5/part_1/02.%20%D0%9F%D0%BE%D0%B4%D0%B1%D0%BE%D1%80%20%D0%BF%D0%BE%D1%80%D1%8F%D0%B4%D0%BA%D0%BE%D0%B2%20%E2%80%94%20%D1%87%D0%B5%D1%82%D1%8B%D1%80%D0%B5%20%D1%81%D1%82%D1%80%D0%B0%D1%82%D0%B5%D0%B3%D0%B8%D0%B8.ipynb) |
| 3 | [Автоподбор порядка](part_1/03.%20%D0%90%D0%B2%D1%82%D0%BE%D0%BF%D0%BE%D0%B4%D0%B1%D0%BE%D1%80%20%D0%BF%D0%BE%D1%80%D1%8F%D0%B4%D0%BA%D0%B0.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_5/part_1/03.%20%D0%90%D0%B2%D1%82%D0%BE%D0%BF%D0%BE%D0%B4%D0%B1%D0%BE%D1%80%20%D0%BF%D0%BE%D1%80%D1%8F%D0%B4%D0%BA%D0%B0.ipynb) |

### Часть 2. SARIMAX и регрессоры

| № | Ноутбук | Colab |
|---|---------|-------|
| 1 | [Механика прогноза — S, X и аналитические интервалы](part_2/01.%20%D0%9C%D0%B5%D1%85%D0%B0%D0%BD%D0%B8%D0%BA%D0%B0%20%D0%BF%D1%80%D0%BE%D0%B3%D0%BD%D0%BE%D0%B7%D0%B0%20%E2%80%94%20S,%20X%20%D0%B8%20%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B5%20%D0%B8%D0%BD%D1%82%D0%B5%D1%80%D0%B2%D0%B0%D0%BB%D1%8B.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_5/part_2/01.%20%D0%9C%D0%B5%D1%85%D0%B0%D0%BD%D0%B8%D0%BA%D0%B0%20%D0%BF%D1%80%D0%BE%D0%B3%D0%BD%D0%BE%D0%B7%D0%B0%20%E2%80%94%20S,%20X%20%D0%B8%20%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B5%20%D0%B8%D0%BD%D1%82%D0%B5%D1%80%D0%B2%D0%B0%D0%BB%D1%8B.ipynb) |
| 2 | [Регрессоры в SARIMAX — где их взять на будущее](part_2/02.%20%D0%A0%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%BE%D1%80%D1%8B%20%D0%B2%20SARIMAX%20%E2%80%94%20%D0%B3%D0%B4%D0%B5%20%D0%B8%D1%85%20%D0%B2%D0%B7%D1%8F%D1%82%D1%8C%20%D0%BD%D0%B0%20%D0%B1%D1%83%D0%B4%D1%83%D1%89%D0%B5%D0%B5.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_5/part_2/02.%20%D0%A0%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%BE%D1%80%D1%8B%20%D0%B2%20SARIMAX%20%E2%80%94%20%D0%B3%D0%B4%D0%B5%20%D0%B8%D1%85%20%D0%B2%D0%B7%D1%8F%D1%82%D1%8C%20%D0%BD%D0%B0%20%D0%B1%D1%83%D0%B4%D1%83%D1%89%D0%B5%D0%B5.ipynb) |
| 3 | [End-to-end — три кейса по одному чек-листу](part_2/03.%20End-to-end%20%E2%80%94%20%D1%82%D1%80%D0%B8%20%D0%BA%D0%B5%D0%B9%D1%81%D0%B0%20%D0%BF%D0%BE%20%D0%BE%D0%B4%D0%BD%D0%BE%D0%BC%D1%83%20%D1%87%D0%B5%D0%BA-%D0%BB%D0%B8%D1%81%D1%82%D1%83.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_5/part_2/03.%20End-to-end%20%E2%80%94%20%D1%82%D1%80%D0%B8%20%D0%BA%D0%B5%D0%B9%D1%81%D0%B0%20%D0%BF%D0%BE%20%D0%BE%D0%B4%D0%BD%D0%BE%D0%BC%D1%83%20%D1%87%D0%B5%D0%BA-%D0%BB%D0%B8%D1%81%D1%82%D1%83.ipynb) |
