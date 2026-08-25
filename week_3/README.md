# Занятие 3. Экспоненциальное сглаживание (ETS) и сезонность

Материалы третьего занятия курса по прогнозированию временных рядов. Занятие про семейство ETS: от рекуррентных формул, которые можно пересчитать в столбик, до автоподбора спецификации на тысячах рядов — и про то, как найти период сезонности, если он заранее неизвестен.

- **Часть 1** — ETS изнутри: фильтрация против прогноза и скрытое состояние модели (SES, Хольт, Хольт–Винтерс) «руками» и в `statsmodels`, всё семейство ETS в `statsforecast`, автоподбор спецификации по AICc против честной кросс-валидации.
- **Часть 2** — практика: полный кейс прогноза почасового ряда через ETS, преобразование Фурье и спектр ряда, определение периода сезонности по спектру.

---

## Ноутбуки

### Часть 1. Модели ETS

| № | Ноутбук | Colab |
|---|---------|-------|
| 1 | [Фильтрация и прогноз в ETS](part_1/01.%20%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80%D0%B0%D1%86%D0%B8%D1%8F%20%D0%B8%20%D0%BF%D1%80%D0%BE%D0%B3%D0%BD%D0%BE%D0%B7%20%D0%B2%20ETS.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_3/part_1/01.%20%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80%D0%B0%D1%86%D0%B8%D1%8F%20%D0%B8%20%D0%BF%D1%80%D0%BE%D0%B3%D0%BD%D0%BE%D0%B7%20%D0%B2%20ETS.ipynb) |
| 2 | [Семейство ETS в `StatsForecast`](part_1/02.%20%D0%A1%D0%B5%D0%BC%D0%B5%D0%B9%D1%81%D1%82%D0%B2%D0%BE%20ETS%20%D0%B2%20StatsForecast.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_3/part_1/02.%20%D0%A1%D0%B5%D0%BC%D0%B5%D0%B9%D1%81%D1%82%D0%B2%D0%BE%20ETS%20%D0%B2%20StatsForecast.ipynb) |
| 3 | [Автоподбор ETS: AICc против кросс-валидации](part_1/03.%20%D0%90%D0%B2%D1%82%D0%BE%D0%BF%D0%BE%D0%B4%D0%B1%D0%BE%D1%80%20ETS%20AICc%20%D0%BF%D1%80%D0%BE%D1%82%D0%B8%D0%B2%20%D0%BA%D1%80%D0%BE%D1%81%D1%81-%D0%B2%D0%B0%D0%BB%D0%B8%D0%B4%D0%B0%D1%86%D0%B8%D0%B8.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_3/part_1/03.%20%D0%90%D0%B2%D1%82%D0%BE%D0%BF%D0%BE%D0%B4%D0%B1%D0%BE%D1%80%20ETS%20AICc%20%D0%BF%D1%80%D0%BE%D1%82%D0%B8%D0%B2%20%D0%BA%D1%80%D0%BE%D1%81%D1%81-%D0%B2%D0%B0%D0%BB%D0%B8%D0%B4%D0%B0%D1%86%D0%B8%D0%B8.ipynb) |

### Часть 2. Кейс и поиск сезонности

| № | Ноутбук | Colab |
|---|---------|-------|
| 1 | [Кейс прогноза через ETS](part_2/01.%20%D0%9A%D0%B5%D0%B9%D1%81%20%D0%BF%D1%80%D0%BE%D0%B3%D0%BD%D0%BE%D0%B7%D0%B0%20%D1%87%D0%B5%D1%80%D0%B5%D0%B7%20ETS.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_3/part_2/01.%20%D0%9A%D0%B5%D0%B9%D1%81%20%D0%BF%D1%80%D0%BE%D0%B3%D0%BD%D0%BE%D0%B7%D0%B0%20%D1%87%D0%B5%D1%80%D0%B5%D0%B7%20ETS.ipynb) |
| 2 | [Преобразование Фурье. Спектр временного ряда](part_2/02.%20%D0%9F%D1%80%D0%B5%D0%BE%D0%B1%D1%80%D0%B0%D0%B7%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%A4%D1%83%D1%80%D1%8C%D0%B5.%20%D0%A1%D0%BF%D0%B5%D0%BA%D1%82%D1%80%20%D0%B2%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D0%BE%D0%B3%D0%BE%20%D1%80%D1%8F%D0%B4%D0%B0.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_3/part_2/02.%20%D0%9F%D1%80%D0%B5%D0%BE%D0%B1%D1%80%D0%B0%D0%B7%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%A4%D1%83%D1%80%D1%8C%D0%B5.%20%D0%A1%D0%BF%D0%B5%D0%BA%D1%82%D1%80%20%D0%B2%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D0%BE%D0%B3%D0%BE%20%D1%80%D1%8F%D0%B4%D0%B0.ipynb) |
| 3 | [Определение периода сезонности временного ряда](part_2/03.%20%D0%9E%D0%BF%D1%80%D0%B5%D0%B4%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%BF%D0%B5%D1%80%D0%B8%D0%BE%D0%B4%D0%B0%20%D1%81%D0%B5%D0%B7%D0%BE%D0%BD%D0%BD%D0%BE%D1%81%D1%82%D0%B8%20%D0%B2%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D0%BE%D0%B3%D0%BE%20%D1%80%D1%8F%D0%B4%D0%B0.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_3/part_2/03.%20%D0%9E%D0%BF%D1%80%D0%B5%D0%B4%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%BF%D0%B5%D1%80%D0%B8%D0%BE%D0%B4%D0%B0%20%D1%81%D0%B5%D0%B7%D0%BE%D0%BD%D0%BD%D0%BE%D1%81%D1%82%D0%B8%20%D0%B2%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D0%BE%D0%B3%D0%BE%20%D1%80%D1%8F%D0%B4%D0%B0.ipynb) |
