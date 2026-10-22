# Занятие 8. Prophet, BATS и TBATS

Материалы восьмого занятия курса по прогнозированию временных рядов. Занятие про две модели для рядов со сложной структурой: Prophet — аддитивная регрессия на кусочно-линейный тренд, Фурье-сезонности и праздники, и TBATS — модель в пространстве состояний для нескольких (в том числе дробных) сезонностей.

- **Часть 1** — Prophet: устройство модели (тренд с изломами, сезонность как сумма Фурье, праздники и регрессоры, Stan-модель и сборка Prophet руками через `sklearn`, три источника неопределённости в интервалах, MCMC, влияние выбросов); компоненты и их гиперпараметры (`changepoint_prior_scale`, `changepoint_range`, ручные изломы, логистический и постоянный тренд, `fourier_order`, свои и условные сезонности, мультипликативная сезонность, календарь событий и государственные праздники); внешняя информация и «неудобные» данные (регрессоры с известным и неизвестным будущим, COVID-локдауны как разовые праздники, внутридневные, почасовые и месячные данные, регулярные пропуски); кросс-валидация, метрики по горизонту и покрытие интервалов, тюнинг гиперпараметров, сравнение с бейзлайнами, случаи, когда Prophet брать не стоит, тёплый старт и сохранение модели.
- **Часть 2** — BATS и TBATS: мотивация через множественную сезонность, уравнения модели, дискретная (BATS) и тригонометрическая (TBATS) сезонность, запись в пространстве состояний, как `AutoTBATS` из statsforecast выбирает модель; сравнение компонент с MSTL, варианты с ARMA-ошибками, трендом и Бокс-Коксом, сравнение AutoTBATS с MSTL и SeasonalNaive на почасовом трафике и потреблении электроэнергии.

---

## Ноутбуки

### Часть 1. Prophet

| № | Ноутбук | Colab |
|---|---------|-------|
| 1 | [Устройство модели](part_1/Prophet%201.%20%D0%A3%D1%81%D1%82%D1%80%D0%BE%D0%B9%D1%81%D1%82%D0%B2%D0%BE%20%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_8/part_1/Prophet%201.%20%D0%A3%D1%81%D1%82%D1%80%D0%BE%D0%B9%D1%81%D1%82%D0%B2%D0%BE%20%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8.ipynb) |
| 2 | [Компоненты и их ручки](part_1/Prophet%202.%20%D0%9A%D0%BE%D0%BC%D0%BF%D0%BE%D0%BD%D0%B5%D0%BD%D1%82%D1%8B%20%D0%B8%20%D0%B8%D1%85%20%D1%80%D1%83%D1%87%D0%BA%D0%B8.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_8/part_1/Prophet%202.%20%D0%9A%D0%BE%D0%BC%D0%BF%D0%BE%D0%BD%D0%B5%D0%BD%D1%82%D1%8B%20%D0%B8%20%D0%B8%D1%85%20%D1%80%D1%83%D1%87%D0%BA%D0%B8.ipynb) |
| 3 | [Внешняя информация и неудобные данные](part_1/Prophet%203.%20%D0%92%D0%BD%D0%B5%D1%88%D0%BD%D1%8F%D1%8F%20%D0%B8%D0%BD%D1%84%D0%BE%D1%80%D0%BC%D0%B0%D1%86%D0%B8%D1%8F%20%D0%B8%20%D0%BD%D0%B5%D1%83%D0%B4%D0%BE%D0%B1%D0%BD%D1%8B%D0%B5%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_8/part_1/Prophet%203.%20%D0%92%D0%BD%D0%B5%D1%88%D0%BD%D1%8F%D1%8F%20%D0%B8%D0%BD%D1%84%D0%BE%D1%80%D0%BC%D0%B0%D1%86%D0%B8%D1%8F%20%D0%B8%20%D0%BD%D0%B5%D1%83%D0%B4%D0%BE%D0%B1%D0%BD%D1%8B%D0%B5%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5.ipynb) |
| 4 | [Валидация, сравнение, ограничения](part_1/Prophet%204.%20%D0%92%D0%B0%D0%BB%D0%B8%D0%B4%D0%B0%D1%86%D0%B8%D1%8F%2C%20%D1%81%D1%80%D0%B0%D0%B2%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5%2C%20%D0%BE%D0%B3%D1%80%D0%B0%D0%BD%D0%B8%D1%87%D0%B5%D0%BD%D0%B8%D1%8F.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_8/part_1/Prophet%204.%20%D0%92%D0%B0%D0%BB%D0%B8%D0%B4%D0%B0%D1%86%D0%B8%D1%8F%2C%20%D1%81%D1%80%D0%B0%D0%B2%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5%2C%20%D0%BE%D0%B3%D1%80%D0%B0%D0%BD%D0%B8%D1%87%D0%B5%D0%BD%D0%B8%D1%8F.ipynb) |

### Часть 2. BATS и TBATS

| № | Ноутбук | Colab |
|---|---------|-------|
| 1 | [Модели BATS и TBATS](part_2/BATS%20%D0%B8%20TBATS.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_8/part_2/BATS%20%D0%B8%20TBATS.ipynb) |
