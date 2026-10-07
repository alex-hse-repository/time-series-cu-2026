# Занятие 6. Декомпозиция, Theta и поиск аномалий

Материалы шестого занятия курса по прогнозированию временных рядов. Занятие про разложение ряда на тренд, сезонность и остаток, использование компонент для прогнозирования, модель Theta как сильный бейзлайн, а также про поиск аномалий и точек смены поведения во временных рядах.

- **Часть 1** — методы декомпозиции (классическая, STL, MSTL) и их гиперпараметры, декомпозиция для прогнозирования (остатки для аномалий, прогноз ряда без сезонности, компоненты как признаки, утечка при двустороннем сглаживании), семейство моделей Theta в statsforecast и сравнение с ETS и ARIMA на рядах M4.
- **Часть 2** — поиск точечных аномалий: rule-based подходы на глобальных и скользящих статистиках, подход через прогнозную модель, сведение к табличной задаче (AutoEncoder, Isolation Forest), ансамблирование детекторов; поиск точек смены поведения (ruptures, скользящие статистики, Prophet).

---

## Ноутбуки

### Часть 1. Декомпозиция и Theta

| № | Ноутбук | Colab |
|---|---------|-------|
| 1 | [Методы декомпозиции](part_1/01.%20%D0%9C%D0%B5%D1%82%D0%BE%D0%B4%D1%8B%20%D0%B4%D0%B5%D0%BA%D0%BE%D0%BC%D0%BF%D0%BE%D0%B7%D0%B8%D1%86%D0%B8%D0%B8.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_6/part_1/01.%20%D0%9C%D0%B5%D1%82%D0%BE%D0%B4%D1%8B%20%D0%B4%D0%B5%D0%BA%D0%BE%D0%BC%D0%BF%D0%BE%D0%B7%D0%B8%D1%86%D0%B8%D0%B8.ipynb) |
| 2 | [Декомпозиция для прогнозирования](part_1/02.%20%D0%94%D0%B5%D0%BA%D0%BE%D0%BC%D0%BF%D0%BE%D0%B7%D0%B8%D1%86%D0%B8%D1%8F%20%D0%B4%D0%BB%D1%8F%20%D0%BF%D1%80%D0%BE%D0%B3%D0%BD%D0%BE%D0%B7%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D1%8F.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_6/part_1/02.%20%D0%94%D0%B5%D0%BA%D0%BE%D0%BC%D0%BF%D0%BE%D0%B7%D0%B8%D1%86%D0%B8%D1%8F%20%D0%B4%D0%BB%D1%8F%20%D0%BF%D1%80%D0%BE%D0%B3%D0%BD%D0%BE%D0%B7%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D1%8F.ipynb) |
| 3 | [Theta](part_1/03.%20Theta.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_6/part_1/03.%20Theta.ipynb) |

### Часть 2. Поиск аномалий

| № | Ноутбук | Colab |
|---|---------|-------|
| 1 | [Задача поиска аномалий](part_2/01.%20%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0%20%D0%BF%D0%BE%D0%B8%D1%81%D0%BA%D0%B0%20%D0%B0%D0%BD%D0%BE%D0%BC%D0%B0%D0%BB%D0%B8%D0%B9.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_6/part_2/01.%20%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0%20%D0%BF%D0%BE%D0%B8%D1%81%D0%BA%D0%B0%20%D0%B0%D0%BD%D0%BE%D0%BC%D0%B0%D0%BB%D0%B8%D0%B9.ipynb) |
