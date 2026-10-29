# Занятие 9. ML-модели в прогнозировании: деревья, глобальные модели и стратегии

Материалы девятого занятия курса по прогнозированию временных рядов. Занятие про то, что происходит с бустингом на временных рядах, когда ряд один, когда рядов много и как из одношаговой модели получить прогноз на весь горизонт.

- **Часть 1** — деревья и постановка задачи: почему дерево не экстраполирует (лист — среднее по train), календарные признаки против тренда, детрендинг и снятие сезонности трансформациями, лаги как способ сообщить модели свежий уровень, ловушка «логарифм + линейный тренд»; ручной пайплайн признаков против `sktime` (`make_reduction`, `WindowSummarizer`, `DateTimeFeatures`), маска теста против утечки, пропуски дат и `shift` по строкам вместо дней; глобальная модель на 414 рядах M4 Hourly в ETNA — `CatBoostPerSegmentModel` против `CatBoostMultiSegmentModel`, зачем снимать масштаб логарифмом и нормировкой.
- **Часть 2** — модели как источники признаков и стратегии прогнозирования: прогноз, компоненты, остатки и параметры ETS как признаки бустинга и ловушка in-sample; компоненты Prophet (тренд как уровень ряда против календаря), остатки `y / trend`, признаки-«оракулы» вроде фактической погоды; механика стратегий — рекурсивная, упрощённая прямая, прямая, DirRec, MIMO и DIRMO, ручные реализации против `sktime`/ETNA и `np.allclose`; сравнение стратегий на синтетике с известной «правдой» — аддитивный и фазовый шум, ошибка по горизонту.

---

## Ноутбуки

### Часть 1. Деревья и глобальные модели

| № | Ноутбук | Colab |
|---|---------|-------|
| 1 | [Деревья и экстраполяция](part_1/01.%20%D0%94%D0%B5%D1%80%D0%B5%D0%B2%D1%8C%D1%8F%20%D0%B8%20%D1%8D%D0%BA%D1%81%D1%82%D1%80%D0%B0%D0%BF%D0%BE%D0%BB%D1%8F%D1%86%D0%B8%D1%8F.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_9/part_1/01.%20%D0%94%D0%B5%D1%80%D0%B5%D0%B2%D1%8C%D1%8F%20%D0%B8%20%D1%8D%D0%BA%D1%81%D1%82%D1%80%D0%B0%D0%BF%D0%BE%D0%BB%D1%8F%D1%86%D0%B8%D1%8F.ipynb) |
| 2 | [Пайплайн руками vs sktime](part_1/02.%20%D0%9F%D0%B0%D0%B9%D0%BF%D0%BB%D0%B0%D0%B9%D0%BD%20%D1%80%D1%83%D0%BA%D0%B0%D0%BC%D0%B8%20vs%20sktime.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_9/part_1/02.%20%D0%9F%D0%B0%D0%B9%D0%BF%D0%BB%D0%B0%D0%B9%D0%BD%20%D1%80%D1%83%D0%BA%D0%B0%D0%BC%D0%B8%20vs%20sktime.ipynb) |
| 3 | [Глобальная модель](part_1/03.%20%D0%93%D0%BB%D0%BE%D0%B1%D0%B0%D0%BB%D1%8C%D0%BD%D0%B0%D1%8F%20%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D1%8C.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_9/part_1/03.%20%D0%93%D0%BB%D0%BE%D0%B1%D0%B0%D0%BB%D1%8C%D0%BD%D0%B0%D1%8F%20%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D1%8C.ipynb) |

### Часть 2. Признаки из моделей и стратегии прогнозирования

| № | Ноутбук | Colab |
|---|---------|-------|
| 1 | [Признаки из ETS](part_2/01.%20%D0%9F%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%D0%B8%20%D0%B8%D0%B7%20ETS.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_9/part_2/01.%20%D0%9F%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%D0%B8%20%D0%B8%D0%B7%20ETS.ipynb) |
| 2 | [Признаки из Prophet](part_2/02.%20%D0%9F%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%D0%B8%20%D0%B8%D0%B7%20Prophet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_9/part_2/02.%20%D0%9F%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%D0%B8%20%D0%B8%D0%B7%20Prophet.ipynb) |
| 3 | [Механика стратегий](part_2/03.%20%D0%9C%D0%B5%D1%85%D0%B0%D0%BD%D0%B8%D0%BA%D0%B0%20%D1%81%D1%82%D1%80%D0%B0%D1%82%D0%B5%D0%B3%D0%B8%D0%B9.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_9/part_2/03.%20%D0%9C%D0%B5%D1%85%D0%B0%D0%BD%D0%B8%D0%BA%D0%B0%20%D1%81%D1%82%D1%80%D0%B0%D1%82%D0%B5%D0%B3%D0%B8%D0%B9.ipynb) |
| 4 | [Сравнение стратегий на синтетике](part_2/04.%20%D0%A1%D1%80%D0%B0%D0%B2%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5%20%D1%81%D1%82%D1%80%D0%B0%D1%82%D0%B5%D0%B3%D0%B8%D0%B9%20%D0%BD%D0%B0%20%D1%81%D0%B8%D0%BD%D1%82%D0%B5%D1%82%D0%B8%D0%BA%D0%B5.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_9/part_2/04.%20%D0%A1%D1%80%D0%B0%D0%B2%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5%20%D1%81%D1%82%D1%80%D0%B0%D1%82%D0%B5%D0%B3%D0%B8%D0%B9%20%D0%BD%D0%B0%20%D1%81%D0%B8%D0%BD%D1%82%D0%B5%D1%82%D0%B8%D0%BA%D0%B5.ipynb) |
