# Занятие 9. ML-модели в прогнозировании: деревья, глобальные модели и стратегии

Материалы девятого занятия курса по прогнозированию временных рядов. Занятие про то, что происходит с бустингом на временных рядах, когда ряд один, когда рядов много и как из одношаговой модели получить прогноз на весь горизонт.

- **Часть 1** — деревья, глобальные модели и признаки ряда: почему дерево не экстраполирует (лист — среднее по train), календарные признаки против тренда, детрендинг и снятие сезонности трансформациями в `sktime`, лаги как способ сообщить модели свежий уровень, ловушка «логарифм + линейный тренд»; глобальная модель на 414 рядах M4 Hourly в ETNA — `CatBoostPerSegmentModel` против `CatBoostMultiSegmentModel`, зачем снимать масштаб логарифмом и нормировкой; «паспорт» ряда из библиотек `catch22`, `tsfresh` и `pyhctsa` — что меряют признаки (синтетика с известным ответом), инвариантность к масштабу, карта рядов через PCA; паспорт как статичные признаки глобальной модели — код сегмента против паспорта, нормировка, утечка при расчёте по всей истории, cold start на новых рядах; признаки `tsfresh` в скользящих окнах (`roll_time_series`), сдвиг таргета на горизонт, почему годовой горизонт — ловушка, сравнение с лагами при $H = 28$ и $H = 7$.
- **Часть 2** — модели как источники признаков и стратегии прогнозирования: прогноз, компоненты, остатки и параметры ETS как признаки бустинга и ловушка in-sample; компоненты Prophet (тренд как уровень ряда против календаря), остатки `y / trend`, признаки-«оракулы» вроде фактической погоды; механика стратегий — рекурсивная, упрощённая прямая, прямая, DirRec, MIMO и DIRMO, ручные реализации против `sktime`/ETNA и `np.allclose`; сравнение стратегий на синтетике с известной «правдой» — аддитивный и фазовый шум, ошибка по горизонту.

---

## Презентации

- [Глобальные признаки](part_1/%D0%93%D0%BB%D0%BE%D0%B1%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B5%20%D0%BF%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%D0%B8.pdf) — к части 1
- [Стратегии прогнозирования](part_2/%D0%A1%D1%82%D1%80%D0%B0%D1%82%D0%B5%D0%B3%D0%B8%D0%B8%20%D0%BF%D1%80%D0%BE%D0%B3%D0%BD%D0%BE%D0%B7%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D1%8F.pdf) — к части 2

## Ноутбуки

### Часть 1. Деревья, глобальные модели и признаки ряда

| № | Ноутбук | Colab |
|---|---------|-------|
| 1 | [Деревья и экстраполяция](part_1/01.%20%D0%94%D0%B5%D1%80%D0%B5%D0%B2%D1%8C%D1%8F%20%D0%B8%20%D1%8D%D0%BA%D1%81%D1%82%D1%80%D0%B0%D0%BF%D0%BE%D0%BB%D1%8F%D1%86%D0%B8%D1%8F.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_9/part_1/01.%20%D0%94%D0%B5%D1%80%D0%B5%D0%B2%D1%8C%D1%8F%20%D0%B8%20%D1%8D%D0%BA%D1%81%D1%82%D1%80%D0%B0%D0%BF%D0%BE%D0%BB%D1%8F%D1%86%D0%B8%D1%8F.ipynb) |
| 2 | [Глобальная модель](part_1/02.%20%D0%93%D0%BB%D0%BE%D0%B1%D0%B0%D0%BB%D1%8C%D0%BD%D0%B0%D1%8F%20%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D1%8C.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_9/part_1/02.%20%D0%93%D0%BB%D0%BE%D0%B1%D0%B0%D0%BB%D1%8C%D0%BD%D0%B0%D1%8F%20%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D1%8C.ipynb) |
| 3 | [Библиотеки признаков](part_1/03.%20%D0%91%D0%B8%D0%B1%D0%BB%D0%B8%D0%BE%D1%82%D0%B5%D0%BA%D0%B8%20%D0%BF%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%D0%BE%D0%B2.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_9/part_1/03.%20%D0%91%D0%B8%D0%B1%D0%BB%D0%B8%D0%BE%D1%82%D0%B5%D0%BA%D0%B8%20%D0%BF%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%D0%BE%D0%B2.ipynb) |
| 4 | [Паспорт в глобальной модели](part_1/04.%20%D0%9F%D0%B0%D1%81%D0%BF%D0%BE%D1%80%D1%82%20%D0%B2%20%D0%B3%D0%BB%D0%BE%D0%B1%D0%B0%D0%BB%D1%8C%D0%BD%D0%BE%D0%B9%20%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_9/part_1/04.%20%D0%9F%D0%B0%D1%81%D0%BF%D0%BE%D1%80%D1%82%20%D0%B2%20%D0%B3%D0%BB%D0%BE%D0%B1%D0%B0%D0%BB%D1%8C%D0%BD%D0%BE%D0%B9%20%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8.ipynb) |
| 5 | [Признаки в окнах (tsfresh)](part_1/05.%20%D0%9F%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%D0%B8%20%D0%B2%20%D0%BE%D0%BA%D0%BD%D0%B0%D1%85%20%28tsfresh%29.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_9/part_1/05.%20%D0%9F%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%D0%B8%20%D0%B2%20%D0%BE%D0%BA%D0%BD%D0%B0%D1%85%20%28tsfresh%29.ipynb) |

### Часть 2. Признаки из моделей и стратегии прогнозирования

| № | Ноутбук | Colab |
|---|---------|-------|
| 1 | [Признаки из ETS](part_2/01.%20%D0%9F%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%D0%B8%20%D0%B8%D0%B7%20ETS.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_9/part_2/01.%20%D0%9F%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%D0%B8%20%D0%B8%D0%B7%20ETS.ipynb) |
| 2 | [Признаки из Prophet](part_2/02.%20%D0%9F%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%D0%B8%20%D0%B8%D0%B7%20Prophet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_9/part_2/02.%20%D0%9F%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%D0%B8%20%D0%B8%D0%B7%20Prophet.ipynb) |
| 3 | [Механика стратегий](part_2/03.%20%D0%9C%D0%B5%D1%85%D0%B0%D0%BD%D0%B8%D0%BA%D0%B0%20%D1%81%D1%82%D1%80%D0%B0%D1%82%D0%B5%D0%B3%D0%B8%D0%B9.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_9/part_2/03.%20%D0%9C%D0%B5%D1%85%D0%B0%D0%BD%D0%B8%D0%BA%D0%B0%20%D1%81%D1%82%D1%80%D0%B0%D1%82%D0%B5%D0%B3%D0%B8%D0%B9.ipynb) |
| 4 | [Сравнение стратегий на синтетике](part_2/04.%20%D0%A1%D1%80%D0%B0%D0%B2%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5%20%D1%81%D1%82%D1%80%D0%B0%D1%82%D0%B5%D0%B3%D0%B8%D0%B9%20%D0%BD%D0%B0%20%D1%81%D0%B8%D0%BD%D1%82%D0%B5%D1%82%D0%B8%D0%BA%D0%B5.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_9/part_2/04.%20%D0%A1%D1%80%D0%B0%D0%B2%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5%20%D1%81%D1%82%D1%80%D0%B0%D1%82%D0%B5%D0%B3%D0%B8%D0%B9%20%D0%BD%D0%B0%20%D1%81%D0%B8%D0%BD%D1%82%D0%B5%D1%82%D0%B8%D0%BA%D0%B5.ipynb) |
