# Занятие 12. Нейросетевые модели (2): PatchTST, TSMixer, TFT

Материалы двенадцатого занятия курса по прогнозированию временных рядов. Второе занятие про нейросети. Разбираемся, что на самом деле нужно трансформеру, чтобы работать на рядах: правильный токен, внимание или подготовка входов. Все модели строим в `neuralforecast`, бейзлайны — в `statsforecast`. Дополнительно в части 2 есть два ноутбука с TFT на `pytorch-forecasting`.

- **Часть 1** — трансформеры против линейных моделей на ETTm2. PatchTST: почему скалярный токен ограничивает внимание, патчинг (сколько токенов и во сколько раз дешевле карта внимания), устройство модели, сравнение с DLinear и поточечным `VanillaTransformer`, главная абляция `patch_len=1`, независимость каналов и RevIN, что патчинг не решает. TSMixer: окно как матрица `L × C`, temporal и feature mixing, почему DLinear — это TSMixer без них, смешивает ли модель каналы на самом деле (опыт с испорченными каналами), TSMixerx — ковариаты как ещё несколько каналов и где смешивание каналов окупается.
- **Часть 2** — TFT как система под прикладную задачу на почасовом потреблении электроэнергии в Испании с календарём и погодой. Три типа входов (статические, исторические, известные в будущем) и главный вопрос к признаку — будет ли он известен в момент прогноза. Планка `SeasonalNaive`. Устройство TFT: GRN, отбор переменных, LSTM как позиционное кодирование, интерпретируемое внимание. TFT без признаков и с признаками, важность переменных, внимание как функция лага, квантили через `MQLoss`, что TFT умеет и чего не умеет. Дополнительно — TFT в `pytorch-forecasting`: прогноз спроса (Stallion) и многомерный прогноз потребления электроэнергии в Тетуане (Марокко).

---

## Ноутбуки

### Часть 1. PatchTST и TSMixer

| № | Ноутбук | Colab |
|---|---------|-------|
| 12.1 | [PatchTST: токен как отрезок](part_1/1.%20PatchTST%20%E2%80%94%20%D1%82%D0%BE%D0%BA%D0%B5%D0%BD%20%D0%BA%D0%B0%D0%BA%20%D0%BE%D1%82%D1%80%D0%B5%D0%B7%D0%BE%D0%BA.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_12/part_1/1.%20PatchTST%20%E2%80%94%20%D1%82%D0%BE%D0%BA%D0%B5%D0%BD%20%D0%BA%D0%B0%D0%BA%20%D0%BE%D1%82%D1%80%D0%B5%D0%B7%D0%BE%D0%BA.ipynb) |
| 12.2 | [TSMixer: а нужен ли трансформер](part_1/2.%20TSMixer%20%E2%80%94%20%D0%B0%20%D0%BD%D1%83%D0%B6%D0%B5%D0%BD%20%D0%BB%D0%B8%20%D1%82%D1%80%D0%B0%D0%BD%D1%81%D1%84%D0%BE%D1%80%D0%BC%D0%B5%D1%80.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_12/part_1/2.%20TSMixer%20%E2%80%94%20%D0%B0%20%D0%BD%D1%83%D0%B6%D0%B5%D0%BD%20%D0%BB%D0%B8%20%D1%82%D1%80%D0%B0%D0%BD%D1%81%D1%84%D0%BE%D1%80%D0%BC%D0%B5%D1%80.ipynb) |

### Часть 2. TFT

| № | Ноутбук | Colab |
|---|---------|-------|
| 12.3 | [TFT: трансформер как система](part_2/1.%20TFT%20%E2%80%94%20%D1%82%D1%80%D0%B0%D0%BD%D1%81%D1%84%D0%BE%D1%80%D0%BC%D0%B5%D1%80%20%D0%BA%D0%B0%D0%BA%20%D1%81%D0%B8%D1%81%D1%82%D0%B5%D0%BC%D0%B0.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_12/part_2/1.%20TFT%20%E2%80%94%20%D1%82%D1%80%D0%B0%D0%BD%D1%81%D1%84%D0%BE%D1%80%D0%BC%D0%B5%D1%80%20%D0%BA%D0%B0%D0%BA%20%D1%81%D0%B8%D1%81%D1%82%D0%B5%D0%BC%D0%B0.ipynb) |
| доп. | [TFT: прогноз спроса Stallion (PyTorch Forecasting)](part_2/2.%20TFT_stallion.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_12/part_2/2.%20TFT_stallion.ipynb) |
| доп. | [TFT: энергопотребление Тетуана, Марокко (PyTorch Forecasting)](part_2/3.%20TFT_energy_Morocco.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_12/part_2/3.%20TFT_energy_Morocco.ipynb) |
