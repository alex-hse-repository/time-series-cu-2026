# Занятие 13. Foundation-модели для временных рядов

- **Часть 1** — foundation-модель изнутри: данные, нормализация, патчи, архитектура и обучение на примере nanoTSFM, Simons и Tabby
- **Часть 2** — foundation-модели на практике: Chronos-2 с ковариатами, библиотеки, ловушки и LLM-агенты; бустинг с бизнес-признаками против Chronos-2

Все ноутбуки рассчитаны на GPU: в Colab **Runtime → Change runtime type → T4 GPU**.

---

## Ноутбуки

### Часть 1. Foundation-модель изнутри

| № | Ноутбук | Colab |
|---|---------|-------|
| 1 | [Foundation-модель изнутри: nanoTSFM, Simons, Tabby](part_1/1.%20Foundation-%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D1%8C%20%D0%B8%D0%B7%D0%BD%D1%83%D1%82%D1%80%D0%B8.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_13/part_1/1.%20Foundation-%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D1%8C%20%D0%B8%D0%B7%D0%BD%D1%83%D1%82%D1%80%D0%B8.ipynb) |

Ноутбук требует Python 3.12+ (так требует nanoTSFM), поэтому в CI на Python 3.11 он не запускается.

### Часть 2. Foundation-модели на практике

| № | Ноутбук | Colab |
|---|---------|-------|
| 1 | [Foundation-модели на практике: библиотеки и агенты](part_2/1.%20Foundation-%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8%20%D0%BD%D0%B0%20%D0%BF%D1%80%D0%B0%D0%BA%D1%82%D0%B8%D0%BA%D0%B5.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_13/part_2/1.%20Foundation-%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8%20%D0%BD%D0%B0%20%D0%BF%D1%80%D0%B0%D0%BA%D1%82%D0%B8%D0%BA%D0%B5.ipynb) |
| 2 | [Бустинг против foundation-модели: бизнес-признаки, дообучение, интерпретация](part_2/2.%20%D0%91%D1%83%D1%81%D1%82%D0%B8%D0%BD%D0%B3%20%D0%BF%D1%80%D0%BE%D1%82%D0%B8%D0%B2%20foundation-%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_13/part_2/2.%20%D0%91%D1%83%D1%81%D1%82%D0%B8%D0%BD%D0%B3%20%D0%BF%D1%80%D0%BE%D1%82%D0%B8%D0%B2%20foundation-%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8.ipynb) |

Для раздела 5 ноутбука «Foundation-модели на практике» (агенты) нужен ключ OpenAI: в Colab добавьте секрет `OPENAI_API_KEY`. Без ключа раздел пропускается.
