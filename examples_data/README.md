# examples_data

Данные для ноутбуков курса. Ноутбуки читают их по прямой ссылке, чтобы запускаться
и в CI, и в Google Colab без локальных файлов:

```
https://raw.githubusercontent.com/alex-hse-repository/time-series-cu-2026/refs/heads/main/examples_data/<файл>
```

Ожидаемые файлы:

| Файл | Где используется |
| --- | --- |
| `Retail_Data_Transactions.csv` | week_1/part_1/1. Работа с ВР в pandas |
| `monthly-australian-wine-sales.csv` | week_1/part_1/2, week_1/part_1/3 |
| `nordic_merch_sales.csv` | week_1/part_1/2, week_1/part_1/3 |
| `nordics_weather.csv` | week_1/part_1/2, week_1/part_1/3 |
| `AirPassengers.csv` | week_1/part_2/1, week_1/part_2/2 |
| `Measurement_summary.csv` | week_1/part_2/2 |
| `monthly-car-sales.csv` | week_1/part_2/1, week_1/part_2/2 (задания) |
| `daily-total-female-births.csv` | week_1/part_2/1, week_1/part_2/2 (задания) |
| `jj.csv` | week_2/part_1/01, week_2/part_1/02, week_2/part_1/03, week_2/part_2/03 |
| `a10.csv` | week_2/part_1/02, week_2/part_2/01, week_2/part_2/03 |
| `a10_missings.csv` | week_2/part_1/02 |
| `weekly_flat.csv` | week_2/part_1/02 |
| `aus-production.csv` | week_2/part_1/02, week_2/part_2/03 |
| `GOOGL.csv` | week_2/part_2/01, week_2/part_2/02, week_2/part_2/03 |
| `results.csv` | week_2/part_2/03 |

Ноутбук `week_1/part_2/3. Открытые источники данных временных рядов.ipynb` не трогали:
он и так качает данные из открытых источников (sktime, kagglehub, HuggingFace, LSTNet)
в локальный кэш `data/`.
