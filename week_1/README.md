# Занятие 1. Введение в анализ временных рядов

Материалы вводного занятия курса по прогнозированию временных рядов.

- **Часть 1** — работа с временными рядами «руками»: `pandas`, виды рядов, характеристики (тренд, сезонность, стационарность, автокорреляция).
- **Часть 2** — обзор ключевых библиотек (`statsmodels`, `statsforecast`, `sktime`) и открытых источников данных.

Ноутбуки можно запускать двумя способами: **локально** (нужен Python 3.11+ и [uv](https://docs.astral.sh/uv/)) или **в Google Colab** — по ссылкам из таблицы, без установки чего-либо.

---

## Ноутбуки

### Часть 1. Временные ряды в pandas

| № | Ноутбук | Colab |
|---|---------|-------|
| 1 | [Работа с ВР в `pandas`](part_1/1.%20%D0%A0%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%20%D1%81%20%D0%92%D0%A0%20%D0%B2%20pandas.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_1/part_1/1.%20%D0%A0%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%20%D1%81%20%D0%92%D0%A0%20%D0%B2%20pandas.ipynb) |
| 2 | [Виды временных рядов](part_1/2.%20%D0%92%D0%B8%D0%B4%D1%8B%20%D0%B2%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D1%8B%D1%85%20%D1%80%D1%8F%D0%B4%D0%BE%D0%B2.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_1/part_1/2.%20%D0%92%D0%B8%D0%B4%D1%8B%20%D0%B2%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D1%8B%D1%85%20%D1%80%D1%8F%D0%B4%D0%BE%D0%B2.ipynb) |
| 3 | [Характеристики временного ряда](part_1/3.%20%D0%A5%D0%B0%D1%80%D0%B0%D0%BA%D1%82%D0%B5%D1%80%D0%B8%D1%81%D1%82%D0%B8%D0%BA%D0%B8%20%D0%B2%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D0%BE%D0%B3%D0%BE%20%D1%80%D1%8F%D0%B4%D0%B0.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_1/part_1/3.%20%D0%A5%D0%B0%D1%80%D0%B0%D0%BA%D1%82%D0%B5%D1%80%D0%B8%D1%81%D1%82%D0%B8%D0%BA%D0%B8%20%D0%B2%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D0%BE%D0%B3%D0%BE%20%D1%80%D1%8F%D0%B4%D0%B0.ipynb) |

### Часть 2. Библиотеки и данные

| № | Ноутбук | Colab |
|---|---------|-------|
| 1 | [`statsmodels` — данные и визуализация](part_2/1.%20statsmodels%20%E2%80%94%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5%20%D0%B8%20%D0%B2%D0%B8%D0%B7%D1%83%D0%B0%D0%BB%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_1/part_2/1.%20statsmodels%20%E2%80%94%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5%20%D0%B8%20%D0%B2%D0%B8%D0%B7%D1%83%D0%B0%D0%BB%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F.ipynb) |
| 2 | [`statsforecast` — данные и визуализация](part_2/2.%20statsforecast%20%E2%80%94%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5%20%D0%B8%20%D0%B2%D0%B8%D0%B7%D1%83%D0%B0%D0%BB%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_1/part_2/2.%20statsforecast%20%E2%80%94%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5%20%D0%B8%20%D0%B2%D0%B8%D0%B7%D1%83%D0%B0%D0%BB%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F.ipynb) |
| 3 | [Открытые источники данных временных рядов](part_2/3.%20%D0%9E%D1%82%D0%BA%D1%80%D1%8B%D1%82%D1%8B%D0%B5%20%D0%B8%D1%81%D1%82%D0%BE%D1%87%D0%BD%D0%B8%D0%BA%D0%B8%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85%20%D0%B2%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D1%8B%D1%85%20%D1%80%D1%8F%D0%B4%D0%BE%D0%B2.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_1/part_2/3.%20%D0%9E%D1%82%D0%BA%D1%80%D1%8B%D1%82%D1%8B%D0%B5%20%D0%B8%D1%81%D1%82%D0%BE%D1%87%D0%BD%D0%B8%D0%BA%D0%B8%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85%20%D0%B2%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D1%8B%D1%85%20%D1%80%D1%8F%D0%B4%D0%BE%D0%B2.ipynb) |
| 4 | [`sktime` — данные и визуализация](part_2/4.%20sktime%20%E2%80%94%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5%20%D0%B8%20%D0%B2%D0%B8%D0%B7%D1%83%D0%B0%D0%BB%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-hse-repository/time-series-cu-2026/blob/main/week_1/part_2/4.%20sktime%20%E2%80%94%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5%20%D0%B8%20%D0%B2%D0%B8%D0%B7%D1%83%D0%B0%D0%BB%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F.ipynb) |


---

## Локальный запуск

### 0. Установить uv

```bash
# macOS / Linux
pip install uv

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Проверка: `uv --version`.

### 1. Создать виртуальное окружение и установить зависимости

Из каталога `week_1` (там лежат `pyproject.toml` и `uv.lock`):

```bash
cd week_1
uv sync
```

`uv sync` сам создаст `.venv` с нужной версией Python (проект требует `>=3.11`) и поставит зависимости строго по `uv.lock` — у всех окружение будет одинаковым.

Если нужной версии Python в системе нет, uv скачает её сам; можно указать версию явно:

```bash
uv python install 3.11
uv venv --python 3.11
uv sync
```

### 2. Зарегистрировать Jupyter kernel

`ipykernel` уже входит в зависимости, поэтому достаточно зарегистрировать окружение как kernel:

```bash
uv run python -m ipykernel install --user --name ts-cu-2026-week1 --display-name "Python (ts-cu-2026 · week 1)"
```

- `--name` — внутренний идентификатор (без пробелов),
- `--display-name` — то, что будет видно в списке kernel'ов Jupyter,
- `--user` — установка в домашний каталог, права администратора не нужны.

Проверить, что kernel появился:

```bash
uv run jupyter kernelspec list
```

### 3. Запустить Jupyter

```bash
uv run --with jupyterlab jupyter lab
```

Либо, если JupyterLab уже установлен в системе (`pipx install jupyterlab` / отдельное окружение) — запустите его как обычно: зарегистрированный на шаге 2 kernel будет доступен любому Jupyter на машине.

В открытом ноутбуке выберите kernel **Python (ts-cu-2026 · week 1)**: меню *Kernel → Change Kernel…*, в VS Code / PyCharm — селектор kernel'а в правом верхнем углу.

### Проверка окружения

```bash
uv run python -c "import pandas, statsmodels, statsforecast, sktime; print('ok')"
```

### Удалить kernel (если больше не нужен)

```bash
uv run jupyter kernelspec uninstall ts-cu-2026-week1
```

---

## Запуск в Google Colab

Нажмите бейдж <img src="https://colab.research.google.com/assets/colab-badge.svg" height="18"> в таблице выше — ноутбук откроется прямо из репозитория.

**TODO**: установка зависимостей в colab

После установки может потребоваться перезапуск среды (*Runtime → Restart session*).

---

## Зависимости

Основное (полный список — в [`pyproject.toml`](pyproject.toml), зафиксированные версии — в `uv.lock`):

| Библиотека | Зачем |
|---|---|
| `pandas`, `numpy` | базовая работа с рядами и датами |
| `matplotlib`, `seaborn`, `plotly` | визуализация |
| `statsmodels` | декомпозиция, ACF/PACF, тесты на стационарность |
| `statsforecast` | быстрое прогнозирование множества рядов (Nixtla) |
| `sktime` | единый scikit-learn-подобный интерфейс |
| `scikit-learn`, `scipy` | вспомогательные преобразования и статистика |
| `holidays` | календарные признаки |
| `datasetsforecast`, `kagglehub`, `datasets`, `requests` | загрузка открытых датасетов |
| `ipykernel`, `ipywidgets` | работа в Jupyter |
