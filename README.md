# Прогнозирование временных рядов (2026)

Материалы курса по прогнозированию временных рядов: ноутбуки семинаров, сгруппированные по занятиям.

## Занятия

| № | Занятие | Материалы |
|---|---------|-----------|
| 1 | Введение в анализ временных рядов | [`week_1`](week_1/README.md) |
| 2 | Базовые методы прогнозирования и оценка качества | [`week_2`](week_2/README.md) |

Структура каталога занятия одинаковая: `part_1/`, `part_2/` с ноутбуками, свои `pyproject.toml` и `uv.lock`, `README.md` со списком ноутбуков и ссылками на Colab.

Ноутбуки можно запускать двумя способами: **локально** (нужен Python 3.11+ и [uv](https://docs.astral.sh/uv/)) или **в Google Colab** — по ссылкам-бейджам из таблиц в README занятия, без установки чего-либо.

Далее везде `week_N` — каталог нужного занятия (`week_1`, `week_2`, …). Окружения занятий независимы: для каждого занятия шаги повторяются в его каталоге.

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

Из каталога занятия (там лежат `pyproject.toml` и `uv.lock`):

```bash
cd week_N
uv sync
```

`uv sync` сам создаст `.venv` с нужной версией Python (проекты требуют `>=3.11`) и поставит зависимости строго по `uv.lock` — у всех окружение будет одинаковым.

Если нужной версии Python в системе нет, uv скачает её сам; можно указать версию явно:

```bash
uv python install 3.11
uv venv --python 3.11
uv sync
```

### 2. Зарегистрировать Jupyter kernel

`ipykernel` уже входит в зависимости, поэтому достаточно зарегистрировать окружение как kernel (из каталога занятия):

```bash
uv run python -m ipykernel install --user --name ts-cu-2026-weekN --display-name "Python (ts-cu-2026 · week N)"
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

В открытом ноутбуке выберите kernel **Python (ts-cu-2026 · week N)**: меню *Kernel → Change Kernel…*, в VS Code / PyCharm — селектор kernel'а в правом верхнем углу.

### Проверка окружения

```bash
uv run python -c "import pandas, statsmodels, statsforecast, sktime; print('ok')"
```

### Удалить kernel (если больше не нужен)

```bash
uv run jupyter kernelspec uninstall ts-cu-2026-weekN
```

---

## Запуск в Google Colab

Нажмите бейдж <img src="https://colab.research.google.com/assets/colab-badge.svg" height="18"> в таблице ноутбуков в README нужного занятия — ноутбук откроется прямо из репозитория.

После установки зависимостей(первая ячейка) может потребоваться перезапуск среды (*Runtime → Restart session*).

---
