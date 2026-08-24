# Запуск линтера и форматтера по всему репозиторию.
# Правила и исключения — в `ruff.toml` в корне; ноутбуки ruff обрабатывает нативно.
#
# Если ruff есть в PATH, используется он; иначе нужную версию скачает uv (`uvx`),
# ставить ruff в окружение занятия не требуется. Обе части переопределяются:
#
#     make check RUFF="uvx ruff@0.13.0"   # версия как в CI, минуя системный ruff
#     make check RUFF=./.venv/bin/ruff    # ruff из конкретного окружения

RUFF_VERSION ?= 0.13.0
RUFF ?= $(shell command -v ruff >/dev/null 2>&1 && echo ruff || echo uvx ruff@$(RUFF_VERSION))

.DEFAULT_GOAL := help

.PHONY: help
help: ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: check
check: lint format-check ## Все проверки без правок — то же, что делает CI

.PHONY: lint
lint: ## Проверить код линтером
	$(RUFF) check

.PHONY: format-check
format-check: ## Проверить форматирование, ничего не меняя
	$(RUFF) format --check --diff

.PHONY: fix
fix: ## Починить замечания линтера
	$(RUFF) check --fix

.PHONY: format
format: ## Отформатировать ячейки с кодом и скрипты
	$(RUFF) format

.PHONY: fmt
fmt: fix format ## Починить и отформатировать за один заход
