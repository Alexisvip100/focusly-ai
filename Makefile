.PHONY: help setup dev dev-reload compile format lint pre-commit test test_only docker-build docker-up docker-down docker-logs clean env

help: ## Muestra este mensaje de ayuda
	@echo "Comandos disponibles en Focusly AI:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

env: ## Crea el archivo .env a partir de .env.example (solo si no existe)
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo ".env creado desde .env.example — recuerda añadir tus API keys."; \
	else \
		echo ".env ya existe, no se sobreescribió."; \
	fi

setup: ## Crea el entorno virtual e instala dependencias de producción y desarrollo
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements-dev.txt
	./venv/bin/pre-commit install
	@echo "Setup completado. Entorno listo."

dev: ## Inicia el servidor de desarrollo FastAPI (sin recarga en vivo)
	./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8001

dev-reload: ## Inicia el servidor de desarrollo FastAPI con recarga en vivo
	./venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

compile: ## Verifica la sintaxis del código con comprobación estática de tipos (Mypy)
	./venv/bin/mypy app

format: ## Formatea y aplica auto-correcciones al código con Ruff
	./venv/bin/ruff format app
	./venv/bin/ruff check --fix app

lint: ## Revisa el estilo de código sin modificar archivos
	./venv/bin/ruff check app
	./venv/bin/mypy app

pre-commit: ## Ejecuta todas las validaciones de git pre-commit
	./venv/bin/pre-commit run --all-files

test: ## Ejecuta la suite completa de pruebas unitarias
	./venv/bin/pytest

test_only: ## Ejecuta solo un archivo o patrón de prueba (ej. make test_only FILE=test_health.py o make test_only K=test_name)
	@if [ -z "$(FILE)" ] && [ -z "$(K)" ]; then \
		echo "Error: Especifica FILE=<archivo> o K=<filtro>. Ej: make test_only FILE=test_health.py"; \
		exit 1; \
	fi
	@if [ -n "$(FILE)" ]; then \
		./venv/bin/pytest $(FILE); \
	elif [ -n "$(K)" ]; then \
		./venv/bin/pytest -k "$(K)"; \
	fi

docker-build: ## Construye la imagen Docker del servicio
	docker compose build

docker-up: ## Inicia el servicio en Docker en segundo plano
	docker compose up -d

docker-down: ## Detiene y elimina los contenedores Docker
	docker compose down

docker-logs: ## Muestra los logs del contenedor en tiempo real
	docker compose logs -f focusly-ai

clean: ## Elimina cachés de Python y archivos temporales
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Limpieza completada."
