# AI Paper Researcher

План проекта: [PLAN.md](PLAN.md). Исходники редактируются локально; запуск,
сборка Docker-образа и проверки с GPU выполняются на сервере.

## Почему выбран этот Docker-вариант

Основа — подход из `SampledSoftmax/LimitedMemoryRule`: зависимости устанавливаются
при сборке образа. Из VLM4TS сохранён подход с монтированием репозитория в `/app`.
Файл `credentials` не нужен: путь к проекту и UID/GID определяются скриптами.

Сервер: четыре GTX 1080 Ti, по 11 GiB, драйвер NVIDIA 580.178.04.
Используется образ `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` с Python 3.11.
`llama-cpp-python` собирается с CUDA под `sm_61` — архитектуру GTX 1080 Ti.
CUDA 13 в выводе `nvidia-smi` обозначает возможности драйвера; для этих Pascal
GPU нужна сборка с CUDA 12.x. PyTorch также закреплён, чтобы обновление пакетов
не заменило его сборкой без поддержки этих GPU.

Справочники: [архитектуры NVIDIA](https://developer.nvidia.com/cuda/gpus/legacy),
[поддержка Pascal в CUDA](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers),
[версии PyTorch](https://pytorch.org/get-started/previous-versions/),
[сборка llama-cpp-python с CUDA](https://llama-cpp-python.readthedocs.io/en/latest/#supported-backends).

## Первый запуск на сервере

Нужны Docker с BuildKit, NVIDIA-драйвер и NVIDIA Container Toolkit.
Скопируй/клонируй весь проект на сервер. Название каталога на сервере произвольное;
в контейнере он всегда доступен как `/app`, а контейнер называется
`ai-paper-researcher`.

Из каталога проекта на сервере:

```bash
./build_image
./launch_container
docker exec -it ai-paper-researcher bash
```

Сборка скачивает большой CUDA/PyTorch-образ, устанавливает зависимости и компилирует
GGUF runtime. Первая сборка занимает время; следующие используют кеш слоёв.
Веса моделей при сборке и запуске контейнера автоматически не скачиваются.

`build_image` по умолчанию показывает полный лог (`--progress=plain`). Скачивание
архива `llama-cpp-python` выделено в отдельный слой перед компиляцией. CMake,
Ninja и scikit-build-core установлены заранее; `--no-build-isolation` позволяет
использовать именно их. При компиляции pip работает с локальным архивом, с
отключённым скачиванием пакетов (`--no-index --no-deps`). Сетевой таймаут pip —
30 секунд бездействия, число повторов — 2; это не ограничение общей длительности
скачивания. Ошибка скачивания и долгая компиляция теперь видны отдельными шагами.

Внутри контейнера:

```bash
pwd
python --version
python -c 'from llama_cpp import Llama; print("Llama import: OK")'
python -c 'import torch; assert torch.cuda.is_available(); x = torch.ones(2, device="cuda"); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), x + x)'
python -c 'from llama_cpp import llama_supports_gpu_offload; assert llama_supports_gpu_offload(); print("GGUF GPU backend: OK")'
```

Последние две команды проверяют разные библиотеки: успешная работа PyTorch сама
по себе не подтверждает CUDA-сборку `llama-cpp-python`. Окончательная проверка
инференса — реальная загрузка весов и генерация через `smoke_llm.py`.

Скачивание GGUF и запуск проверки, также внутри контейнера:

```bash
hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir /app/models
python scripts/smoke_llm.py
```

Текущий путь `./models/Qwen3-8B-Q4_K_M.gguf` в скрипте работает при запуске из `/app`.

## Что сохраняется и когда нужна пересборка

Весь каталог проекта примонтирован в `/app`. Поэтому исходники, `models/`,
`data/`, `artifacts/` и `.cache/` остаются на сервере после удаления контейнера.
Python-окружение находится в `/opt/venv` внутри образа и не скрывается этим mount.
Не синхронизируй локальные virtualenv и содержимое `models/` поверх серверных
каталогов с удалением файлов.

После изменения исходников достаточно передать изменения на сервер и снова
запустить скрипт. Контейнер и образ пересоздавать не нужно.
`./launch_container` повторно использует запущенный контейнер или запускает
остановленный. Выход из `docker exec` оставляет контейнер работать.

Если меняются зависимости, обнови `requirements.in`, пересоздай `requirements.txt`
и собери образ. Существующий контейнер продолжает использовать прежний образ.
Чтобы применить новый образ, после завершения текущих задач выполни на сервере:

```bash
docker stop ai-paper-researcher
docker rm ai-paper-researcher
./launch_container
```

Файлы в `/app` сохранятся. Пакеты, установленные вручную только внутри старого
контейнера, в новый контейнер не попадут.

## Зависимости

В образ заранее включены зависимости для smoke-проверки и запланированных этапов:

| Задача | Основные библиотеки |
|---|---|
| GGUF и скачивание весов | llama-cpp-python, huggingface-hub |
| Embeddings | PyTorch, sentence-transformers, transformers |
| API и настройки | FastAPI, Uvicorn, Pydantic, pydantic-settings |
| HTTP и поиск статей | HTTPX, arxiv |
| PDF | PyMuPDF |
| База и миграции | SQLAlchemy, Alembic, psycopg, pgvector |
| Агенты и графы | LangChain, LangGraph, интеграции Hugging Face/OpenAI |
| Тесты и разработка | pytest, pytest-asyncio, Ruff, ipykernel |

Пакеты `psycopg` и `pgvector` — Python-клиенты. Сам PostgreSQL с расширением
pgvector потребуется запустить отдельным сервисом на этапе сохранения статей.
FastAPI-сервис пока не реализован и автоматически не запускается.

`requirements.in` содержит понятный список выбранных зависимостей и границы
версий; `requirements.txt` — согласованные точные версии всех зависимостей для
Linux x86_64 / Python 3.11. Он проверен резолвером; реальная сборка и GPU-проверки
должны выполняться на сервере.

Для обновления фиксации версий с установленным `uv`:

```bash
uv pip compile requirements.in --python-version 3.11 --python-platform x86_64-unknown-linux-gnu --index-url https://pypi.org/simple --no-header --no-annotate --output-file requirements.txt
```

При изменении версии `llama-cpp-python` обнови её и в `requirements.in`, и в
соответствующем слое Dockerfile, затем пересоздай `requirements.txt`.

## Параметры запуска

По умолчанию используется физическая GPU 0. Можно выбрать другую или несколько:

```bash
GPU_DEVICES=1 ./launch_container
GPU_DEVICES=0,1 ./launch_container
GPU_DEVICES=all ./launch_container
```

Внутри процесса единственная выбранная карта будет доступна как `cuda:0`.
Выбор нескольких карт делает их доступными библиотекам; фактическое распределение
модели зависит от параметров runtime.
Параметры применяются при создании контейнера, а не при повторном использовании.

Дополнительные настройки задаются перед командой:

```bash
BUILD_JOBS=2 ./build_image
API_PORT=8001 CPU_LIMIT=8 MEMORY_LIMIT=32g ./launch_container
```

`BUILD_JOBS` ограничивает параллельность компиляции; `SHM_SIZE` по умолчанию `2g`.
Жёсткие лимиты CPU/RAM по умолчанию не заданы. API-порт 8000 контейнера проброшен
на `127.0.0.1:8000` сервера. Для доступа с локальной машины можно использовать SSH
port forwarding; будущий Uvicorn должен слушать `0.0.0.0:8000` внутри контейнера.
`DOCKER_NETWORK` позволяет подключить контейнер к уже созданной Docker-сети,
например для PostgreSQL. Если есть `.env`, его значения передаются приложению
через `--env-file`; этот файл не используется как shell-скрипт.
