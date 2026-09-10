# Allure Failure Clustering

Скрипт группирует упавшие тесты из Allure-отчёта по причинам падения.

Он читает файлы `*-result.json` из папки `allure-results`, оставляет только тесты
со статусом `failed` / `broken`, вычищает из текста ошибки динамику (UUID,
таймстампы, длительности, длинные числа) и собирает падения с одинаковой причиной
в один кластер. Для каждого кластера показывает, сколько тестов в него попало,
типовой текст ошибки и шаг, на котором тесты чаще всего падают.

## Что нужно установить

- **Python 3.9+** (проверить: `python3 --version`)
- Python-пакеты из [requirements.txt](requirements.txt): `scikit-learn`, `numpy`

## Установка (один раз)

Открой терминал и перейди в папку проекта:

```bash
cd /Users/geras/Desktop/Clusterization
```

Создай виртуальное окружение и поставь зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

После этого в начале строки терминала появится `(.venv)` — значит окружение
активно.

## Запуск

> Каждый раз в новом терминале сначала активируй окружение:
> `cd /Users/geras/Desktop/Clusterization && source .venv/bin/activate`

### Базовый режим (по умолчанию, `exact`)

Точное совпадение нормализованного текста ошибки. Быстро и предсказуемо.

```bash
python allure_failure_clustering.py allure-results
```

### ML-режим (`ml`)

TF-IDF + кластеризация HDBSCAN. Схлопывает похожие по смыслу, но по-разному
сформулированные ошибки (разные обёртки одного исключения, разная пунктуация),
чего `exact` не умеет.

```bash
python allure_failure_clustering.py allure-results --mode ml
```

### HTML-дашборд

К любому режиму можно добавить `--html`, чтобы получить наглядный отчёт
(самодостаточный HTML, без интернета) с диаграммой и раскрывающимися кластерами:

```bash 
python allure_failure_clustering.py allure-results --mode ml --html report.html
```

Потом открой `report.html` двойным кликом в браузере.

## Все параметры

| Параметр               | Значение по умолчанию | Описание                                                        |
| ---------------------- | --------------------- | ------------------------------------------------------------- |
| `results_dir`          | — (обязательный)      | Путь к папке с результатами Allure, например `allure-results` |
| `--mode {exact,ml}`    | `exact`               | Способ группировки                                            |
| `--min-cluster-size N` | `2`                   | Минимальный размер кластера для режима `ml`                   |
| `--html PATH`          | не создаётся          | Куда сохранить HTML-дашборд, например `report.html`          |

Справка в терминале:

```bash
python allure_failure_clustering.py --help
```

## Пример вывода

```
Всего упавших тестов: 12
Уникальных кластеров причин: 3

Кластер 1 (7 тестов)
  Паттерн ошибки: Condition with alias 'transaction in topic' didn't complete ...
  Чаще всего падает на шаге: Проверка транзакции в топике
  Примеры тестов: test_payment_flow, test_refund, test_hold

Кластер 2 (3 теста)
  ...
```

## Если что-то не работает

- **`command not found: python3`** — установи Python с [python.org](https://www.python.org/downloads/)
  или через `brew install python`.
- **`ModuleNotFoundError: No module named 'sklearn'`** — не активировано окружение
  или не установлены зависимости. Повтори шаги из раздела «Установка».
- **`Папка не найдена: allure-results`** — запускай скрипт из корня проекта либо
  укажи полный путь к папке с результатами.
- **`Упавших тестов не найдено`** — в папке нет файлов `*-result.json` со статусом
  `failed` / `broken`.
