"""
Группировка причин падений тестов по данным Allure.

Режим по умолчанию (--mode exact): точное совпадение нормализованного
(regex) сообщения об ошибке — без ML, быстро и предсказуемо.

Режим --mode ml: TF-IDF векторизация нормализованных сообщений +
кластеризация HDBSCAN. Группирует семантически похожие, но текстуально
разные сообщения (разные обёртки одного исключения, разная пунктуация,
частично разный текст) — то, что exact-режим не схлопнёт.

Использование:
    python allure_failure_clustering.py /path/to/allure-results
    python allure_failure_clustering.py /path/to/allure-results --mode ml
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# 1. Парсинг Allure JSON
# ---------------------------------------------------------------------------

class TestFailure:
    def __init__(self, test_name: str, status: str, message: str,
                 trace: str, failed_step: Optional[str], start: int):
        self.test_name = test_name
        self.status = status          # failed / broken
        self.message = message
        self.trace = trace
        self.failed_step = failed_step
        self.start = start


def find_failed_step(steps: list) -> Optional[str]:
    """Рекурсивно находит имя первого упавшего шага (создание фигуранта /
    отправка транзакции / проверка в топике и т.д.)."""
    for step in steps:
        if step.get("status") in ("failed", "broken"):
            # спускаемся глубже, если есть вложенные шаги
            nested = step.get("steps", [])
            deeper = find_failed_step(nested) if nested else None
            return deeper or step.get("name")
    return None


def load_failures(results_dir: Path) -> list[TestFailure]:
    """Читает все *-result.json из папки allure-results и оставляет
    только failed/broken тесты."""
    failures = []

    for path in results_dir.glob("*-result.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        status = data.get("status")
        if status not in ("failed", "broken"):
            continue

        details = data.get("statusDetails", {}) or {}
        message = details.get("message", "") or ""
        trace = details.get("trace", "") or ""
        steps = data.get("steps", []) or []

        failures.append(TestFailure(
            test_name=data.get("name", path.stem),
            status=status,
            message=message,
            trace=trace,
            failed_step=find_failed_step(steps),
            start=data.get("start", 0),
        ))

    return failures


# ---------------------------------------------------------------------------
# 2. Нормализация текста ошибки
# ---------------------------------------------------------------------------

# Каждый паттерн: (regex, замена). Порядок важен — сначала более специфичные.
NORMALIZATION_RULES = [
    (re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"), "<UUID>"),
    (re.compile(r"\btransactionId\s*=\s*\S+", re.IGNORECASE), "transactionId=<ID>"),
    (re.compile(r"\bsubjectId\s*=\s*\S+", re.IGNORECASE), "subjectId=<ID>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?"), "<TIMESTAMP>"),
    (re.compile(r"\bafter\s+\d+\s*ms\b", re.IGNORECASE), "after <N>ms"),
    (re.compile(r"\b\d+\s*ms\b"), "<N>ms"),
    (re.compile(r"\b\d+\.\d+\b"), "<NUM>"),   # суммы, score
    (re.compile(r"\b\d{5,}\b"), "<NUM>"),      # длинные числа (id, суммы)
]


def normalize_message(message: str) -> str:
    """Убирает динамические части (id, timestamp, длительность), чтобы
    сообщения об одной и той же причине совпадали текстуально."""
    normalized = message.strip()
    for pattern, replacement in NORMALIZATION_RULES:
        normalized = pattern.sub(replacement, normalized)
    # схлопываем лишние пробелы
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


# ---------------------------------------------------------------------------
# 3a. Группировка — точное совпадение (без ML)
# ---------------------------------------------------------------------------

def group_failures_exact(failures: list[TestFailure]) -> dict[str, list[TestFailure]]:
    groups: dict[str, list[TestFailure]] = defaultdict(list)
    for f in failures:
        key = normalize_message(f.message) or "<пустое сообщение об ошибке>"
        groups[key].append(f)
    return groups


# ---------------------------------------------------------------------------
# 3b. Группировка — TF-IDF + HDBSCAN (ML)
# ---------------------------------------------------------------------------

def group_failures_ml(
    failures: list[TestFailure],
    min_cluster_size: int = 2,
) -> dict[str, list[TestFailure]]:
    """Векторизует нормализованные сообщения через TF-IDF (по словам и
    символьным n-граммам — полезно для стектрейсов, где важны куски вроде
    'Timeout', 'ConnectException') и кластеризует HDBSCAN'ом.

    HDBSCAN выбран вместо k-means, потому что заранее не известно число
    "настоящих причин" падений, и он умеет помечать несхожие сообщения
    как шум (label -1), а не пихать их в ближайший кластер силой.
    """
    normalized = [normalize_message(f.message) or "<empty>" for f in failures]

    if len(normalized) < min_cluster_size:
        # тестов слишком мало для осмысленной кластеризации
        return {"Кластер 1 (мало данных для ML)": failures}

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(normalized)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="cosine",
        store_centers="medoid",
        copy=True,
    )
    labels = clusterer.fit_predict(matrix.toarray())

    groups: dict[str, list[TestFailure]] = defaultdict(list)
    label_to_examples: dict[int, list[str]] = defaultdict(list)

    for f, msg, label in zip(failures, normalized, labels):
        label_to_examples[label].append(msg)

    for f, msg, label in zip(failures, normalized, labels):
        if label == -1:
            # шум — по одному "кластеру" на уникальное сообщение,
            # чтобы не сваливать разнородные единичные ошибки в одну кучу
            key = f"[уникальная] {msg[:150]}"
        else:
            # имя кластера — самое частое нормализованное сообщение внутри него
            examples = label_to_examples[label]
            representative = max(set(examples), key=examples.count)
            key = f"[кластер {label}] {representative[:150]}"
        groups[key].append(f)

    return groups


# ---------------------------------------------------------------------------
# 5. HTML-дашборд (self-contained, без внешних CDN — работает офлайн)
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """Экранирование для безопасной вставки в HTML."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _build_bar_chart_svg(sorted_groups: list, max_bars: int = 15) -> str:
    """Простой горизонтальный bar chart, целиком нарисованный в SVG.
    Не тянет никаких JS-библиотек с CDN — важно для офлайн-агентов."""
    top = sorted_groups[:max_bars]
    if not top:
        return "<p>Нет данных для графика.</p>"

    max_count = max(len(items) for _, items in top)
    bar_h = 28
    gap = 10
    label_w = 420
    chart_w = 420
    row_h = bar_h + gap
    svg_h = row_h * len(top) + gap
    svg_w = label_w + chart_w + 60

    bars = []
    for i, (pattern, items) in enumerate(top):
        y = gap + i * row_h
        count = len(items)
        bar_len = int((count / max_count) * (chart_w - 10)) if max_count else 0
        label = pattern if len(pattern) <= 60 else pattern[:57] + "..."
        bars.append(f'''
        <text x="{label_w - 10}" y="{y + bar_h / 2 + 5}" text-anchor="end"
              font-size="12" fill="var(--fg,#333)">{_esc(label)}</text>
        <rect x="{label_w}" y="{y}" width="{bar_len}" height="{bar_h}"
              rx="4" fill="var(--bar,#4f7cff)" />
        <text x="{label_w + bar_len + 8}" y="{y + bar_h / 2 + 5}"
              font-size="12" fill="var(--fg,#333)">{count}</text>
        ''')

    return f'''
    <svg viewBox="0 0 {svg_w} {svg_h}" width="100%" style="max-width:900px">
        {''.join(bars)}
    </svg>
    '''


def build_dashboard_html(groups: dict, mode: str, results_dir: str) -> str:
    sorted_groups = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    total = sum(len(v) for v in groups.values())
    chart_svg = _build_bar_chart_svg(sorted_groups)

    rows = []
    for i, (pattern, items) in enumerate(sorted_groups, start=1):
        steps = [f.failed_step for f in items if f.failed_step]
        top_step = max(set(steps), key=steps.count) if steps else "—"
        test_list = "".join(f"<li>{_esc(f.test_name)}</li>" for f in items)
        rows.append(f'''
        <details class="cluster">
            <summary>
                <span class="badge">{len(items)}</span>
                <span class="pattern">{_esc(pattern[:200])}</span>
                <span class="step">чаще падает на: {_esc(top_step)}</span>
            </summary>
            <ul>{test_list}</ul>
        </details>
        ''')

    return f'''<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Allure Failure Clustering — отчёт</title>
<style>
    :root {{ --fg:#222; --bar:#4f7cff; --bg:#fafafa; --card:#fff; --border:#e3e3e8; }}
    body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:var(--bg); color:var(--fg); margin:0; padding:32px; }}
    h1 {{ font-size:20px; margin-bottom:4px; }}
    .meta {{ color:#777; font-size:13px; margin-bottom:24px; }}
    .card {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:20px; margin-bottom:20px; }}
    .cluster {{ border-bottom:1px solid var(--border); padding:10px 0; }}
    .cluster:last-child {{ border-bottom:none; }}
    summary {{ cursor:pointer; display:flex; gap:12px; align-items:center; font-size:14px; }}
    summary::-webkit-details-marker {{ color:#999; }}
    .badge {{ background:var(--bar); color:#fff; border-radius:12px; padding:2px 10px; font-size:12px; font-weight:600; min-width:24px; text-align:center; }}
    .pattern {{ flex:1; font-family: ui-monospace, Menlo, monospace; font-size:13px; }}
    .step {{ color:#888; font-size:12px; white-space:nowrap; }}
    ul {{ margin:8px 0 4px 40px; padding:0; font-size:13px; color:#555; }}
</style>
</head>
<body>
    <h1>Allure Failure Clustering</h1>
    <div class="meta">
        Источник: {_esc(results_dir)} &middot; режим: {_esc(mode)} &middot;
        всего упавших тестов: {total} &middot; кластеров: {len(groups)}
    </div>

    <div class="card">
        {chart_svg}
    </div>

    <div class="card">
        {''.join(rows)}
    </div>
</body>
</html>
'''


# ---------------------------------------------------------------------------
# 6. Отчёт (консоль)
# ---------------------------------------------------------------------------

def print_report(groups: dict[str, list[TestFailure]]) -> None:
    # сортируем кластеры по размеру, самые массовые — первыми
    sorted_groups = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)

    total = sum(len(v) for v in groups.values())
    print(f"Всего упавших тестов: {total}")
    print(f"Уникальных кластеров причин: {len(groups)}\n")

    for i, (pattern, items) in enumerate(sorted_groups, start=1):
        print(f"Кластер {i} ({len(items)} тестов)")
        print(f"  Паттерн ошибки: {pattern[:200]}")

        # чаще всего упавший шаг в этом кластере — хороший намёк на причину
        steps = [f.failed_step for f in items if f.failed_step]
        if steps:
            most_common_step = max(set(steps), key=steps.count)
            print(f"  Чаще всего падает на шаге: {most_common_step}")

        example_names = [f.test_name for f in items[:3]]
        print(f"  Примеры тестов: {', '.join(example_names)}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Группировка причин падений тестов по Allure-результатам")
    parser.add_argument("results_dir", type=Path, help="Путь к папке allure-results")
    parser.add_argument(
        "--mode",
        choices=["exact", "ml"],
        default="exact",
        help="exact — точное совпадение нормализованного текста (по умолчанию); "
             "ml — TF-IDF + HDBSCAN кластеризация",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Минимальный размер кластера для режима ml (по умолчанию 2)",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Путь для сохранения HTML-дашборда (например, report.html). "
             "Если не указан, дашборд не создаётся — только консольный вывод.",
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        print(f"Папка не найдена: {args.results_dir}")
        sys.exit(1)

    failures = load_failures(args.results_dir)
    if not failures:
        print("Упавших тестов не найдено (или папка не содержит allure-results).")
        return

    if args.mode == "ml":
        groups = group_failures_ml(failures, min_cluster_size=args.min_cluster_size)
    else:
        groups = group_failures_exact(failures)

    print_report(groups)

    if args.html:
        html = build_dashboard_html(groups, mode=args.mode, results_dir=str(args.results_dir))
        args.html.write_text(html, encoding="utf-8")
        print(f"HTML-дашборд сохранён: {args.html.resolve()}")


if __name__ == "__main__":
    main()
