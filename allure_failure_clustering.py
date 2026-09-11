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
import base64
import io
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # без GUI-бэкенда — рендерим только в файл/буфер
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# 1. Парсинг Allure JSON
# ---------------------------------------------------------------------------

class TestFailure:
    def __init__(self, test_name: str, status: str, message: str,
                 trace: str, failed_step: Optional[str], start: int,
                 retries: int = 0):
        self.test_name = test_name
        self.status = status          # failed / broken
        self.message = message
        self.trace = trace
        self.failed_step = failed_step
        self.start = start
        # для flaky-тестов — сколько раз тест упал, прежде чем пройти
        # (0 для обычных, "окончательно упавших" тестов)
        self.retries = retries


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


def load_all_attempts(results_dir: Path) -> list[dict]:
    """Читает ВСЕ *-result.json (независимо от статуса) — нужно, чтобы
    видеть все попытки теста при ретраях, а не только упавшие."""
    records = []

    for path in results_dir.glob("*-result.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        # historyId — стабильный идентификатор теста у Allure, одинаковый
        # у всех попыток одного и того же теста (в т.ч. при ретраях).
        # Если его почему-то нет (старый адаптер/кастомный раннер) —
        # откатываемся на имя теста, это хуже, но не ломает пайплайн.
        history_id = data.get("historyId") or data.get("name")

        records.append({
            "history_id": history_id,
            "name": data.get("name", path.stem),
            "status": data.get("status"),
            "message": (data.get("statusDetails", {}) or {}).get("message", "") or "",
            "trace": (data.get("statusDetails", {}) or {}).get("trace", "") or "",
            "steps": data.get("steps", []) or [],
            "start": data.get("start", 0),
        })

    return records


def partition_flaky(records: list[dict]) -> tuple[list[TestFailure], list[TestFailure]]:
    """Группирует попытки по historyId и разделяет на:
      - genuine_failures — тесты, у которых ВСЕ попытки упали (реальная
        проблема, не флейки). Берём последнюю по времени попытку —
        это финальное состояние теста.
      - flaky_failures — тесты, у которых есть и failed/broken, и passed
        попытки: упали, но в итоге прошли на ретрае. Для кластеризации
        причины берём их ПЕРВУЮ по времени упавшую попытку — именно она
        объясняет, из-за чего тест изначально зашатался; каждой такой
        записи проставляем retries — сколько раз тест падал перед тем,
        как пройти.
    """
    by_history: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_history[r["history_id"]].append(r)

    genuine_failures = []
    flaky_failures = []

    for history_id, attempts in by_history.items():
        statuses = {a["status"] for a in attempts}

        has_passed = "passed" in statuses
        has_failed = bool(statuses & {"failed", "broken"})

        if has_passed and has_failed:
            # тест шатался: упал, потом прошёл (или наоборот) — флейки,
            # не ошибка в смысле кластеризации причин падений, но у него
            # есть своя причина первого падения — кластеризуем отдельно
            sorted_attempts = sorted(attempts, key=lambda a: a.get("start", 0))
            first_failure = next(
                a for a in sorted_attempts if a["status"] in ("failed", "broken")
            )
            fail_count = sum(1 for a in attempts if a["status"] in ("failed", "broken"))
            flaky_failures.append(TestFailure(
                test_name=first_failure["name"],
                status=first_failure["status"],
                message=first_failure["message"],
                trace=first_failure["trace"],
                failed_step=find_failed_step(first_failure["steps"]),
                start=first_failure["start"],
                retries=fail_count,
            ))
            continue

        if not has_failed:
            # все попытки passed (или другой нейтральный статус) —
            # тест просто прошёл, не нужен ни в failures, ни в flaky
            continue

        # все попытки упали — берём последнюю по времени как финальную
        last = max(attempts, key=lambda a: a.get("start", 0))
        genuine_failures.append(TestFailure(
            test_name=last["name"],
            status=last["status"],
            message=last["message"],
            trace=last["trace"],
            failed_step=find_failed_step(last["steps"]),
            start=last["start"],
        ))

    return genuine_failures, flaky_failures


def load_failures(results_dir: Path) -> list[TestFailure]:
    """Обратная совместимость: как раньше, но теперь под капотом
    исключает флейки-тесты (упал → потом прошёл на ретрае)."""
    records = load_all_attempts(results_dir)
    genuine_failures, _ = partition_flaky(records)
    return genuine_failures


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

    # max_df=0.95 при малом числе сообщений (частый случай для flaky —
    # их обычно меньше, чем "настоящих" падений) может вырезать ВСЕ
    # термы разом, если сообщения текстуально совпадают (каждый терм
    # встречается в 100% документов) — тогда TfidfVectorizer падает с
    # "no terms remain". Отключаем отсечение по частоте при малой
    # выборке, где оно бесполезно и опасно.
    max_df = 0.95 if len(normalized) >= 10 else 1.0

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_df=max_df,
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


def _build_bar_chart_png(sorted_groups: list, bar_color: str = "#4f7cff", max_bars: int = 15) -> str:
    """Горизонтальный bar chart через matplotlib, встроенный в HTML как
    base64 PNG. Не тянет никаких JS-библиотек с CDN — важно для
    офлайн-агентов, а сам график рисует matplotlib, а не рукописная
    SVG-разметка."""
    top = sorted_groups[:max_bars]
    if not top:
        return "<p>Нет данных для графика.</p>"

    # снизу вверх для barh — переворачиваем, чтобы самый большой кластер
    # оказался сверху, как в списке
    labels = [(p if len(p) <= 55 else p[:52] + "...") for p, _ in top][::-1]
    counts = [len(items) for _, items in top][::-1]

    fig_h = max(1.6, 0.5 * len(top) + 0.6)
    fig, ax = plt.subplots(figsize=(8, fig_h), dpi=150)
    bars = ax.barh(labels, counts, color=bar_color, height=0.6)

    max_count = max(counts)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max_count * 0.015, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=9, color="#333")

    ax.set_xlim(0, max_count * 1.15)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="Гистограмма кластеров" style="max-width:100%;height:auto">'


def _render_section(groups: dict, title: str, total_label: str, bar_color: str) -> str:
    """Рендерит один раздел дашборда (заголовок + график + список
    кластеров) — под конкретный набор групп (падения ИЛИ флейки).
    Несколько таких разделов можно поставить на одну страницу подряд —
    так на одном дашборде оказываются сразу два графика."""
    sorted_groups = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    total = sum(len(v) for v in groups.values())
    chart_img = _build_bar_chart_png(sorted_groups, bar_color=bar_color)

    rows = []
    for i, (pattern, items) in enumerate(sorted_groups, start=1):
        steps = [f.failed_step for f in items if f.failed_step]
        top_step = max(set(steps), key=steps.count) if steps else "—"
        test_list = "".join(f"<li>{_esc(_fmt_test_name(f))}</li>" for f in items)
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

    return f'''
    <section class="section" style="--bar:{bar_color}">
        <h2>{_esc(title)}</h2>
        <div class="meta">
            {_esc(total_label)}: {total} &middot; кластеров: {len(groups)}
        </div>

        <div class="card">
            {chart_img}
        </div>

        <div class="card">
            {''.join(rows) if rows else '<p>Нет данных.</p>'}
        </div>
    </section>
    '''


def build_dashboard_html(sections: list[dict], mode: str, results_dir: str) -> str:
    """Собирает страницу дашборда из одного или нескольких разделов.

    Каждый элемент sections — {"groups", "title", "total_label",
    "bar_color"}. Один раздел — как раньше, один график; два раздела
    (падения + флейки) — два графика на одной странице друг под другом,
    каждый со своим цветом.
    """
    rendered = "".join(
        _render_section(
            s["groups"], s["title"], s["total_label"],
            s.get("bar_color", "#4f7cff"),
        )
        for s in sections
    )

    return f'''<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Allure Failure Clustering — отчёт</title>
<style>
    :root {{ --fg:#222; --bar:#4f7cff; --bg:#fafafa; --card:#fff; --border:#e3e3e8; }}
    body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:var(--bg); color:var(--fg); margin:0; padding:32px; }}
    h1 {{ font-size:20px; margin-bottom:4px; }}
    h2 {{ font-size:16px; margin:0 0 4px; }}
    .top-meta {{ color:#777; font-size:13px; margin-bottom:24px; }}
    .section {{ margin-bottom:36px; }}
    .section:last-child {{ margin-bottom:0; }}
    .meta {{ color:#777; font-size:13px; margin-bottom:16px; }}
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
    <div class="top-meta">
        Источник: {_esc(results_dir)} &middot; режим: {_esc(mode)}
    </div>

    {rendered}
</body>
</html>
'''


# ---------------------------------------------------------------------------
# 6. JSON-экспорт (вход для run_summary.py)
# ---------------------------------------------------------------------------

def build_summary_json(groups: dict, mode: str, results_dir: str, kind: str = "failures") -> dict:
    """Компактное представление результата кластеризации — без полных
    трейсов и лишних деталей, чтобы легко передавать дальше (в скрипт
    сводки, в историю прогонов и т.д.)."""
    sorted_groups = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    total = sum(len(v) for v in groups.values())

    clusters = []
    for pattern, items in sorted_groups:
        steps = [f.failed_step for f in items if f.failed_step]
        top_step = max(set(steps), key=steps.count) if steps else None
        clusters.append({
            "pattern": pattern[:300],
            "count": len(items),
            "share": round(len(items) / total, 4) if total else 0,
            "most_common_failed_step": top_step,
            "example_tests": [f.test_name for f in items[:5]],
        })

    return {
        "generated_at": None,  # заполняется в main() реальным временем
        "results_dir": str(results_dir),
        "mode": mode,
        "kind": kind,          # "failures" — окончательные падения, "flaky" — упал → прошёл
        "total_failed": total,
        "cluster_count": len(groups),
        "clusters": clusters,
    }


# ---------------------------------------------------------------------------
# 7. Отчёт (консоль)
# ---------------------------------------------------------------------------

def _fmt_test_name(f: TestFailure) -> str:
    """Имя теста + пометка о ретраях для flaky-тестов."""
    if f.retries:
        return f"{f.test_name} (упал {f.retries} раз(а), затем прошёл)"
    return f.test_name


def print_report(groups: dict[str, list[TestFailure]], total_label: str = "Всего упавших тестов") -> None:
    # сортируем кластеры по размеру, самые массовые — первыми
    sorted_groups = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)

    total = sum(len(v) for v in groups.values())
    print(f"{total_label}: {total}")
    print(f"Уникальных кластеров причин: {len(groups)}\n")

    for i, (pattern, items) in enumerate(sorted_groups, start=1):
        print(f"Кластер {i} ({len(items)} тестов)")
        print(f"  Паттерн ошибки: {pattern[:200]}")

        # чаще всего упавший шаг в этом кластере — хороший намёк на причину
        steps = [f.failed_step for f in items if f.failed_step]
        if steps:
            most_common_step = max(set(steps), key=steps.count)
            print(f"  Чаще всего падает на шаге: {most_common_step}")

        example_names = [_fmt_test_name(f) for f in items[:3]]
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
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Путь для сохранения результата кластеризации в JSON "
             "(вход для run_summary.py и для сравнения прогонов между собой).",
    )
    parser.add_argument(
        "--include-flaky",
        action="store_true",
        help="Не исключать флейки-тесты (упал → потом прошёл на ретрае) "
             "из основной кластеризации причин падений. По умолчанию они "
             "исключаются оттуда (но всё равно кластеризуются отдельно, "
             "см. --flaky-html/--flaky-json).",
    )
    parser.add_argument(
        "--flaky-html",
        type=Path,
        default=None,
        help="Путь для сохранения отдельного HTML-дашборда по flaky-тестам "
             "(упал → прошёл на ретрае), например flaky-report.html.",
    )
    parser.add_argument(
        "--flaky-json",
        type=Path,
        default=None,
        help="Путь для сохранения кластеризации flaky-тестов в JSON.",
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        print(f"Папка не найдена: {args.results_dir}")
        sys.exit(1)

    records = load_all_attempts(args.results_dir)
    if not records:
        print("Результаты не найдены (папка не содержит allure-results).")
        return

    genuine_failures, flaky_failures = partition_flaky(records)

    if flaky_failures:
        print(f"Обнаружено флейки-тестов (упал → прошёл на ретрае): {len(flaky_failures)}")
        for f in flaky_failures[:10]:
            print(f"  - {_fmt_test_name(f)}")
        if len(flaky_failures) > 10:
            print(f"  ... и ещё {len(flaky_failures) - 10}")
        print()

    if args.include_flaky:
        # если явно попросили не исключать — считаем флейки обычным failed
        # на основе их последней упавшей попытки (а не первой, которую
        # используем для отдельной flaky-кластеризации ниже)
        for f in flaky_failures:
            attempts = [
                a for a in records
                if a["name"] == f.test_name and a["status"] in ("failed", "broken")
            ]
            if attempts:
                last = max(attempts, key=lambda a: a.get("start", 0))
                genuine_failures.append(TestFailure(
                    test_name=last["name"], status=last["status"],
                    message=last["message"], trace=last["trace"],
                    failed_step=find_failed_step(last["steps"]), start=last["start"],
                ))

    failures = genuine_failures
    if not failures and not flaky_failures:
        print("Тестов с окончательным падением не найдено (все либо прошли, либо флейки).")
        return

    groups = None
    if failures:
        if args.mode == "ml":
            groups = group_failures_ml(failures, min_cluster_size=args.min_cluster_size)
        else:
            groups = group_failures_exact(failures)

        print_report(groups)

        if args.json:
            summary = build_summary_json(groups, mode=args.mode, results_dir=str(args.results_dir))
            summary["generated_at"] = datetime.now(timezone.utc).isoformat()
            args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"JSON сохранён: {args.json.resolve()}")

    # -----------------------------------------------------------------
    # Отдельная кластеризация flaky-тестов — по причине их ПЕРВОГО
    # падения (до того, как они прошли на ретрае). Помогает увидеть,
    # что чаще всего "шатает" тесты, даже если итоговый прогон зелёный.
    # -----------------------------------------------------------------
    flaky_groups = None
    if flaky_failures:
        if args.mode == "ml":
            flaky_groups = group_failures_ml(flaky_failures, min_cluster_size=args.min_cluster_size)
        else:
            flaky_groups = group_failures_exact(flaky_failures)

        print("=" * 60)
        print("Кластеризация flaky-тестов (упал → прошёл на ретрае)")
        print("=" * 60)
        print_report(flaky_groups, total_label="Всего flaky-тестов")

        if args.flaky_json:
            flaky_summary = build_summary_json(
                flaky_groups, mode=args.mode, results_dir=str(args.results_dir), kind="flaky",
            )
            flaky_summary["generated_at"] = datetime.now(timezone.utc).isoformat()
            args.flaky_json.write_text(json.dumps(flaky_summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"JSON по flaky сохранён: {args.flaky_json.resolve()}")

    # -----------------------------------------------------------------
    # HTML: --html рисует оба графика (падения + флейки) на одной
    # странице, если есть и то, и другое — --flaky-html дополнительно
    # сохраняет отдельный файл только с графиком флейки.
    # -----------------------------------------------------------------
    if args.html:
        sections = []
        if groups:
            sections.append({
                "groups": groups, "title": "Причины падений",
                "total_label": "всего упавших тестов", "bar_color": "#4f7cff",
            })
        if flaky_groups:
            sections.append({
                "groups": flaky_groups, "title": "Flaky-тесты (упал → прошёл на ретрае)",
                "total_label": "всего flaky-тестов", "bar_color": "#f59f00",
            })
        html = build_dashboard_html(sections, mode=args.mode, results_dir=str(args.results_dir))
        args.html.write_text(html, encoding="utf-8")
        chart_note = " (2 графика: падения + флейки)" if len(sections) > 1 else ""
        print(f"HTML-дашборд сохранён: {args.html.resolve()}{chart_note}")

    if args.flaky_html and flaky_groups:
        flaky_html = build_dashboard_html(
            [{
                "groups": flaky_groups, "title": "Flaky-тесты (упал → прошёл на ретрае)",
                "total_label": "всего flaky-тестов", "bar_color": "#f59f00",
            }],
            mode=args.mode, results_dir=str(args.results_dir),
        )
        args.flaky_html.write_text(flaky_html, encoding="utf-8")
        print(f"HTML-дашборд по flaky сохранён: {args.flaky_html.resolve()}")


if __name__ == "__main__":
    main()
