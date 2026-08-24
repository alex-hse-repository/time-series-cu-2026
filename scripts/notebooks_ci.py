"""Запуск ноутбуков в CI: определение изменённых занятий и их прогон.

Репозиторий устроен как набор независимых занятий `week_N/`: у каждого свои
`pyproject.toml` и `uv.lock`, ноутбуки лежат в подпапках `part_*/`. Поэтому
единица прогона — занятие (для него поднимается своё окружение), а внутри
занятия прогоняются только затронутые ноутбуки.

Две команды:

    python scripts/notebooks_ci.py list-weeks
        Печатает JSON-массив занятий, которых коснулись изменения, — он
        подставляется в matrix workflow'а.

    python scripts/notebooks_ci.py run --week week_1
        Прогоняет изменённые ноутбуки занятия через `uv run ... nbconvert`.

Диапазон изменений берётся из `--base`/`--head` или переменных окружения
`BASE_SHA`/`HEAD_SHA` и считается от точки ветвления (`base...head`). Если база
недоступна (force-push, ручной запуск) или задан `--all` / `RUN_ALL=true`,
прогоняется всё.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
WEEK_GLOB = "week_*"
SKIP_FILE = Path(__file__).resolve().parent / "ci_skip_notebooks.txt"

# Ноутбуки исполняются ядром `python3` из окружения занятия: имя ядра в
# метаданных ноутбука (`seminar-1`, `seminar-2`) локальное и в CI не
# зарегистрировано.
KERNEL_NAME = "python3"
CELL_TIMEOUT = 1800


def git(*args: str) -> str:
    # quotepath=false: иначе git отдаёт кириллические имена ноутбуков
    # экранированными (`"week_1/\320\240..."`), и они не совпадают с путями
    # на диске.
    result = subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        cwd=ROOT_PATH,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def commit_exists(ref: str) -> bool:
    if not ref or set(ref) == {"0"}:
        return False
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{ref}^{{commit}}"],
        cwd=ROOT_PATH,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def changed_files(base: str, head: str) -> list[str] | None:
    """Пути, изменённые между base и head. None — если диапазон недоступен."""
    if not commit_exists(base) or not commit_exists(head):
        return None
    # Три точки — diff от точки ветвления (`git merge-base`), а не от самой базы.
    # В PR база успевает уехать вперёд, и двухточечный diff показал бы ещё и
    # чужие коммиты main — прогонялись бы ноутбуки, которых PR не касался.
    #
    # Строчная `d` в diff-filter исключает удалённые файлы: их запускать нечего.
    output = git("diff", "--name-only", "--diff-filter=d", f"{base}...{head}")
    return [line for line in output.splitlines() if line]


def all_weeks() -> list[str]:
    return sorted(
        path.name for path in ROOT_PATH.glob(WEEK_GLOB) if (path / "pyproject.toml").is_file()
    )


def skipped_notebooks() -> set[str]:
    if not SKIP_FILE.is_file():
        return set()
    lines = SKIP_FILE.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.strip().startswith("#")}


def week_notebooks(week: str) -> list[str]:
    return sorted(
        path.relative_to(ROOT_PATH).as_posix() for path in (ROOT_PATH / week).glob("part_*/*.ipynb")
    )


def notebooks_to_run(week: str, changed: list[str] | None) -> list[str]:
    """Ноутбуки занятия, которые нужно прогнать.

    Если менялось окружение (`pyproject.toml`/`uv.lock`), прогоняется всё
    занятие: поменяться могло поведение любой библиотеки. Иначе — только сами
    изменённые ноутбуки.
    """
    if changed is None:
        selected = week_notebooks(week)
    else:
        env_files = {f"{week}/pyproject.toml", f"{week}/uv.lock"}
        if env_files & set(changed):
            selected = week_notebooks(week)
        else:
            selected = sorted(
                path
                for path in changed
                if path.startswith(f"{week}/part_") and path.endswith(".ipynb")
            )
    skip = skipped_notebooks()
    return [path for path in selected if path not in skip]


def changed_weeks(changed: list[str] | None) -> list[str]:
    weeks = all_weeks()
    if changed is None:
        return [week for week in weeks if notebooks_to_run(week, None)]
    touched = {path.split("/", 1)[0] for path in changed}
    return [week for week in weeks if week in touched and notebooks_to_run(week, changed)]


def run_notebook(week: str, notebook: str, output_dir: Path) -> bool:
    print(f"::group::{notebook}", flush=True)
    # nbconvert исполняет ноутбук с рабочим каталогом самого ноутбука, поэтому
    # относительные пути к данным и картинкам внутри ячеек продолжают работать.
    command = [
        "uv",
        "run",
        "--project",
        week,
        "--with",
        "nbconvert",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        f"--ExecutePreprocessor.kernel_name={KERNEL_NAME}",
        f"--ExecutePreprocessor.timeout={CELL_TIMEOUT}",
        "--output-dir",
        str(output_dir),
        notebook,
    ]
    env = {**os.environ, "MPLBACKEND": "Agg"}
    ok = subprocess.run(command, cwd=ROOT_PATH, env=env).returncode == 0
    print("::endgroup::", flush=True)
    print(f"{'OK  ' if ok else 'FAIL'} {notebook}", flush=True)
    return ok


def command_list_weeks(args: argparse.Namespace) -> int:
    changed = None if args.all else changed_files(args.base, args.head)
    print(json.dumps(changed_weeks(changed), ensure_ascii=False))
    return 0


def command_run(args: argparse.Namespace) -> int:
    changed = None if args.all else changed_files(args.base, args.head)
    notebooks = notebooks_to_run(args.week, changed)
    if not notebooks:
        print(f"Нечего запускать в {args.week}")
        return 0

    output_dir = ROOT_PATH / args.output_dir / args.week
    output_dir.mkdir(parents=True, exist_ok=True)

    failed = [
        notebook for notebook in notebooks if not run_notebook(args.week, notebook, output_dir)
    ]
    print(f"\nЗапущено: {len(notebooks)}, упало: {len(failed)}")
    for notebook in failed:
        print(f"  {notebook}")
    return 1 if failed else 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--base", default=os.environ.get("BASE_SHA", ""))
    common.add_argument("--head", default=os.environ.get("HEAD_SHA", "HEAD"))
    common.add_argument(
        "--all",
        action="store_true",
        default=os.environ.get("RUN_ALL", "").lower() == "true",
        help="прогнать все ноутбуки, игнорируя diff",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list-weeks", parents=[common]).set_defaults(func=command_list_weeks)

    run_parser = subparsers.add_parser("run", parents=[common])
    run_parser.add_argument("--week", required=True, choices=all_weeks())
    run_parser.add_argument("--output-dir", default=".ci-notebooks")
    run_parser.set_defaults(func=command_run)

    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed = parse_args(sys.argv[1:])
    raise SystemExit(parsed.func(parsed))
