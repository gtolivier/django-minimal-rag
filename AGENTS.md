# AGENTS.md

Instructions for coding agents working in this repository. Read
[README.md](README.md) first for what this package is and is not.

## Commands

- Install: `uv sync`
- Test: `uv run pytest`
- Lint: `uv run ruff check` — format: `uv run ruff format`
- Type check: `uv run mypy` (strict, with django-stubs)

Always go through `uv run`; do not rely on an activated virtualenv.

## Layout

- `src/django_minimal_rag/` — the package (src layout: tests run against
  the installed package, not the working directory).
- `tests/` — pytest tests.

## Rules

- **No dependency on django-model-rag.** This package indexes documents it
  is given; it must never import or depend on django-model-rag. The two are
  connected by the host project, not by each other.
- **Dependencies:** add or remove them with `uv add` / `uv remove`, never by
  editing `pyproject.toml` by hand — a manual edit leaves `uv.lock` stale.
- **Generated files:** when an official tool can produce a file
  (`django-admin startapp`, `manage.py makemigrations`, `uv init`…), use it
  instead of writing the file.
- **Test-first:** every behavior in `src/` is introduced by a failing test.
- **Clean code, enforced where a tool can:** ruff flags magic values in
  comparisons (numbers and strings, outside `tests/`), complexity, naming,
  unused arguments and commented-out code; mypy runs in strict mode, so
  every function is annotated. A `# noqa` or `# type: ignore` must name
  the error it silences (both tools enforce it) and come with a comment
  saying why.
- **Supported versions:** Python 3.11+, Django 5.2 LTS / 6.0 / 6.1. Keep the
  CI matrix in `.github/workflows/ci.yml` in sync when this changes.
- **No `CLAUDE.md`:** this file is the single source of agent instructions.
  A `CLAUDE.md` next to it would make Claude Code ignore it.
