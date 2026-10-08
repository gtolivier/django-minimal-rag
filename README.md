# django-minimal-rag

A minimal RAG layer for Django: chunking, pgvector storage, retrieval and
cited LLM answers.

**It is** the retrieval-augmented generation part of a Django project: it
splits documents into chunks, stores their embeddings in PostgreSQL with
pgvector, retrieves the relevant chunks for a question — discarding those
below a relevance threshold — and asks an LLM to answer from them only,
with citations that point back to the source pages.

**It is not** a content-discovery tool: it does not know your models, it
indexes the documents you give it. To extract documents from arbitrary
Django models, see
[django-model-rag](https://github.com/gtolivier/django-model-rag) — this
package does not depend on it. It is not framework-agnostic either: it is a
Django app, and it requires PostgreSQL with the pgvector extension.

## Status

Pre-alpha. Only the document Protocol, chunking and the storage models
exist so far: see
[ROADMAP.md](ROADMAP.md) for the planned architecture, the decisions
already made and the features to come.

## Documents

This package indexes any object shaped like `django_minimal_rag.Document`,
a `typing.Protocol`: `text`, `source_key`, `title` and `url` (`str`),
`language` (`str | None`) and `permissions` (`AbstractSet[str]`), all
read-only. Any class with those members — a dataclass, frozen or not, a
class with properties or with plain attributes — satisfies it by its shape
alone: no base class to inherit, nothing to import from the producer.

## Chunking

`django_minimal_rag.chunking.chunk_group(documents)` splits the documents
of a group into chunks, ranked from 0 across the group in the documents'
order. Each chunk is at most `max_length` long (1000 by default), as
measured by `length` (`len` by default) — pass a tokenizer's count to
bound chunks in tokens:

```python
chunks = chunk_group(documents, max_length=500, length=count_tokens)
```

Text is split on blank lines first, then between lines, then between
words; a word longer than `max_length` is cut. Paragraphs, lines and words
that fit are packed together, joined by a blank line, a newline and a space
respectively. Every line boundary `str.splitlines` recognizes — `\r\n`,
a lone `\r`, `\u2028`… — becomes `\n`. The blank lines around the text and
the whitespace at the end of each line are stripped, but each line keeps
its indentation, so Markdown lists and code keep their structure; only a
line split between words loses it. Chunks do not overlap.

`max_length` must be at least 1, and every character of a word that has to
be cut must fit in it on its own: otherwise `ValueError` is raised rather
than a chunk exceeding the limit. Whitespace never reaches a chunk unless it
fits, so it never raises. `length` is expected to grow with the text, as
`len` and token counts do.

## Storage

The app's models hold what gets indexed:

- `Source` — one group of documents, identified by its unique
  `source_key`;
- `Document` — a document of a source: its `title`, `url` (unbounded
  text), `language` (`None` when unknown) and `permissions` (a list of
  permission names, empty by default);
- `Chunk` — a chunk of a document: its `rank`, `text`, the name of the
  `embedding_model` that produced its `embedding`, and that embedding.

The embedding column has no fixed dimension, so vectors of different
models, and of different dimensions, are stored side by side; each chunk
records which model produced it. Deleting a source deletes its documents,
and deleting a document deletes its chunks. Primary keys are
`BigAutoField`, whatever the project's `DEFAULT_AUTO_FIELD`.

The app's migration creates the pgvector extension (`CREATE EXTENSION IF
NOT EXISTS vector`), which needs a database role allowed to create it.

## Requirements

- Python 3.11+
- Django 5.2 LTS, 6.0 or 6.1
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector)
  extension
- `django.contrib.postgres` in `INSTALLED_APPS`, next to the app — its
  models use PostgreSQL array fields:

  ```python
  INSTALLED_APPS = [
      # ...
      "django.contrib.postgres",
      "django_minimal_rag",
  ]
  ```

## Development

The tests need PostgreSQL with pgvector. `compose.yaml` starts one on port
5433, which the test settings default to; the libpq environment variables
(`PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`) point them elsewhere:

```sh
uv sync
docker compose up -d --wait
uv run pytest
uv run ruff check
uv run ruff format --check
```

## License

MIT — see [LICENSE](LICENSE).
