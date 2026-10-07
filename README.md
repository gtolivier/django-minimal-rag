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

Pre-alpha. Only the document Protocol and chunking exist so far: see
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

Text is split on blank lines first, then between words; chunks do not
overlap.

## Requirements

- Python 3.11+
- Django 5.2 LTS, 6.0 or 6.1
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector)
  extension

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
