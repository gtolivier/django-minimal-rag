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

Pre-alpha. Nothing is implemented yet.

## Requirements

- Python 3.11+
- Django 5.2 LTS, 6.0 or 6.1
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector)
  extension

## Development

```sh
uv sync
uv run pytest
uv run ruff check
uv run ruff format --check
```

## License

MIT — see [LICENSE](LICENSE).
