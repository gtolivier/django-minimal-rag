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

Pre-alpha. Only the document Protocol, chunking, the storage models, the
embedding backend setting and indexing exist so far: see
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
  `source_key` (at most 500 characters: PostgreSQL cannot index much
  longer unique values);
- `Document` — a document of a source: its `title`, `url` and `language`
  (unbounded text, the language tag stored as given, `None` when unknown)
  and `permissions` (a list of permission names, empty by default);
- `Chunk` — a chunk of a document: its `rank`, `text`, the name of the
  `embedding_model` that produced its `embedding`, and that embedding.

The embedding column has no fixed dimension, so vectors of different
models, and of different dimensions, are stored side by side; each chunk
records which model produced it. Deleting a source deletes its documents,
and deleting a document deletes its chunks. Primary keys are
`BigAutoField`, whatever the project's `DEFAULT_AUTO_FIELD`. `str()` of a
document is its title, and of a chunk its text, cut to 80 characters with
a trailing `…`.

The app's migration creates the pgvector extension when the database does
not have it yet, which needs a database role allowed to create it. With a
role that is not, have a database administrator run `CREATE EXTENSION
vector` first: the migration then leaves it as it is. Rolling the app's
migrations back leaves the extension installed, since other apps may use
it.

## Embeddings

The project chooses its embedding model: the package ships none. Name a
backend class in the `MINIMAL_RAG_EMBEDDINGS` setting, with the keyword
arguments to build it with:

```python
MINIMAL_RAG_EMBEDDINGS = {
    "BACKEND": "myproject.embeddings.OpenAIEmbeddings",
    "OPTIONS": {"model": "text-embedding-3-small"},
}
```

`django_minimal_rag.embeddings.get_embeddings()` imports `BACKEND` and
builds `BACKEND(**OPTIONS)` anew on every call, so a backend holds nothing
costly to build per instance; `OPTIONS` is optional. It raises
`ImproperlyConfigured` when the setting is missing or not a mapping, when
`BACKEND` is missing, not a string or cannot be imported, and when
`OPTIONS` is not a mapping.

A backend is any class shaped like `django_minimal_rag.embeddings.Embeddings`,
a `typing.Protocol`:

- `model` (`str`) — the name of the model it embeds with, stored with each
  chunk;
- `embed(texts)` — takes a `Sequence[str]` and returns one vector, a
  `list[float]`, per text, in the same order. `replace()` sends all its
  new texts in one call, however many: a backend whose API limits the
  size of a request splits them into batches itself.

Changing models does not mix vectors: a chunk embedded by another model
than the current backend's is re-embedded by `replace()`, and retrieval
only reads chunks of the current model (a feature to come).

`django_minimal_rag.embeddings.FakeEmbeddings` is a backend for tests: no
network, no model, deterministic across runs and processes. Its
`dimensions` option (8 by default, at least 2) sets the length of its
vectors, and its `model` is `fake-<dimensions>`. Different texts get
different vectors, but their components are all positive, so any two
vectors are close: it suits tests that look a text up by its own vector,
not tests of semantic similarity. Its components are whole numbers below
`2**24`, exact in single precision, so a vector stored in a chunk's
`embedding` reads back equal.

```python
MINIMAL_RAG_EMBEDDINGS = {
    "BACKEND": "django_minimal_rag.embeddings.FakeEmbeddings",
    "OPTIONS": {"dimensions": 3},
}
```

## Indexing

`django_minimal_rag.indexing.Indexer().replace(groups)` stores documents
with their chunks and embeddings. `groups` maps each `source_key` to the
complete sequence of that source's documents:

```python
Indexer().replace({"faq:1": [entry], "guide:3": [part_1, part_2]})
```

- Each group replaces everything stored under its `source_key`: its
  documents are chunked as `chunk_group` chunks them (default options),
  embedded with the backend `MINIMAL_RAG_EMBEDDINGS` configures, and
  stored; sources absent from the call are left as they are. A document
  given twice in a group is stored twice, each time with its own chunks.
- An empty group removes its source, with its documents and chunks; for a
  key that is not stored, it does nothing.
- A chunk whose text is already stored in its source, embedded by the
  current backend's model, keeps its vector: only new or changed texts,
  and texts embedded by another model, are sent to the backend.
- The new texts of all groups are sent to the backend in a single
  `embed()` call, each text once, however many groups or documents share
  it. A call whose groups are all empty builds no backend, so it works
  without `MINIMAL_RAG_EMBEDDINGS`.
- A call is one transaction, embedding included: if the backend raises,
  the exception propagates and nothing of the call is stored. The row of
  each replaced or removed source is locked (`SELECT … FOR UPDATE`) before
  anything of it changes, until the transaction ends, so two calls on the
  same source run one after the other instead of interleaving their
  documents. The rows are locked in sorted `source_key` order, so two
  calls sharing several sources do not deadlock, whatever the order of
  their `groups`.
- `ValueError` is raised, and nothing of the call is stored, when a
  document's `source_key` differs from the key of its group (the message
  names both keys), and when the backend returns another number of vectors
  than it was given texts (the message gives both counts).
- Each group is iterated once, so a sequence that yields its items only
  the first time is stored whole; documents need not be hashable.

`Indexer().prune(model_label, kept_keys)` removes the sources of a model
that are not kept:

```python
Indexer().prune("faq", {"faq:1", "faq:3"})
```

- Every stored source whose `source_key` starts with `model_label`
  followed by a colon, and is not in `kept_keys`, is removed with its
  documents and chunks. Other sources are left as they are: `app.note`
  never matches `app.notebook:3`, and the label is matched literally
  (`_` and `%` are not wildcards).
- A call is one transaction. The rows of the removed sources are locked
  before anything is deleted, in the order `replace()` locks them (by
  code point, not by the database's collation), so a prune and a
  replacement sharing sources do not deadlock.

`replace()` and `prune()` form django-model-rag's output Protocol.

## Requirements

- Python 3.11+
- Django 5.2 LTS, 6.0 or 6.1
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector)
  extension, in a UTF-8 database (PostgreSQL's usual default): `prune()`
  orders its locks by byte, which matches `replace()`'s code-point order
  only in UTF-8
- `django.contrib.postgres` in `INSTALLED_APPS`, next to the app: its
  models use PostgreSQL array fields, which Django 6.0 and later refuse to
  use without it (system check `postgres.E005`):

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
