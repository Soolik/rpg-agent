# rpg-agent

Backend agenta AI do worldbuildingu RPG zintegrowanego z Google Docs/Drive, Supabase Postgres i warstwą RAG.

## Co jest teraz

- FastAPI do zapytań, ingestu notatek i planowania zmian.
- Google Drive / Docs jako edytowalne źródło wiedzy świata.
- Postgres z `pgvector` do wyszukiwania semantycznego.
- Przepływ `propose_changes -> apply_changes` pod kontrolowane zmiany w dokumentach.

## Najważniejsze endpointy

- `POST /ask` - odpowiedzi na pytania z RAG
- `POST /reindex` - reindeksacja dokumentów świata
- `POST /ingest_session` - zamiana surowych notatek sesji na patch
- `GET /world_status` - podgląd dokumentów świata
- `POST /propose_changes` - plan zmian
- `POST /apply_changes` - wykonanie zatwierdzonych zmian

## Lokalne uruchomienie

1. Skopiuj `.env.example` do `.env` i uzupełnij wartości.
2. Zainstaluj zależności:

```bash
python -m pip install -r requirements.txt
```

3. Uruchom API:

```bash
uvicorn main:app --reload
```

## Baza danych

Początkowy schemat jest w `sql/001_initial_schema.sql`.

Najważniejsze tabele:

- `chunks` - aktualny indeks RAG
- `world_docs` - metadane dokumentów świata
- `doc_snapshots` - wersje treści do późniejszego diff/sync
- `proposals` - zapis planów zmian
- `apply_runs` - audit trail wykonań zmian

## Obecne założenia architektoniczne

- Google Docs pozostają warstwą redakcyjną dla ludzi.
- Supabase/Postgres jest warstwą operacyjną i retrieval layer.
- Reindex jest idempotentny per dokument i obejmuje cały świat znaleziony przez `DriveStore`.
- `Thread Tracker` dalej ma specjalny tryb parsowania tabeli przez eksport HTML.

## Następne kroki

- dodać trwały sync `world_docs -> doc_snapshots -> chunks`
- rozwinąć `proposals` i `apply_runs` o widoki/listingi oraz filtrowanie
- dołożyć częściowy reindex po zmianach
- dodać testy i rozdzielić `main.py` na mniejsze moduły
