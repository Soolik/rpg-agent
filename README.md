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

## V1 chat API

Nowy kontrakt dla klienta chatowego i UI worldbuildingu:

- `POST /v1/chat` - glowny endpoint asystenta z trybami `create`, `guard`, `editor`
- `POST /v1/assistant/actions` - wykonanie akcji zwrotnych z `next_actions`
- `POST /v1/imports/canonical-files` - import z lokalnego folderu albo z folderu Google Drive
- `GET /v1/conversations`
- `POST /v1/conversations`
- `GET /v1/conversations/{conversation_id}/messages`
- `POST /v1/conversations/{conversation_id}/messages`

Streaming:

- `stream=true` zwraca SSE z eventami `start`, `delta`, `complete`
- dla prostych odpowiedzi bez RAG/artefaktow backend uzywa bezposredniego streamingu modelu
- dla flow zlozonych (`guard`, `editor`, creative artifacts, save_output, telemetry, continuity checks) backend spada do buffered SSE i streamuje gotowa odpowiedz
- event `start` zwraca `stream_debug`, z powodem wyboru `direct` albo `buffered`

Przykladowy request `create`:

```json
{
  "message": "Przygotuj 3 hooki na kolejna sesje o Red Blade i Captain Mira.",
  "mode": "create",
  "artifact_type": "session_hooks",
  "save_output": true
}
```

Przykladowy request `guard`:

```json
{
  "message": "Sprawdz zgodnosc z kanonem.",
  "mode": "guard",
  "candidate_text": "* Captain Mira wspolpracuje z Red Blade.\n* Skup pojawia sie jako nowa frakcja."
}
```

Przykladowy request `editor`:

```json
{
  "message": "Dodaj nowego NPC powiazanego z Red Blade.",
  "mode": "editor"
}
```

Przykladowa odpowiedz `v1/chat`:

```json
{
  "request_id": "req_123",
  "trace_id": "req_123",
  "kind": "creative",
  "mode": "create",
  "reply_markdown": "## Hook 1\n...",
  "conversation_id": "conv_123",
  "artifact": {
    "artifact_type": "session_hooks",
    "text": "Tytul: ...",
    "format": "markdown"
  },
  "continuity": {
    "ok": false,
    "proposed_new_names": ["Skup"],
    "issues": [
      {
        "code": "new_proper_noun",
        "severity": "warning",
        "message": "Nowa nazwa wlasna: Skup"
      }
    ]
  },
  "next_actions": [
    {
      "type": "revise",
      "label": "Przerob odpowiedz",
      "payload": {
        "artifact_type": "session_hooks"
      }
    }
  ]
}
```

Przykladowy request do wykonania `next_actions`:

```json
{
  "action_type": "accept_world_change",
  "proposal_id": 42,
  "actor": "mg"
}
```

Przykladowy request importu kanonu z lokalnego folderu:

```json
{
  "source_path": "C:\\Users\\sular\\Desktop\\NoweMateriałyKampanii\\Krew_na_Gwiazdach_CANONICAL_2026_03_08_2046",
  "dry_run": true,
  "replace_existing": true,
  "reindex_after_import": true
}
```

Przykladowy request importu z folderu Google Drive:

```json
{
  "source_drive_folder_id": "1U4b9Abw835oo7vPkMUJBMkJF5cGnQSw7",
  "dry_run": true,
  "replace_existing": true,
  "reindex_after_import": true
}
```

Uwaga praktyczna:

- endpoint `POST /v1/imports/canonical-files` widzi tylko sciezki dostepne dla procesu API
- to zadziala lokalnie albo na serwerze z zamontowanym folderem
- Cloud Run nie zobaczy sciezki `C:\\Users\\...` z Twojego komputera
- dla lokalnego importu masz runner `tools/import_canonical_folder.py`

Przykladowe uruchomienie lokalnego importu:

```bash
python tools/import_canonical_folder.py --path "C:\Users\sular\Desktop\NoweMateriałyKampanii\Krew_na_Gwiazdach_CANONICAL_2026_03_08_2046"
python tools/import_canonical_folder.py --path "C:\Users\sular\Desktop\NoweMateriałyKampanii\Krew_na_Gwiazdach_CANONICAL_2026_03_08_2046" --apply
```

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

## Testowanie

Praktyczna instrukcja testow automatycznych, smoke testow i scenariuszy end-to-end jest w `docs/TESTING.md`.

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
