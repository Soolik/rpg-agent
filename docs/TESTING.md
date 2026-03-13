# Testing Guide

Ten dokument rozdziela testy na 3 poziomy:

- testy automatyczne lokalne
- smoke testy po deployu
- scenariusze end-to-end dla kluczowych use case'ow

## 1. Testy automatyczne lokalne

Uruchamiaj po kazdej wiekszej zmianie w kodzie:

```powershell
cd C:\Users\sular\Documents\GitHub\rpg-agent
.\.venv313\Scripts\python.exe -m unittest discover -s tests -v
.\.venv313\Scripts\python.exe -m compileall main.py app tests
.\.venv313\Scripts\python.exe -c "import main; print(main.app.title)"
```

Co oznacza sukces:

- `unittest`: wszystkie testy sa `ok`
- `compileall`: brak bledow kompilacji
- `import main`: wypisuje `rpg-agent`

## 2. Warunki do testow integracyjnych

Przed testami na Cloud Run upewnij sie, ze:

- wdrozona jest aktualna rewizja uslugi
- `DB_URL` wskazuje na baze z migracjami `sql/001_initial_schema.sql` i `sql/002_workflow_runs.sql`
- `GEMINI_API_KEY` jest ustawiony
- wszystkie `*_FOLDER_ID` sa ustawione
- service account Cloud Run ma dostep do Google Drive / Google Docs

Przyklady ponizej zakladaja:

```powershell
$BASE_URL = "https://YOUR-CLOUD-RUN-URL"
```

## 3. Smoke test po deployu

To jest minimalny zestaw testow, ktory mowi czy usluga zyje.

### 3.1 Health

```powershell
Invoke-RestMethod "$BASE_URL/health"
```

Oczekiwane:

- `ok = true`
- poprawny `campaign_id`

### 3.2 World status

```powershell
Invoke-RestMethod "$BASE_URL/world_status"
```

Oczekiwane:

- lista dokumentow nie jest pusta
- foldery maja sensowne liczniki
- `indexed_chunks` jest liczba albo `null` przed pierwszym reindexem

### 3.3 Full reindex

```powershell
Invoke-RestMethod `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"clean": false}' `
  "$BASE_URL/reindex"
```

Oczekiwane:

- `ok = true`
- `mode = "full"`
- `indexed_docs > 0`
- `indexed_chunks > 0`

### 3.4 Workflow list endpoints

```powershell
Invoke-RestMethod "$BASE_URL/proposals"
Invoke-RestMethod "$BASE_URL/apply_runs"
```

Oczekiwane:

- endpointy odpowiadaja `200`
- zwracaja tablice, nawet jesli puste

## 4. Scenariusze testowe

Ponizej sa scenariusze, ktore realnie sprawdzaja funkcje produktu.

### Scenariusz A: Odczyt wiedzy swiata

Cel:
- sprawdzic, czy RAG odpowiada z dokumentow kampanii

Kroki:

```powershell
$body = @{
  question = "Jakie glowne frakcje wystepuja w kampanii?"
  top_k = 6
  include_sources = $true
  mode = "campaign"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -ContentType "application/json" `
  -Body $body `
  "$BASE_URL/ask"
```

Oczekiwane:

- odpowiedz jest po polsku
- nie jest pusta
- jesli `include_sources = true`, zwraca `sources`
- odpowiedz nie wyglada na halucynacje niezwiązane z dokumentami

### Scenariusz B: Ingest notatek z sesji

Cel:
- sprawdzic, czy system umie zamienic surowe notatki na patch

Przykladowe notatki:

```text
Gracze spotkali kapitan Mire w porcie na stacji Helion.
Mira ujawnila, ze pracuje dla frakcji Czerwone Ostrze.
Watek przemytu broni przeszedl z fazy podejrzen do otwartego sledztwa.
```

Kroki:

```powershell
$body = @{
  raw_notes = "Gracze spotkali kapitan Mire w porcie na stacji Helion.`nMira ujawnila, ze pracuje dla frakcji Czerwone Ostrze.`nWatek przemytu broni przeszedl z fazy podejrzen do otwartego sledztwa."
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -ContentType "application/json" `
  -Body $body `
  "$BASE_URL/ingest_session"
```

Oczekiwane:

- `session_summary` nie jest pusty
- `thread_tracker_patch` ma sensowne wpisy
- `entities_patch` zawiera NPC, lokacje albo frakcje z notatek

### Scenariusz C: Proposal workflow

Cel:
- sprawdzic, czy agent umie zaplanowac zmiany i zapisac proposal

Kroki:

```powershell
$body = @{
  instruction = "Dodaj do NPC Kapitan Mira sekcje Secrets z informacja, ze wspolpracuje z Czerwonym Ostrzem."
  mode = "update"
  dry_run = $true
} | ConvertTo-Json

$proposal = Invoke-RestMethod `
  -Method Post `
  -ContentType "application/json" `
  -Body $body `
  "$BASE_URL/propose_changes"

$proposal
```

Oczekiwane:

- zwraca `summary`, `actions`, `needs_confirmation`
- `proposal_id` jest ustawione
- `actions` wskazuja konkretny dokument lub sekcje

Dodatkowa walidacja:

```powershell
Invoke-RestMethod "$BASE_URL/proposals"
Invoke-RestMethod "$BASE_URL/proposals/$($proposal.proposal_id)"
```

Oczekiwane:

- proposal jest widoczny na liscie
- detail zawiera `request` i `proposal`

### Scenariusz D: Apply workflow

Cel:
- sprawdzic, czy proposal da sie wykonac, zapisac do Google Docs i zalogowac w audycie

Uwaga:
- wykonuje realna zmiane w dokumentach Google Docs
- testuj na bezpiecznym dokumencie testowym albo na sekcji przeznaczonej do testow

Kroki:

```powershell
$applyBody = @{
  proposal_id = $proposal.proposal_id
  proposal = $proposal
  approved = $true
  approved_by = "stakeholder"
  reindex_after_apply = $true
} | ConvertTo-Json -Depth 10

$apply = Invoke-RestMethod `
  -Method Post `
  -ContentType "application/json" `
  -Body $applyBody `
  "$BASE_URL/apply_changes"

$apply
```

Oczekiwane:

- `ok = true`
- `apply_run_id` jest ustawione
- `reindex_result.mode = "partial"` dla zmienionych dokumentow

Dodatkowa walidacja:

```powershell
Invoke-RestMethod "$BASE_URL/apply_runs"
Invoke-RestMethod "$BASE_URL/apply_runs/$($apply.apply_run_id)"
```

Oczekiwane:

- apply run jest widoczny na liscie
- detail zawiera `request` i `response`

### Scenariusz E: Weryfikacja efektu po apply

Cel:
- sprawdzic, czy zmiana rzeczywiscie trafila do dokumentu i do indeksu

Kroki:

1. Odczytaj dokument:

```powershell
$docBody = @{
  folder = "03 NPC"
  title = "Captain Mira"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -ContentType "application/json" `
  -Body $docBody `
  "$BASE_URL/read_world_doc"
```

2. Zadaj pytanie o nowy fakt:

```powershell
$askBody = @{
  question = "Jakie sekrety ma Kapitan Mira?"
  top_k = 6
  include_sources = $true
  mode = "campaign"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -ContentType "application/json" `
  -Body $askBody `
  "$BASE_URL/ask"
```

Oczekiwane:

- `read_world_doc` pokazuje nowa tresc sekcji
- `ask` uwzglednia nowy fakt po partial reindex

## 5. Co testowac po kazdym deployu

Minimalny pakiet akceptacyjny:

1. `GET /health`
2. `GET /world_status`
3. `POST /reindex`
4. `POST /ask`
5. `POST /propose_changes`
6. `GET /proposals`
7. `POST /apply_changes` na bezpiecznym obiekcie testowym
8. `GET /apply_runs`

## 6. Najczestsze problemy

- `500 Missing env`
  - brak wymaganej zmiennej srodowiskowej
- `500 Gemini ...`
  - problem z `GEMINI_API_KEY` albo modelem
- puste `world_status`
  - Cloud Run nie widzi folderow albo nie ma dostepu do Drive
- `reindex` zwraca `indexed_docs = 0`
  - foldery sa puste albo service account nie ma dostepu
- `propose_changes` dziala, ale `apply_changes` nie zmienia dokumentu
  - proposal wskazuje zly target albo dokument nie zostal znaleziony
- `ask` nie widzi nowych zmian
  - apply nie trafil w poprawny dokument albo partial reindex nie mial targetu
