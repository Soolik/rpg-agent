from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import unicodedata

from .api_models import AssistantMode
from .chat_models import ArtifactType, ChatIntent


class TaskType(str, Enum):
    answer_question = "answer_question"
    create_artifact = "create_artifact"
    propose_doc_change = "propose_doc_change"
    apply_change = "apply_change"
    audit_consistency = "audit_consistency"
    ingest_session = "ingest_session"


ARTIFACT_LABELS = {
    "session_hooks": "hooki na sesje",
    "npc_brief": "brief NPC-a",
    "pre_session_brief": "brief przed sesja",
    "gm_brief": "brief MG",
    "scene_seed": "zalazek sceny",
    "twist_pack": "pakiet twistow",
    "player_summary": "podsumowanie dla graczy",
    "session_report": "raport z sesji",
}

SAVE_HINTS = (
    "zapisz",
    "wrzuc do drive",
    "dodaj do drive",
    "dodaj do google docs",
    "utworz dokument",
    "stworz dokument",
    "zapisz wynik",
)

EDITOR_HINTS = (
    "dodaj do kanonu",
    "dopisz do kanonu",
    "zapisz do kanonu",
    "wprowadz do kanonu",
    "zaktualizuj kanon",
    "zmien w kanonie",
    "edytuj kanon",
    "usun z kanonu",
    "popraw kanon",
    "zaktualizuj world model",
    "dopisz do world modelu",
)

GUARD_HINTS = (
    "sprawdz to z kanonem",
    "sprawdz ten tekst",
    "sprawdz ten opis",
    "sprawdz ten fragment",
    "sprawdz continuity",
    "sprawdz ciaglosc tego tekstu",
    "sprawdz zgodnosc tego tekstu",
    "czy to pasuje do kanonu",
    "czy ten tekst jest zgodny z kanonem",
    "wykryj sprzecznosci w tekscie",
    "przejrzyj pod katem kanonu",
    "sprawdz zgodnosc logiczna",
    "sprawdz logike kanonu",
)

ARTIFACT_HINTS = (
    ("session_hooks", ("hook", "hooki")),
    ("npc_brief", ("npc", "bn", "bohater niezalezny")),
    ("pre_session_brief", ("brief przed sesja", "brief na sesje", "checklista mg", "prep na sesje")),
    ("gm_brief", ("brief mg",)),
    ("scene_seed", ("scene", "sceny", "scene seed", "zalazek sceny")),
    ("twist_pack", ("twist", "zwrot akcji", "komplikacje")),
    ("player_summary", ("podsumowanie dla graczy", "summary dla graczy")),
    ("session_report", ("raport z sesji", "podsumowanie sesji", "session report")),
)

CHARACTER_CREATION_VERBS = ("wymysl", "pomysl", "zaproponuj", "stworz")
CHARACTER_CREATION_NOUNS = (
    "postac",
    "bohatera",
    "bohaterke",
    "pirata",
    "piratke",
    "kapitana",
    "kapitanke",
)

CREATIVE_MARKERS = (
    "wymysl ",
    "pomysl ",
    "zaproponuj ",
    "daj 3 pomysly",
    "hook",
    "hooki",
    "twist",
    "twisty",
    "seed",
    "scene seed",
    "nowego npc",
    "nowy npc",
    "stworz npc",
    "npc brief",
)

PROPOSAL_MARKERS = (
    "dodaj ",
    "podmien ",
    "zamien ",
    "zmien ",
    "utworz ",
    "stworz ",
    "uzupelnij ",
    "w dokumencie ",
    "sekcje ",
)

EDITOR_VERBS = (
    "dodaj",
    "dopisz",
    "zaktualizuj",
    "zmien",
    "usun",
    "popraw",
    "edytuj",
)


@dataclass(frozen=True)
class TaskSpec:
    task_type: TaskType
    assistant_mode: AssistantMode
    chat_intent: ChatIntent
    artifact_type: Optional[ArtifactType]
    save_output: bool
    output_title: Optional[str]
    requires_confirmation: bool
    confirmation_title: Optional[str]
    confirmation_body: Optional[str]
    reason: str


def _normalized(text: str) -> str:
    compact = " ".join((text or "").strip().lower().split())
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", compact) if not unicodedata.combining(ch)
    )


def _infer_artifact_type(message_norm: str, requested_artifact_type: Optional[str]) -> Optional[ArtifactType]:
    if requested_artifact_type:
        return requested_artifact_type  # type: ignore[return-value]
    if any(verb in message_norm for verb in CHARACTER_CREATION_VERBS) and any(
        noun in message_norm for noun in CHARACTER_CREATION_NOUNS
    ):
        return "npc_brief"
    for candidate_artifact, hints in ARTIFACT_HINTS:
        if any(hint in message_norm for hint in hints):
            return candidate_artifact  # type: ignore[return-value]
    return None


def _infer_chat_intent(message_norm: str, artifact_type: Optional[ArtifactType], requested_intent: ChatIntent) -> ChatIntent:
    if requested_intent != "auto":
        return requested_intent
    if artifact_type in {"session_hooks", "scene_seed", "npc_brief", "twist_pack"}:
        return "creative"
    if any(marker in message_norm for marker in CREATIVE_MARKERS):
        return "creative"
    if any(marker in message_norm for marker in PROPOSAL_MARKERS):
        return "proposal"
    if "notatki z sesji" in message_norm or "sesja:" in message_norm or "raw notes" in message_norm:
        return "session_sync"
    return "answer"


class TaskRouter:
    def classify(
        self,
        *,
        message: str,
        requested_mode: AssistantMode,
        requested_intent: ChatIntent,
        requested_artifact_type: Optional[str],
        requested_save_output: bool,
        requested_output_title: Optional[str],
        candidate_text: Optional[str],
    ) -> TaskSpec:
        message_norm = _normalized(message)
        artifact_type = _infer_artifact_type(message_norm, requested_artifact_type)
        editor_hint = any(hint in message_norm for hint in EDITOR_HINTS) or (
            "kanon" in message_norm and any(verb in message_norm for verb in EDITOR_VERBS)
        ) or (
            "world model" in message_norm and any(verb in message_norm for verb in EDITOR_VERBS)
        )

        if requested_mode != AssistantMode.auto:
            assistant_mode = requested_mode
            reason = "explicit_assistant_mode"
        elif candidate_text and candidate_text.strip():
            assistant_mode = AssistantMode.guard
            reason = "candidate_text_present"
        elif any(hint in message_norm for hint in GUARD_HINTS):
            assistant_mode = AssistantMode.guard
            reason = "guard_hint"
        elif editor_hint:
            assistant_mode = AssistantMode.editor
            reason = "editor_hint"
        else:
            assistant_mode = AssistantMode.create
            reason = "default_chat_mode"

        chat_intent = _infer_chat_intent(message_norm, artifact_type, requested_intent)
        if assistant_mode == AssistantMode.guard:
            task_type = TaskType.audit_consistency
            chat_intent = "answer"
        elif assistant_mode == AssistantMode.editor or chat_intent == "proposal":
            task_type = TaskType.propose_doc_change
            chat_intent = "proposal"
        elif chat_intent == "session_sync":
            task_type = TaskType.ingest_session
        elif artifact_type is not None or chat_intent == "creative":
            task_type = TaskType.create_artifact
        else:
            task_type = TaskType.answer_question

        save_output = bool(requested_save_output)
        if not save_output and task_type in {TaskType.answer_question, TaskType.create_artifact}:
            save_output = any(hint in message_norm for hint in SAVE_HINTS)

        output_title = (requested_output_title or "").strip() or None
        if not output_title and save_output and artifact_type:
            output_title = ARTIFACT_LABELS.get(artifact_type, artifact_type.replace("_", " ").title())

        requires_confirmation = assistant_mode == AssistantMode.editor or save_output
        confirmation_title = None
        confirmation_body = None
        if requires_confirmation:
            if assistant_mode == AssistantMode.editor:
                confirmation_title = "Potwierdz zmiane kanonu"
                confirmation_body = (
                    "Rozumiem to jako prosbe o przygotowanie zmiany kanonu. "
                    "Najpierw utworze propozycje zmiany do review, bez automatycznego zastosowania."
                )
            else:
                artifact_label = ARTIFACT_LABELS.get(artifact_type or "", "odpowiedz")
                confirmation_title = "Potwierdz zapis"
                confirmation_body = (
                    f"Rozumiem to jako prosbe o wygenerowanie materialu ({artifact_label}) "
                    f"i zapisanie go do Google Docs jako '{output_title or 'automatyczny tytul'}'."
                )

        return TaskSpec(
            task_type=task_type,
            assistant_mode=assistant_mode,
            chat_intent=chat_intent,
            artifact_type=artifact_type,
            save_output=save_output,
            output_title=output_title,
            requires_confirmation=requires_confirmation,
            confirmation_title=confirmation_title,
            confirmation_body=confirmation_body,
            reason=reason,
        )
