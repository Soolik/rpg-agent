from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import unicodedata

from .api_models import AssistantMode


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


@dataclass(frozen=True)
class AssistantDecision:
    mode: AssistantMode
    artifact_type: Optional[str]
    save_output: bool
    output_title: Optional[str]
    requires_confirmation: bool
    confirmation_title: Optional[str]
    confirmation_body: Optional[str]


def _normalized(text: str) -> str:
    compact = " ".join((text or "").strip().lower().split())
    return "".join(
        char for char in unicodedata.normalize("NFKD", compact) if not unicodedata.combining(char)
    )


def infer_assistant_decision(
    *,
    message: str,
    requested_mode: AssistantMode,
    requested_artifact_type: Optional[str],
    requested_save_output: bool,
    requested_output_title: Optional[str],
    candidate_text: Optional[str],
) -> AssistantDecision:
    if requested_mode != AssistantMode.auto:
        return AssistantDecision(
            mode=requested_mode,
            artifact_type=requested_artifact_type,
            save_output=requested_save_output,
            output_title=requested_output_title,
            requires_confirmation=False,
            confirmation_title=None,
            confirmation_body=None,
        )

    message_norm = _normalized(message)
    mode = AssistantMode.create
    if candidate_text and candidate_text.strip():
        mode = AssistantMode.guard
    elif any(hint in message_norm for hint in GUARD_HINTS):
        mode = AssistantMode.guard
    elif any(hint in message_norm for hint in EDITOR_HINTS):
        mode = AssistantMode.editor

    artifact_type = requested_artifact_type
    if not artifact_type and mode == AssistantMode.create:
        for candidate_artifact, hints in ARTIFACT_HINTS:
            if any(hint in message_norm for hint in hints):
                artifact_type = candidate_artifact
                break

    save_output = bool(requested_save_output)
    if not save_output and mode == AssistantMode.create:
        save_output = any(hint in message_norm for hint in SAVE_HINTS)

    output_title = (requested_output_title or "").strip() or None
    if not output_title and save_output and artifact_type:
        output_title = ARTIFACT_LABELS.get(artifact_type, artifact_type.replace("_", " ").title())

    requires_confirmation = mode == AssistantMode.editor or save_output
    if not requires_confirmation:
        return AssistantDecision(
            mode=mode,
            artifact_type=artifact_type,
            save_output=save_output,
            output_title=output_title,
            requires_confirmation=False,
            confirmation_title=None,
            confirmation_body=None,
        )

    if mode == AssistantMode.editor:
        title = "Potwierdz zmiane kanonu"
        body = (
            "Rozumiem to jako prosbe o przygotowanie zmiany kanonu. "
            "Najpierw utworze propozycje zmiany do review, bez automatycznego zastosowania."
        )
    else:
        artifact_label = ARTIFACT_LABELS.get(artifact_type or "", "odpowiedz")
        title = "Potwierdz zapis"
        body = (
            f"Rozumiem to jako prosbe o wygenerowanie materialu ({artifact_label}) "
            f"i zapisanie go do Google Docs jako '{output_title or 'automatyczny tytul'}'."
        )

    return AssistantDecision(
        mode=mode,
        artifact_type=artifact_type,
        save_output=save_output,
        output_title=output_title,
        requires_confirmation=True,
        confirmation_title=title,
        confirmation_body=body,
    )
