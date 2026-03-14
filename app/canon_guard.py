from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Iterable, List, Optional

from .api_models import ContinuityIssue, ContinuityReport
from .text_normalization import normalize_text_artifacts


PROPER_NOUN_IGNORE_KEYS = {
    "active threads",
    "campaign bible",
    "campaign state",
    "co przygotowac",
    "dmg",
    "factions",
    "gm",
    "glossary",
    "hook",
    "hooks",
    "imie",
    "jak uzyc tej postaci na sesji",
    "key npcs",
    "key npcs and factions",
    "mg",
    "npc",
    "pierwsze wrazenie",
    "prep",
    "prep checklist",
    "pre-session brief",
    "pre session brief",
    "pressure points",
    "relacje",
    "risks and pressure points",
    "rola w kampanii",
    "rules and tone",
    "scene opportunities",
    "sekret",
    "session",
    "stawki",
    "thread tracker",
    "threads",
    "tytul",
}

IGNORED_SINGLE_WORDS = {
    "bg",
    "bohaterowie",
    "cel",
    "czy",
    "jezeli",
    "jesli",
    "kiedy",
    "kto",
    "lokalne",
    "miasto",
    "misja",
    "musi",
    "ostatnie",
    "plotki",
    "prawda",
    "przygotuj",
    "scena",
    "sceny",
    "session",
    "spotkanie",
    "to",
    "wladze",
    "zima",
}


def normalize_key(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", normalize_text_artifacts(value or "").strip().lower())


def extract_titlecase_phrases(text: str) -> List[str]:
    ignored = {
        "wymysl",
        "przygotuj",
        "stworz",
        "potrzebny",
        "daj",
        "zrob",
        "opisz",
    }
    titlecase_word = r"[A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9'_-]*"
    normalized_text = normalize_text_artifacts(text or "")
    matches = re.findall(rf"\b{titlecase_word}(?:\s+{titlecase_word}){{0,3}}\b", normalized_text)
    phrases: List[str] = []
    seen = set()
    for match in matches:
        cleaned = match.strip()
        key = normalize_key(cleaned)
        if not cleaned or key in seen:
            continue
        if " " not in cleaned and key in ignored:
            continue
        seen.add(key)
        phrases.append(cleaned)
    return phrases


def looks_like_proper_noun_label(value: str) -> bool:
    cleaned = re.sub(r"\s+", " ", (value or "").strip(" -*:;,.!?()[]{}\"'")).strip()
    if not cleaned:
        return False
    key = normalize_key(cleaned)
    if key in PROPER_NOUN_IGNORE_KEYS:
        return False
    if " " not in cleaned:
        return key not in IGNORED_SINGLE_WORDS and cleaned[0].isupper()
    words = cleaned.split()
    return all(re.match(r"^[A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9'_-]*$", word) for word in words)


def extract_proper_noun_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    seen = set()

    def add_name(value: str) -> None:
        cleaned = re.sub(r"\s+", " ", (value or "").strip()).strip(" -*:;,.!?()[]{}\"'")
        key = normalize_key(cleaned)
        if not cleaned or not key or key in seen:
            return
        seen.add(key)
        candidates.append(cleaned)

    for match in re.findall(r"\*\*([^*\n]{2,80})\*\*", normalize_text_artifacts(text or "")):
        if looks_like_proper_noun_label(match):
            add_name(match)

    for match in re.findall(
        r"(?m)^\s*[\*\-]?\s*([A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9'_-]{2,})\s*:",
        normalize_text_artifacts(text or ""),
    ):
        if looks_like_proper_noun_label(match):
            add_name(match)

    for match in re.findall(
        r"(?mi)^\s*(?:imie|tytul)\s*:\s*([^\n]{2,80})$",
        normalize_text_artifacts(text or ""),
    ):
        if looks_like_proper_noun_label(match):
            add_name(match)

    for phrase in extract_titlecase_phrases(text):
        if " " not in phrase:
            continue
        if looks_like_proper_noun_label(phrase):
            add_name(phrase)

    return candidates


def _dedupe_names(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        cleaned = normalize_text_artifacts(value or "").strip()
        key = normalize_key(cleaned)
        if not cleaned or not key or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def build_continuity_report(
    *,
    message: str,
    generated_text: str,
    known_entity_names: Optional[Iterable[str]] = None,
    known_thread_names: Optional[Iterable[str]] = None,
    extra_allowed_names: Optional[Iterable[str]] = None,
    allow_proposed_new_names: bool = False,
) -> ContinuityReport:
    known_names = _dedupe_names([*(known_entity_names or []), *(known_thread_names or [])])
    known_keys = {normalize_key(name): name for name in known_names}
    request_names = _dedupe_names([*extract_titlecase_phrases(message), *(extra_allowed_names or [])])
    request_keys = {normalize_key(name): name for name in request_names}
    source_backed: List[str] = []
    inferred: List[str] = []
    normalized_text = normalize_key(generated_text)
    for name in known_names:
        key = normalize_key(name)
        if key and key in normalized_text:
            source_backed.append(name)
    for name in request_names:
        key = normalize_key(name)
        if key and key not in known_keys and key in normalized_text:
            inferred.append(name)
    output_names = _dedupe_names(extract_proper_noun_candidates(generated_text))

    source_backed = _dedupe_names(source_backed)
    inferred = _dedupe_names(inferred)
    proposed_new: List[str] = []
    possible_conflicts: List[str] = []
    issues: List[ContinuityIssue] = []

    for name in output_names:
        key = normalize_key(name)
        if key in known_keys:
            source_backed.append(known_keys[key])
            continue
        if key in request_keys:
            inferred.append(request_keys[key])
            issues.append(
                ContinuityIssue(
                    code="inferred_claim",
                    severity="info",
                    message=f"Nazwa `{name}` pochodzi z prosby uzytkownika, ale nie ma potwierdzenia w world model.",
                    related_name=name,
                    source="request",
                )
            )
            continue

        proposed_new.append(name)
        close_match = get_close_matches(name, known_names, n=1, cutoff=0.82)
        if close_match:
            possible_conflicts.append(f"{name} ~= {close_match[0]}")
            issues.append(
                ContinuityIssue(
                    code="possible_name_conflict",
                    severity="error" if not allow_proposed_new_names else "warning",
                    message=f"Nazwa `{name}` jest podejrzanie podobna do znanej nazwy `{close_match[0]}`.",
                    related_name=name,
                    evidence=close_match[0],
                    source="generated",
                )
            )
            continue

        if allow_proposed_new_names:
            issues.append(
                ContinuityIssue(
                    code="new_proper_noun",
                    severity="info",
                    message=f"Generator zaproponowal nowa nazwe wlasna `{name}`.",
                    related_name=name,
                    source="generated",
                )
            )
        else:
            issues.append(
                ContinuityIssue(
                    code="new_proper_noun",
                    severity="warning",
                    message=f"Generator dodal nowa nazwe wlasna `{name}` poza znanym kanonem i prosba uzytkownika.",
                    related_name=name,
                    source="generated",
                )
            )

    ok = not any(issue.severity in {"warning", "error"} for issue in issues)
    return ContinuityReport(
        ok=ok,
        source_backed_names=_dedupe_names(source_backed),
        inferred_names=_dedupe_names(inferred),
        proposed_new_names=_dedupe_names(proposed_new),
        possible_conflicts=_dedupe_names(possible_conflicts),
        issues=issues,
    )
