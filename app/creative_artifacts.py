from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional


ArtifactTypeName = Literal[
    "gm_brief",
    "session_report",
    "player_summary",
    "pre_session_brief",
    "session_hooks",
    "scene_seed",
    "npc_brief",
    "twist_pack",
]


def build_creative_artifact_sections(artifact_type: ArtifactTypeName) -> str:
    if artifact_type == "pre_session_brief":
        return """
# Pre-Session Brief

## Campaign State

## Active Threads

## Key NPCs and Factions

## Risks and Pressure Points

## Scene Opportunities

## Prep Checklist
""".strip()
    if artifact_type == "session_hooks":
        return """
Tytul:

Hook 1:

Hook 2:

Hook 3:

Stawki:

Co przygotowac:
""".strip()
    if artifact_type == "scene_seed":
        return """
Tytul sceny:

Cel sceny:

Miejsce:

Zaangazowane postacie:

Przebieg:

Komplikacja:

Mozliwe skutki:
""".strip()
    if artifact_type == "npc_brief":
        return """
Imie:

Rola w kampanii:

Pierwsze wrazenie:

Motywacja:

Sekret:

Relacje:

Jak uzyc tej postaci na sesji:
""".strip()
    return """
Twist 1:

Twist 2:

Twist 3:

Foreshadowing:

Ryzyko dla kampanii:
""".strip()


def artifact_required_markers(artifact_type: ArtifactTypeName) -> List[str]:
    if artifact_type == "pre_session_brief":
        return [
            "# Pre-Session Brief",
            "## Campaign State",
            "## Active Threads",
            "## Key NPCs and Factions",
            "## Risks and Pressure Points",
            "## Scene Opportunities",
            "## Prep Checklist",
        ]
    if artifact_type == "session_hooks":
        return ["Tytul:", "Hook 1:", "Hook 2:", "Hook 3:", "Stawki:", "Co przygotowac:"]
    if artifact_type == "scene_seed":
        return [
            "Tytul sceny:",
            "Cel sceny:",
            "Miejsce:",
            "Zaangazowane postacie:",
            "Przebieg:",
            "Komplikacja:",
            "Mozliwe skutki:",
        ]
    if artifact_type == "npc_brief":
        return [
            "Imie:",
            "Rola w kampanii:",
            "Pierwsze wrazenie:",
            "Motywacja:",
            "Sekret:",
            "Relacje:",
            "Jak uzyc tej postaci na sesji:",
        ]
    if artifact_type == "twist_pack":
        return ["Twist 1:", "Twist 2:", "Twist 3:", "Foreshadowing:", "Ryzyko dla kampanii:"]
    return []


def artifact_style_guidance(artifact_type: ArtifactTypeName) -> str:
    if artifact_type == "pre_session_brief":
        return (
            "- Kazda sekcja ma byc zwiezla i praktyczna.\n"
            "- Uzywaj 2-5 bulletow na sekcje, zamiast dlugich akapitow.\n"
            '- W "Scene Opportunities" dawaj konkretne sceny do zagrania.\n'
            '- W "Prep Checklist" dawaj konkretne rzeczy do przygotowania przez MG.'
        )
    if artifact_type == "session_hooks":
        return (
            "- Daj dokladnie trzy rozne hooki.\n"
            "- Kazdy hook: 2-4 zdania.\n"
            "- Sekcje 'Stawki' i 'Co przygotowac' wypelnij krotszymi bulletami."
        )
    if artifact_type == "scene_seed":
        return "- Kazda sekcja ma byc krotka, konkretna i gotowa do uzycia na sesji."
    if artifact_type == "npc_brief":
        return (
            "- Kazda sekcja ma byc konkretna i zwiezla.\n"
            "- Nie pisz dlugich blokow tekstu; zwykle 2-4 zdania na sekcje.\n"
            "- Sekcje 'Relacje' i 'Jak uzyc tej postaci na sesji' moga byc listami."
        )
    if artifact_type == "twist_pack":
        return "- Kazdy twist ma byc odrebny, konkretny i opisany w 2-4 zdaniach."
    return "- Pisz zwiezle i praktycznie."


def extract_artifact_sections(text: str, artifact_type: ArtifactTypeName) -> Dict[str, str]:
    markers = artifact_required_markers(artifact_type)
    sections: Dict[str, str] = {}
    current_marker: Optional[str] = None
    buffer: List[str] = []

    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        matched_marker: Optional[str] = None
        inline_value = ""

        for marker in markers:
            if stripped == marker:
                matched_marker = marker
                break
            if not marker.startswith("#") and stripped.startswith(marker):
                matched_marker = marker
                inline_value = stripped[len(marker):].strip()
                break

        if matched_marker:
            if current_marker is not None:
                sections[current_marker] = "\n".join(buffer).strip()
            current_marker = matched_marker
            buffer = [inline_value] if inline_value else []
            continue

        if current_marker is not None:
            buffer.append(line)

    if current_marker is not None:
        sections[current_marker] = "\n".join(buffer).strip()
    return sections


def normalize_section_body(content: str) -> str:
    normalized = re.sub(r"^[\-\*\d\.\)\s]+", "", (content or "").strip(), flags=re.MULTILINE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def bullet_count(content: str) -> int:
    return sum(1 for line in (content or "").splitlines() if re.match(r"^\s*[\*\-]\s+", line))


def ends_with_sentence_punctuation(content: str) -> bool:
    return bool(re.search(r"[.!?\)\]\"']\s*$", (content or "").strip()))


def sentence_count(content: str) -> int:
    return len(re.findall(r"[.!?](?=(?:[\"')\]]*)?(?:\s|$))", (content or "").strip()))


def extract_bullet_items(content: str) -> List[str]:
    items: List[str] = []
    current: List[str] = []

    for raw_line in (content or "").splitlines():
        line = raw_line.rstrip()
        bullet_match = re.match(r"^\s*[\*\-]\s+(.*)$", line)
        if bullet_match:
            if current:
                items.append(" ".join(part for part in current if part).strip())
            current = [bullet_match.group(1).strip()]
            continue
        if current and line.strip():
            current.append(line.strip())

    if current:
        items.append(" ".join(part for part in current if part).strip())
    return [item for item in items if item]


def complete_bullet_items(content: str, minimum_length: int = 8) -> List[str]:
    items: List[str] = []
    for item in extract_bullet_items(content):
        normalized = normalize_section_body(item)
        if (
            len(normalized) >= minimum_length
            and "do doprecyzowania" not in normalized.lower()
            and ends_with_sentence_punctuation(item)
        ):
            items.append(re.sub(r"\s+", " ", item).strip())
    return items


def complete_bullet_count(content: str, minimum_length: int = 8) -> int:
    return len(complete_bullet_items(content, minimum_length=minimum_length))


def trim_to_complete_sentences(content: str) -> str:
    raw = (content or "").strip()
    if not raw or ends_with_sentence_punctuation(raw):
        return raw

    matches = list(re.finditer(r"[.!?](?=(?:[\"')\]]*)?(?:\s|$))", raw))
    if not matches:
        return raw

    trimmed = raw[:matches[-1].end()].strip()
    if len(normalize_section_body(trimmed)) >= max(20, len(normalize_section_body(raw)) // 2):
        return trimmed
    return raw


def normalize_single_line_section(content: str) -> str:
    lines = [re.sub(r"^\s*[\*\-]\s+", "", line).strip() for line in (content or "").splitlines() if line.strip()]
    if not lines:
        return ""
    first_line = lines[0]
    first_line = re.sub(r"\s+", " ", first_line).strip()
    sentence_parts = re.split(r"(?<=[.!?])\s+", first_line)
    return sentence_parts[0].strip() if sentence_parts else first_line


def strip_section_marker(text: str, marker: str) -> str:
    stripped = (text or "").strip()
    if stripped.lower().startswith(marker.lower()):
        stripped = stripped[len(marker):].lstrip()
    return stripped.strip()


def sanitize_generated_section(marker: str, content: str) -> str:
    raw = strip_section_marker(content, marker)
    if marker in {"Tytul:", "Tytul sceny:", "Imie:"}:
        return normalize_single_line_section(raw)
    if marker in {"Stawki:", "Co przygotowac:", "Relacje:", "Jak uzyc tej postaci na sesji:"}:
        items = [trim_to_complete_sentences(item) for item in extract_bullet_items(raw)]
        items = complete_bullet_items("\n".join(f"* {item}" for item in items if item.strip()))
        items = [re.sub(r"\s+", " ", item).strip() for item in items if item.strip()]
        return "\n".join(f"* {item}" for item in items)
    if marker.startswith("## "):
        items = [trim_to_complete_sentences(item) for item in extract_bullet_items(raw)]
        items = complete_bullet_items("\n".join(f"* {item}" for item in items if item.strip()))
        items = [re.sub(r"\s+", " ", item).strip() for item in items if item.strip()]
        return "\n".join(f"* {item}" for item in items)
    return trim_to_complete_sentences(raw)


def is_bullet_section_marker(artifact_type: ArtifactTypeName, marker: str) -> bool:
    if marker in {"Stawki:", "Co przygotowac:", "Relacje:", "Jak uzyc tej postaci na sesji:"}:
        return True
    return artifact_type == "pre_session_brief" and marker.startswith("## ")


def section_target_bullet_count(artifact_type: ArtifactTypeName, marker: str) -> int:
    if artifact_type == "session_hooks" and marker in {"Stawki:", "Co przygotowac:"}:
        return 4
    if artifact_type == "npc_brief" and marker in {"Relacje:", "Jak uzyc tej postaci na sesji:"}:
        return 3
    if artifact_type == "pre_session_brief" and marker.startswith("## "):
        return 3
    return 2


def section_retry_rule(artifact_type: ArtifactTypeName, marker: str) -> str:
    if artifact_type == "session_hooks":
        if marker == "Tytul:":
            return "Zwroc dokladnie jeden krotki wiersz tytulu. Bez listy, bez lamanych linii."
        if marker in {"Hook 1:", "Hook 2:", "Hook 3:"}:
            return "Zwroc 2-4 pelne zdania i zakoncz sekcje pelnym zdaniem z kropka, pytajnikiem albo wykrzyknikiem."
        if marker in {"Stawki:", "Co przygotowac:"}:
            return "Zwroc co najmniej 2 osobne bullety zaczynajace sie od '* '. Kazdy bullet zakoncz pelnym zdaniem."
    if artifact_type == "npc_brief":
        if marker == "Imie:":
            return "Zwroc tylko imie albo imie i nazwisko w jednym wierszu."
        if marker in {"Relacje:", "Jak uzyc tej postaci na sesji:"}:
            return "Zwroc co najmniej 2 osobne bullety zaczynajace sie od '* '. Kazdy bullet zakoncz pelnym zdaniem."
        return "Zwroc 2-4 pelne zdania i zakoncz sekcje pelnym zdaniem."
    if artifact_type == "pre_session_brief" and marker.startswith("## "):
        return "Zwroc co najmniej 2 osobne bullety zaczynajace sie od '* '. Kazdy bullet zakoncz pelnym zdaniem."
    return "Uzupelnij sekcje pelna i konkretna trescia."


def compact_retry_rule(artifact_type: ArtifactTypeName, marker: str) -> str:
    if artifact_type == "session_hooks" and marker in {"Hook 1:", "Hook 2:", "Hook 3:"}:
        return "Zwroc dokladnie 2 krotkie pelne zdania. Kazde zdanie ma byc konkretne i zakonczone kropka."
    if artifact_type == "npc_brief" and marker in {
        "Rola w kampanii:",
        "Pierwsze wrazenie:",
        "Motywacja:",
        "Sekret:",
    }:
        return "Zwroc dokladnie 2 krotkie pelne zdania. Kazde zdanie zakoncz kropka."
    return section_retry_rule(artifact_type, marker)


def section_min_length(artifact_type: ArtifactTypeName, marker: str) -> int:
    if artifact_type == "pre_session_brief":
        return 60 if marker == "## Campaign State" else 35
    if artifact_type == "session_hooks":
        if marker == "Tytul:":
            return 8
        if marker in {"Stawki:", "Co przygotowac:"}:
            return 35
        return 50
    if artifact_type == "scene_seed":
        return 25
    if artifact_type == "npc_brief":
        return 3 if marker == "Imie:" else 45
    if artifact_type == "twist_pack":
        return 40
    return 20


def section_needs_fill(
    *,
    artifact_type: ArtifactTypeName,
    marker: str,
    content: str,
    is_last_marker: bool,
) -> bool:
    raw = (content or "").strip()
    normalized = normalize_section_body(content)
    lowered = normalized.lower()
    if not normalized:
        return True
    if "do doprecyzowania" in lowered:
        return True
    if artifact_type == "session_hooks":
        if marker == "Tytul:":
            return "\n" in raw or len(raw.split()) > 12
        if marker in {"Hook 1:", "Hook 2:", "Hook 3:"}:
            return (
                len(normalized) < section_min_length(artifact_type, marker)
                or not ends_with_sentence_punctuation(raw)
                or sentence_count(raw) < 2
            )
        if marker in {"Stawki:", "Co przygotowac:"}:
            return complete_bullet_count(raw) < 2
    if artifact_type == "npc_brief":
        if marker == "Imie:":
            return "\n" in raw or len(normalized) < 2
        if marker in {"Relacje:", "Jak uzyc tej postaci na sesji:"}:
            return complete_bullet_count(raw) < 2
        return len(normalized) < section_min_length(artifact_type, marker) or not ends_with_sentence_punctuation(raw)
    if artifact_type == "pre_session_brief" and marker.startswith("## "):
        return complete_bullet_count(raw) < 2
    if len(normalized) < section_min_length(artifact_type, marker):
        return True
    if is_last_marker and artifact_type in {"pre_session_brief", "npc_brief"}:
        if len(normalized) >= section_min_length(artifact_type, marker) and not re.search(r"[.!?)]$", normalized):
            return True
    return False


def markers_requiring_fill(text: str, artifact_type: ArtifactTypeName) -> List[str]:
    sections = extract_artifact_sections(text, artifact_type)
    markers = artifact_required_markers(artifact_type)
    required: List[str] = []
    for idx, marker in enumerate(markers):
        if marker.startswith("# ") and not marker.startswith("## "):
            if marker not in sections:
                required.append(marker)
            continue
        content = sections.get(marker, "")
        if section_needs_fill(
            artifact_type=artifact_type,
            marker=marker,
            content=content,
            is_last_marker=idx == len(markers) - 1,
        ):
            required.append(marker)
    return required


def missing_artifact_markers(text: str, artifact_type: ArtifactTypeName) -> List[str]:
    lowered = (text or "").lower()
    return [marker for marker in artifact_required_markers(artifact_type) if marker.lower() not in lowered]


def render_artifact_section_block(marker: str, content: str) -> List[str]:
    body = (content or "").strip()
    lines: List[str] = []
    if marker.startswith("# ") and not marker.startswith("## "):
        lines.append(marker)
        return lines

    if marker.startswith("#"):
        lines.append(marker)
        lines.append("")
        if body:
            lines.extend(body.splitlines())
        else:
            lines.append("- Do doprecyzowania.")
        return lines

    if marker in {"Tytul:", "Tytul sceny:", "Imie:"} and body and "\n" not in body and not body.lstrip().startswith(("*", "-")):
        lines.append(f"{marker} {body}".rstrip())
        return lines

    lines.append(marker)
    if body:
        lines.extend(body.splitlines())
    else:
        lines.append("Do doprecyzowania.")
    return lines


def merge_artifact_sections(base_text: str, supplement_text: str, artifact_type: ArtifactTypeName) -> str:
    base_sections = extract_artifact_sections(base_text, artifact_type)
    supplement_sections = extract_artifact_sections(supplement_text, artifact_type)
    merged = {**base_sections, **supplement_sections}

    lines: List[str] = []
    for marker in artifact_required_markers(artifact_type):
        if lines:
            lines.append("")
        lines.extend(render_artifact_section_block(marker, merged.get(marker, "")))
    return "\n".join(lines).strip()


def build_placeholder_sections(artifact_type: ArtifactTypeName, markers: List[str]) -> str:
    lines: List[str] = []
    for marker in markers:
        if lines:
            lines.append("")
        if marker.startswith("#"):
            lines.extend([marker, "", "- Do doprecyzowania."])
        else:
            lines.extend([marker, "Do doprecyzowania."])
    return "\n".join(lines).strip()


def creative_section_specs(artifact_type: ArtifactTypeName) -> List[Dict[str, Any]]:
    if artifact_type == "pre_session_brief":
        return [
            {
                "marker": "## Campaign State",
                "instruction": "Daj 3 krotkie bullety o aktualnym stanie kampanii i ostatnim zwrocie sytuacji.",
                "require_canonical_name": True,
            },
            {
                "marker": "## Active Threads",
                "instruction": "Daj 3 krotkie bullety o najwazniejszych aktywnych watkach i ich aktualnym stanie.",
                "require_canonical_name": True,
            },
            {
                "marker": "## Key NPCs and Factions",
                "instruction": "Daj 3 krotkie bullety o kluczowych NPC i frakcjach istotnych przed kolejna sesja.",
                "require_canonical_name": True,
            },
            {
                "marker": "## Risks and Pressure Points",
                "instruction": "Daj 3 krotkie bullety o ryzykach, presji i potencjalnej eskalacji.",
                "require_canonical_name": True,
            },
            {
                "marker": "## Scene Opportunities",
                "instruction": "Daj 3 krotkie bullety z konkretnymi scenami do rozegrania na sesji.",
                "require_canonical_name": True,
            },
            {
                "marker": "## Prep Checklist",
                "instruction": "Daj 3 konkretne bullety rzeczy do przygotowania przez MG przed sesja.",
                "require_canonical_name": True,
            },
        ]
    if artifact_type == "session_hooks":
        return [
            {"marker": "Tytul:", "instruction": "Jedna krotka linia, 4-8 slow.", "require_canonical_name": False},
            {
                "marker": "Hook 1:",
                "instruction": "Napisz 2-4 zdania. To ma byc konkretny incydent otwierajacy sesje.",
                "require_canonical_name": True,
            },
            {
                "marker": "Hook 2:",
                "instruction": "Napisz 2-4 zdania. Ten hook ma byc wyraznie inny od poprzedniego.",
                "require_canonical_name": True,
            },
            {
                "marker": "Hook 3:",
                "instruction": "Napisz 2-4 zdania. Ten hook ma byc wyraznie inny od poprzednich.",
                "require_canonical_name": True,
            },
            {
                "marker": "Stawki:",
                "instruction": "Daj 4-6 krotkich bulletow zaczynajacych sie od '* '.",
                "require_canonical_name": False,
            },
            {
                "marker": "Co przygotowac:",
                "instruction": "Daj 4-6 krotkich bulletow zaczynajacych sie od '* '.",
                "require_canonical_name": False,
            },
        ]
    if artifact_type == "npc_brief":
        return [
            {"marker": "Imie:", "instruction": "Podaj imie albo imie i nazwisko nowej postaci.", "require_canonical_name": False},
            {
                "marker": "Rola w kampanii:",
                "instruction": "Napisz 2-4 zdania o roli tej postaci w kampanii.",
                "require_canonical_name": True,
            },
            {
                "marker": "Pierwsze wrazenie:",
                "instruction": "Napisz 2-4 zdania o wygladzie i aurze postaci.",
                "require_canonical_name": False,
            },
            {
                "marker": "Motywacja:",
                "instruction": "Napisz 2-4 zdania o motywacji i presji tej postaci.",
                "require_canonical_name": False,
            },
            {
                "marker": "Sekret:",
                "instruction": "Napisz 2-4 zdania o sekrecie lub ukrytym koszcie tej postaci.",
                "require_canonical_name": True,
            },
            {
                "marker": "Relacje:",
                "instruction": "Daj 3-5 bulletow zaczynajacych sie od '* ' o relacjach postaci.",
                "require_canonical_name": True,
            },
            {
                "marker": "Jak uzyc tej postaci na sesji:",
                "instruction": "Daj 3-5 bulletow zaczynajacych sie od '* ' z praktycznymi sposobami uzycia.",
                "require_canonical_name": True,
            },
        ]
    return []


def render_partial_artifact_sections(section_values: Dict[str, str], artifact_type: ArtifactTypeName) -> str:
    lines: List[str] = []
    for marker in artifact_required_markers(artifact_type):
        if marker not in section_values:
            continue
        if lines:
            lines.append("")
        lines.extend(render_artifact_section_block(marker, section_values[marker]))
    return "\n".join(lines).strip()


def append_missing_artifact_sections(text: str, artifact_type: ArtifactTypeName) -> str:
    missing = missing_artifact_markers(text, artifact_type)
    if not missing:
        return text.strip()

    lines = [text.rstrip()]
    for marker in missing:
        lines.extend(["", marker])
        if marker.startswith("## "):
            lines.append("- Do doprecyzowania.")
        else:
            lines.append("Do doprecyzowania.")
    return "\n".join(lines).strip()
