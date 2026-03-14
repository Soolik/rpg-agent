from __future__ import annotations

import re


MOJIBAKE_RE = re.compile(
    "(\u00c3.|\u00c5.|\u00c4.|\u00c2.|\u00e2\u20ac\u2122|\u00e2\u20ac\u0153|"
    "\u00e2\u20ac\u009d|\u00e2\u20ac\u201c|\u00e2\u20ac\u201d|\u00e2\u20ac\u00a6|"
    "\u00e2\u20ac|\u00ca.)"
)


def normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def mojibake_score(text: str) -> int:
    if not text:
        return 0
    return len(MOJIBAKE_RE.findall(text))


def repair_common_mojibake(text: str) -> str:
    best = normalize_newlines(text).lstrip("\ufeff")
    best_score = mojibake_score(best)
    if best_score == 0:
        return best

    current = best
    for _ in range(2):
        improved = False
        for source_encoding in ("cp1252", "latin-1"):
            try:
                candidate = current.encode(source_encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            candidate = normalize_newlines(candidate).lstrip("\ufeff")
            candidate_score = mojibake_score(candidate)
            if candidate_score < best_score:
                best = candidate
                best_score = candidate_score
                current = candidate
                improved = True
        if not improved:
            break
    return best


def normalize_text_artifacts(text: str) -> str:
    return repair_common_mojibake(text)
