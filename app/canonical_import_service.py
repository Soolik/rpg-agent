from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional
from xml.etree import ElementTree

from .drive_store import DriveStore
from .models_v2 import DocumentRef, WorldEntityType
from .text_normalization import normalize_text_artifacts


SUPPORTED_IMPORT_EXTENSIONS = {".txt", ".md", ".docx"}
TIMESTAMP_SUFFIX_RE = re.compile(r"_\d{4}_\d{2}_\d{2}_\d{4}$")
WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
SKIPPED_SUPPORT_FILES = {"manifest.txt", "readme.txt", "readme.md"}


@dataclass
class CanonicalImportFileResult:
    source_path: str
    source_name: str
    format: str
    folder: str | None
    title: str | None
    entity_type: str | None
    action: str
    status: str
    chars: int = 0
    doc_id: str | None = None
    path: str | None = None
    message: str = ""


@dataclass
class CanonicalImportResult:
    source_path: str
    dry_run: bool
    imported_count: int
    created_count: int
    updated_count: int
    skipped_count: int
    results: list[CanonicalImportFileResult]
    reindex_result: dict | None = None
    warnings: list[str] | None = None


def _collapse_blank_lines(lines: Iterable[str]) -> str:
    collapsed: list[str] = []
    blank = False
    for raw in lines:
        line = raw.rstrip()
        if not line:
            if blank:
                continue
            collapsed.append("")
            blank = True
            continue
        collapsed.append(line)
        blank = False
    return "\n".join(collapsed).strip()


def _normalize_import_title(path: Path) -> str:
    stem = TIMESTAMP_SUFFIX_RE.sub("", path.stem).strip()
    return re.sub(r"\s+", " ", stem).strip()


def _extract_docx_inline_text(node: ElementTree.Element) -> str:
    parts: list[str] = []
    for child in node.iter():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "t":
            parts.append(child.text or "")
        elif tag == "tab":
            parts.append("\t")
        elif tag in {"br", "cr"}:
            parts.append("\n")
    return "".join(parts)


def read_local_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("word/document.xml")

    root = ElementTree.fromstring(raw)
    body = root.find("./w:body", WORD_NS)
    if body is None:
        return ""

    lines: list[str] = []
    for child in list(body):
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = _extract_docx_inline_text(child).strip()
            lines.append(text)
        elif tag == "tbl":
            for row in child.findall("./w:tr", WORD_NS):
                cells = [_collapse_blank_lines(_extract_docx_inline_text(cell).splitlines()) for cell in row.findall("./w:tc", WORD_NS)]
                cells = [cell.strip() for cell in cells]
                if any(cells):
                    lines.append(" | ".join(cells))

    return normalize_text_artifacts(_collapse_blank_lines(lines))


def read_local_canonical_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return read_local_docx(path)
    if suffix in {".txt", ".md"}:
        return normalize_text_artifacts(path.read_text(encoding="utf-8-sig"))
    raise ValueError(f"Unsupported file type: {path.suffix}")


def classify_canonical_file(path: Path) -> tuple[str, str, WorldEntityType]:
    title = _normalize_import_title(path)
    key = title.lower()

    exact_map = {
        "campaign bible": ("01 Bible", "Campaign Bible", WorldEntityType.bible),
        "glossary": ("01 Bible", "Glossary", WorldEntityType.glossary),
        "rules and tone": ("01 Bible", "Rules And Tone", WorldEntityType.rules),
        "thread tracker": ("06 Threads", "Thread Tracker", WorldEntityType.thread),
        "npcs": ("03 NPC", "NPCs", WorldEntityType.npc),
        "places": ("04 Locations", "Places", WorldEntityType.location),
        "locations": ("04 Locations", "Locations", WorldEntityType.location),
        "factions": ("05 Factions", "Factions", WorldEntityType.faction),
        "secrets": ("07 Secrets", "Secrets", WorldEntityType.secret),
        "events": ("02 Sessions", "Events", WorldEntityType.session),
        "sessions log": ("02 Sessions", "Sessions Log", WorldEntityType.session),
    }
    if key in exact_map:
        return exact_map[key]

    if "rozdzial" in key or "chapter" in key or "dossier" in key or key.startswith("session "):
        return "02 Sessions", title, WorldEntityType.session
    if "guide" in key or "przewodnik" in key or "shackles" in key:
        return "01 Bible", title, WorldEntityType.other
    if "thread" in key:
        return "06 Threads", title, WorldEntityType.thread

    return "01 Bible", title, WorldEntityType.other


@dataclass
class CanonicalImportService:
    drive_store: DriveStore
    reindex_fn: Optional[Callable[[list[DocumentRef]], dict]] = None

    def import_folder(
        self,
        *,
        source_path: str,
        dry_run: bool = True,
        replace_existing: bool = True,
        reindex_after_import: bool = True,
    ) -> CanonicalImportResult:
        root = Path(source_path).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")
        if not root.is_dir():
            raise ValueError(f"Source path is not a directory: {source_path}")

        results: list[CanonicalImportFileResult] = []
        reindex_targets: list[DocumentRef] = []
        warnings: list[str] = []

        files = sorted([path for path in root.rglob("*") if path.is_file()], key=lambda item: str(item).lower())
        for path in files:
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_IMPORT_EXTENSIONS:
                results.append(
                    CanonicalImportFileResult(
                        source_path=str(path),
                        source_name=path.name,
                        format=suffix.lstrip(".") or "unknown",
                        folder=None,
                        title=None,
                        entity_type=None,
                        action="skip",
                        status="skipped",
                        message="Unsupported extension.",
                    )
                )
                continue
            if path.name.lower() in SKIPPED_SUPPORT_FILES:
                results.append(
                    CanonicalImportFileResult(
                        source_path=str(path),
                        source_name=path.name,
                        format=suffix.lstrip("."),
                        folder=None,
                        title=None,
                        entity_type=None,
                        action="skip",
                        status="skipped",
                        message="Support file skipped by default.",
                    )
                )
                continue

            folder, title, entity_type = classify_canonical_file(path)
            text = read_local_canonical_text(path).strip()
            if not text:
                results.append(
                    CanonicalImportFileResult(
                        source_path=str(path),
                        source_name=path.name,
                        format=suffix.lstrip("."),
                        folder=folder,
                        title=title,
                        entity_type=entity_type.value,
                        action="skip",
                        status="skipped",
                        message="No importable text extracted from file.",
                    )
                )
                continue

            existing = self.drive_store.find_doc(folder=folder, title=title)
            if existing and not replace_existing:
                results.append(
                    CanonicalImportFileResult(
                        source_path=str(path),
                        source_name=path.name,
                        format=suffix.lstrip("."),
                        folder=folder,
                        title=title,
                        entity_type=entity_type.value,
                        action="skip_existing",
                        status="skipped",
                        chars=len(text),
                        doc_id=existing.doc_id,
                        path=existing.path_hint,
                        message="Existing document left unchanged because replace_existing is false.",
                    )
                )
                continue

            if dry_run:
                action = "replace_doc" if existing else "create_doc"
                results.append(
                    CanonicalImportFileResult(
                        source_path=str(path),
                        source_name=path.name,
                        format=suffix.lstrip("."),
                        folder=folder,
                        title=title,
                        entity_type=entity_type.value,
                        action=action,
                        status="planned",
                        chars=len(text),
                        doc_id=existing.doc_id if existing else None,
                        path=existing.path_hint if existing else f"{folder}/{title}",
                        message="Dry run only.",
                    )
                )
                continue

            if existing and existing.doc_id:
                target = DocumentRef(folder=existing.folder, title=existing.title, doc_id=existing.doc_id, path_hint=existing.path_hint)
                self.drive_store.replace_doc(target, text)
                reindex_targets.append(target)
                results.append(
                    CanonicalImportFileResult(
                        source_path=str(path),
                        source_name=path.name,
                        format=suffix.lstrip("."),
                        folder=folder,
                        title=title,
                        entity_type=entity_type.value,
                        action="replace_doc",
                        status="updated",
                        chars=len(text),
                        doc_id=existing.doc_id,
                        path=existing.path_hint,
                        message="Existing Google Doc replaced from local canonical file.",
                    )
                )
            else:
                created = self.drive_store.create_doc(folder=folder, title=title, content=text, entity_type=entity_type)
                target = DocumentRef(folder=created.folder, title=created.title, doc_id=created.doc_id, path_hint=created.path_hint)
                reindex_targets.append(target)
                results.append(
                    CanonicalImportFileResult(
                        source_path=str(path),
                        source_name=path.name,
                        format=suffix.lstrip("."),
                        folder=folder,
                        title=title,
                        entity_type=entity_type.value,
                        action="create_doc",
                        status="created",
                        chars=len(text),
                        doc_id=created.doc_id,
                        path=created.path_hint,
                        message="Google Doc created from local canonical file.",
                    )
                )

        reindex_result = None
        if not dry_run and reindex_after_import and reindex_targets and self.reindex_fn:
            try:
                reindex_result = self.reindex_fn(reindex_targets)
            except Exception as exc:
                warnings.append(f"Reindex after import failed: {exc}")
                reindex_result = {"ok": False, "error": str(exc)}

        created_count = sum(1 for item in results if item.status == "created")
        updated_count = sum(1 for item in results if item.status == "updated")
        skipped_count = sum(1 for item in results if item.status == "skipped")
        imported_count = created_count + updated_count

        return CanonicalImportResult(
            source_path=str(root),
            dry_run=dry_run,
            imported_count=imported_count,
            created_count=created_count,
            updated_count=updated_count,
            skipped_count=skipped_count,
            results=results,
            reindex_result=reindex_result,
            warnings=warnings,
        )
