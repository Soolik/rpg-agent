from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from googleapiclient.errors import HttpError

from .drive_store import (
    DOCX_MIME,
    GOOGLE_DOC_MIME,
    TEXT_MARKDOWN_MIME,
    TEXT_PLAIN_MIME,
    DriveFileInfo,
    DriveStore,
    decode_docx_bytes,
    decode_google_export_text,
)
from .models_v2 import DocumentRef, WorldEntityType
from .text_normalization import normalize_text_artifacts


SUPPORTED_IMPORT_EXTENSIONS = {".txt", ".md", ".docx"}
SUPPORTED_DRIVE_MIME_TYPES = {GOOGLE_DOC_MIME, TEXT_PLAIN_MIME, TEXT_MARKDOWN_MIME, DOCX_MIME}
TIMESTAMP_SUFFIX_RE = re.compile(r"_\d{4}_\d{2}_\d{2}_\d{4}$")
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
    error_count: int
    results: list[CanonicalImportFileResult]
    reindex_result: dict | None = None
    warnings: list[str] | None = None


@dataclass
class CanonicalImportSourceFile:
    source_path: str
    source_name: str
    format: str
    text: str


def _normalize_import_title(name: str) -> str:
    stem = Path(name).stem
    stem = TIMESTAMP_SUFFIX_RE.sub("", stem).strip()
    return re.sub(r"\s+", " ", stem).strip()


def read_local_canonical_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return decode_docx_bytes(path.read_bytes())
    if suffix in {".txt", ".md"}:
        return normalize_text_artifacts(path.read_text(encoding="utf-8-sig"))
    raise ValueError(f"Unsupported file type: {path.suffix}")


def classify_canonical_name(name: str) -> tuple[str, str, WorldEntityType]:
    title = _normalize_import_title(name)
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


def classify_canonical_file(path: Path) -> tuple[str, str, WorldEntityType]:
    return classify_canonical_name(path.name)


@dataclass
class CanonicalImportService:
    drive_store: DriveStore
    reindex_fn: Optional[Callable[[list[DocumentRef]], dict]] = None

    def _describe_write_error(self, exc: Exception) -> str:
        if isinstance(exc, HttpError):
            status = getattr(getattr(exc, "resp", None), "status", None)
            reason = None
            detail_message = None
            content = getattr(exc, "content", None)
            if content:
                try:
                    payload = json.loads(content.decode("utf-8", errors="ignore"))
                    errors = payload.get("error", {}).get("errors", [])
                    if errors:
                        detail_message = errors[0].get("message")
                        reason = errors[0].get("reason")
                except Exception:
                    pass
            if reason == "storageQuotaExceeded":
                return "Drive storage quota exceeded while creating a Google Doc."
            if status == 403:
                return detail_message or "Google Drive denied write access for this document."
            if status == 404:
                return detail_message or "Target Google Drive document or folder was not found."
            if detail_message:
                return detail_message
        return str(exc)

    def _iter_local_files(self, source_path: str) -> Iterable[CanonicalImportSourceFile]:
        root = Path(source_path).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")
        if not root.is_dir():
            raise ValueError(f"Source path is not a directory: {source_path}")

        files = sorted([path for path in root.rglob("*") if path.is_file()], key=lambda item: str(item).lower())
        for path in files:
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_IMPORT_EXTENSIONS:
                yield CanonicalImportSourceFile(
                    source_path=str(path),
                    source_name=path.name,
                    format=suffix.lstrip(".") or "unknown",
                    text="__UNSUPPORTED__",
                )
                continue
            if path.name.lower() in SKIPPED_SUPPORT_FILES:
                yield CanonicalImportSourceFile(
                    source_path=str(path),
                    source_name=path.name,
                    format=suffix.lstrip("."),
                    text="__SKIP_SUPPORT__",
                )
                continue
            yield CanonicalImportSourceFile(
                source_path=str(path),
                source_name=path.name,
                format=suffix.lstrip("."),
                text=read_local_canonical_text(path).strip(),
            )

    def _read_drive_source_file(self, item: DriveFileInfo) -> CanonicalImportSourceFile:
        if item.name.lower() in SKIPPED_SUPPORT_FILES:
            return CanonicalImportSourceFile(
                source_path=f"gdrive://{item.file_id}",
                source_name=item.name,
                format=item.mime_type,
                text="__SKIP_SUPPORT__",
            )
        if item.mime_type not in SUPPORTED_DRIVE_MIME_TYPES:
            return CanonicalImportSourceFile(
                source_path=f"gdrive://{item.file_id}",
                source_name=item.name,
                format=item.mime_type,
                text="__UNSUPPORTED__",
            )
        try:
            text = self.drive_store.read_drive_file_text(item.file_id, item.mime_type).strip()
        except HttpError as exc:
            status = getattr(getattr(exc, "resp", None), "status", None)
            if status == 403:
                raise PermissionError(f"Drive file is not accessible: {item.name}") from exc
            if status == 404:
                raise FileNotFoundError(f"Drive file not found: {item.name}") from exc
            raise
        return CanonicalImportSourceFile(
            source_path=f"gdrive://{item.file_id}",
            source_name=item.name,
            format=item.mime_type,
            text=text,
        )

    def _iter_drive_files(self, folder_id: str) -> Iterable[CanonicalImportSourceFile]:
        try:
            items = self.drive_store.list_drive_folder_files(folder_id)
        except HttpError as exc:
            status = getattr(getattr(exc, "resp", None), "status", None)
            if status == 403:
                raise PermissionError(f"Drive folder is not accessible: {folder_id}") from exc
            if status == 404:
                raise FileNotFoundError(f"Drive folder not found: {folder_id}") from exc
            raise
        if not items:
            raise FileNotFoundError(f"Drive folder is empty or not accessible: {folder_id}")
        for item in items:
            yield self._read_drive_source_file(item)

    def _process_sources(
        self,
        *,
        source_label: str,
        source_files: Iterable[CanonicalImportSourceFile],
        dry_run: bool,
        replace_existing: bool,
        reindex_after_import: bool,
    ) -> CanonicalImportResult:
        results: list[CanonicalImportFileResult] = []
        reindex_targets: list[DocumentRef] = []
        warnings: list[str] = []

        for item in source_files:
            if item.text == "__UNSUPPORTED__":
                results.append(
                    CanonicalImportFileResult(
                        source_path=item.source_path,
                        source_name=item.source_name,
                        format=item.format,
                        folder=None,
                        title=None,
                        entity_type=None,
                        action="skip",
                        status="skipped",
                        message="Unsupported extension or Drive MIME type.",
                    )
                )
                continue
            if item.text == "__SKIP_SUPPORT__":
                results.append(
                    CanonicalImportFileResult(
                        source_path=item.source_path,
                        source_name=item.source_name,
                        format=item.format,
                        folder=None,
                        title=None,
                        entity_type=None,
                        action="skip",
                        status="skipped",
                        message="Support file skipped by default.",
                    )
                )
                continue

            folder, title, entity_type = classify_canonical_name(item.source_name)
            text = item.text.strip()
            if not text:
                results.append(
                    CanonicalImportFileResult(
                        source_path=item.source_path,
                        source_name=item.source_name,
                        format=item.format,
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
                        source_path=item.source_path,
                        source_name=item.source_name,
                        format=item.format,
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
                        source_path=item.source_path,
                        source_name=item.source_name,
                        format=item.format,
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
                try:
                    self.drive_store.replace_doc(target, text)
                    reindex_targets.append(target)
                    results.append(
                        CanonicalImportFileResult(
                            source_path=item.source_path,
                            source_name=item.source_name,
                            format=item.format,
                            folder=folder,
                            title=title,
                            entity_type=entity_type.value,
                            action="replace_doc",
                            status="updated",
                            chars=len(text),
                            doc_id=existing.doc_id,
                            path=existing.path_hint,
                            message="Existing Google Doc replaced from canonical source.",
                        )
                    )
                except Exception as exc:
                    message = self._describe_write_error(exc)
                    warnings.append(f"replace_doc failed for {folder}/{title}: {message}")
                    results.append(
                        CanonicalImportFileResult(
                            source_path=item.source_path,
                            source_name=item.source_name,
                            format=item.format,
                            folder=folder,
                            title=title,
                            entity_type=entity_type.value,
                            action="replace_doc",
                            status="error",
                            chars=len(text),
                            doc_id=existing.doc_id,
                            path=existing.path_hint,
                            message=message,
                        )
                    )
            else:
                try:
                    created = self.drive_store.create_doc(folder=folder, title=title, content=text, entity_type=entity_type)
                    target = DocumentRef(folder=created.folder, title=created.title, doc_id=created.doc_id, path_hint=created.path_hint)
                    reindex_targets.append(target)
                    results.append(
                        CanonicalImportFileResult(
                            source_path=item.source_path,
                            source_name=item.source_name,
                            format=item.format,
                            folder=folder,
                            title=title,
                            entity_type=entity_type.value,
                            action="create_doc",
                            status="created",
                            chars=len(text),
                            doc_id=created.doc_id,
                            path=created.path_hint,
                            message="Google Doc created from canonical source.",
                        )
                    )
                except Exception as exc:
                    message = self._describe_write_error(exc)
                    warnings.append(f"create_doc failed for {folder}/{title}: {message}")
                    results.append(
                        CanonicalImportFileResult(
                            source_path=item.source_path,
                            source_name=item.source_name,
                            format=item.format,
                            folder=folder,
                            title=title,
                            entity_type=entity_type.value,
                            action="create_doc",
                            status="error",
                            chars=len(text),
                            doc_id=None,
                            path=f"{folder}/{title}",
                            message=message,
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
        error_count = sum(1 for item in results if item.status == "error")
        imported_count = created_count + updated_count

        return CanonicalImportResult(
            source_path=source_label,
            dry_run=dry_run,
            imported_count=imported_count,
            created_count=created_count,
            updated_count=updated_count,
            skipped_count=skipped_count,
            error_count=error_count,
            results=results,
            reindex_result=reindex_result,
            warnings=warnings,
        )

    def import_folder(
        self,
        *,
        source_path: str,
        dry_run: bool = True,
        replace_existing: bool = True,
        reindex_after_import: bool = True,
    ) -> CanonicalImportResult:
        return self._process_sources(
            source_label=str(Path(source_path).expanduser()),
            source_files=self._iter_local_files(source_path),
            dry_run=dry_run,
            replace_existing=replace_existing,
            reindex_after_import=reindex_after_import,
        )

    def import_drive_folder(
        self,
        *,
        folder_id: str,
        dry_run: bool = True,
        replace_existing: bool = True,
        reindex_after_import: bool = True,
    ) -> CanonicalImportResult:
        return self._process_sources(
            source_label=f"gdrive://{folder_id}",
            source_files=self._iter_drive_files(folder_id),
            dry_run=dry_run,
            replace_existing=replace_existing,
            reindex_after_import=reindex_after_import,
        )
