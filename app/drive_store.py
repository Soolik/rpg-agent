from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Sequence
from xml.etree import ElementTree

import google.auth
from googleapiclient.discovery import build

from .models_v2 import DocumentRef, WorldDocInfo, WorldEntityType
from .text_normalization import normalize_text_artifacts

GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
GOOGLE_FOLDER_MIME = "application/vnd.google-apps.folder"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TEXT_PLAIN_MIME = "text/plain"
TEXT_MARKDOWN_MIME = "text/markdown"
WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def decode_google_export_text(data: bytes) -> str:
    text = data.decode("utf-8-sig", errors="ignore")
    return normalize_text_artifacts(text)


def decode_docx_bytes(data: bytes) -> str:
    with zipfile.ZipFile(BytesIO(data)) as archive:
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


def _collapse_blank_lines(lines: List[str]) -> str:
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


@dataclass
class DriveFileInfo:
    file_id: str
    name: str
    mime_type: str
    parents: list[str]


def normalize_section_name(section: str) -> str:
    return re.sub(r"\s+", " ", section.strip().strip("# ")).strip().lower()


def replace_section_content(document_text: str, section: str, content: str) -> str:
    normalized = normalize_section_name(section)
    cleaned_content = content.strip()
    original_lines = document_text.splitlines()
    had_trailing_newline = document_text.endswith("\n")

    start_idx = None
    start_level = None
    heading_line = None

    for idx, line in enumerate(original_lines):
        match = HEADING_RE.match(line.strip())
        if not match:
            continue
        level = len(match.group(1))
        title = normalize_section_name(match.group(2))
        if title == normalized:
            start_idx = idx
            start_level = level
            heading_line = line.strip()
            break

    replacement_lines = [f"## {section.strip().strip('# ').strip()}"] if heading_line is None else [heading_line]
    replacement_lines.append("")
    if cleaned_content:
        replacement_lines.extend(cleaned_content.splitlines())

    if start_idx is None:
        base = document_text.rstrip()
        suffix = "\n\n" if base else ""
        result = base + suffix + "\n".join(replacement_lines)
        return result + "\n"

    end_idx = len(original_lines)
    for idx in range(start_idx + 1, len(original_lines)):
        match = HEADING_RE.match(original_lines[idx].strip())
        if match and len(match.group(1)) <= start_level:
            end_idx = idx
            break

    updated_lines = original_lines[:start_idx] + replacement_lines + original_lines[end_idx:]
    result = "\n".join(updated_lines).rstrip()
    if had_trailing_newline or result:
        result += "\n"
    return result


def _paragraph_text(block: dict[str, Any]) -> str:
    paragraph = block.get("paragraph") or {}
    parts: list[str] = []
    for element in paragraph.get("elements", []):
        text_run = element.get("textRun") or {}
        content = text_run.get("content")
        if content:
            parts.append(content)
    return "".join(parts)


def _heading_details(block: dict[str, Any]) -> tuple[int, str, str] | None:
    paragraph = block.get("paragraph")
    if not paragraph:
        return None
    text = _paragraph_text(block).strip()
    if not text:
        return None
    style = ((paragraph.get("paragraphStyle") or {}).get("namedStyleType") or "").upper()
    if style.startswith("HEADING_"):
        try:
            return int(style.split("_", 1)[1]), normalize_section_name(text), text
        except Exception:
            pass
    match = HEADING_RE.match(text)
    if match:
        return len(match.group(1)), normalize_section_name(match.group(2)), match.group(0).strip()
    return None


@dataclass
class DriveStore:
    """
    Real adapter for Google Drive / Docs used by worldbuilding v2 endpoints.

    Expected envs (recommended):
    - ADMIN_FOLDER_ID
    - BIBLE_FOLDER_ID
    - SESSIONS_FOLDER_ID
    - NPC_FOLDER_ID
    - LOCATIONS_FOLDER_ID
    - FACTIONS_FOLDER_ID
    - THREADS_FOLDER_ID
    - SECRETS_FOLDER_ID
    - OUTPUTS_FOLDER_ID

    Optional fallback core docs:
    - BIBLE_DOC_ID
    - GLOSSARY_DOC_ID
    - RULES_DOC_ID
    - THREADS_DOC_ID
    """

    folder_map: Dict[str, str]
    core_doc_map: Dict[str, str]
    credentials_provider: Optional[Callable[[Sequence[str]], object]] = None

    def _credentials(self, scopes: Sequence[str]):
        if self.credentials_provider is not None:
            return self.credentials_provider(scopes)
        creds, _ = google.auth.default(scopes=list(scopes))
        return creds

    def _drive(self):
        creds = self._credentials(["https://www.googleapis.com/auth/drive"])
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    def _docs(self):
        creds = self._credentials(["https://www.googleapis.com/auth/documents", "https://www.googleapis.com/auth/drive"])
        return build("docs", "v1", credentials=creds, cache_discovery=False)

    def _export_plain_text(self, file_id: str) -> str:
        drive = self._drive()
        data = drive.files().export(fileId=file_id, mimeType="text/plain").execute()
        return decode_google_export_text(data)

    def _download_file_bytes(self, file_id: str) -> bytes:
        drive = self._drive()
        return drive.files().get_media(fileId=file_id).execute()

    def _folder_query_docs(self, folder_id: str) -> List[WorldDocInfo]:
        drive = self._drive()
        out: List[WorldDocInfo] = []
        page_token = None
        while True:
            resp = drive.files().list(
                q=f"'{folder_id}' in parents and mimeType='{GOOGLE_DOC_MIME}' and trashed=false",
                fields="nextPageToken, files(id, name, parents)",
                pageToken=page_token,
                pageSize=200,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()
            for f in resp.get("files", []):
                out.append(
                    WorldDocInfo(
                        folder=self._logical_folder_for_id(folder_id),
                        title=f["name"],
                        doc_id=f["id"],
                        path_hint=f"{self._logical_folder_for_id(folder_id)}/{f['name']}",
                        entity_type=self._infer_entity_type(self._logical_folder_for_id(folder_id), f["name"]),
                    )
                )
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return out

    def _logical_folder_for_id(self, folder_id: str) -> str:
        for logical, fid in self.folder_map.items():
            if fid == folder_id:
                return logical
        return "unknown"

    def _infer_entity_type(self, folder: str, title: str) -> WorldEntityType:
        title_l = title.lower()
        if folder == "03 NPC" or title_l.startswith("npc_"):
            return WorldEntityType.npc
        if folder == "04 Locations" or title_l.startswith("location_"):
            return WorldEntityType.location
        if folder == "05 Factions" or title_l.startswith("faction_"):
            return WorldEntityType.faction
        if folder == "06 Threads" or title_l.startswith("thread_"):
            return WorldEntityType.thread
        if folder == "07 Secrets" or title_l.startswith("secret_"):
            return WorldEntityType.secret
        if folder == "02 Sessions" or title_l.startswith("session_"):
            return WorldEntityType.session
        if folder == "08 Outputs":
            return WorldEntityType.output
        if title == "Campaign Bible":
            return WorldEntityType.bible
        if title == "Glossary":
            return WorldEntityType.glossary
        if "Rules" in title:
            return WorldEntityType.rules
        return WorldEntityType.other

    def list_world_docs(self) -> List[WorldDocInfo]:
        docs: List[WorldDocInfo] = []
        seen_ids = set()

        for logical_folder, folder_id in self.folder_map.items():
            if not folder_id:
                continue
            for doc in self._folder_query_docs(folder_id):
                if doc.doc_id and doc.doc_id in seen_ids:
                    continue
                if doc.doc_id:
                    seen_ids.add(doc.doc_id)
                docs.append(doc)

        # Fallback core docs if folders are not yet configured.
        for title, doc_id in self.core_doc_map.items():
            if not doc_id or doc_id in seen_ids:
                continue
            try:
                drive = self._drive()
                meta = drive.files().get(fileId=doc_id, fields="id,name", supportsAllDrives=True).execute()
                inferred_folder = self._infer_core_folder(title)
                docs.append(
                    WorldDocInfo(
                        folder=inferred_folder,
                        title=meta.get("name", title),
                        doc_id=meta.get("id"),
                        path_hint=f"{inferred_folder}/{meta.get('name', title)}",
                        entity_type=self._infer_entity_type(inferred_folder, meta.get("name", title)),
                    )
                )
                seen_ids.add(doc_id)
            except Exception:
                continue

        docs.sort(key=lambda d: (d.folder, d.title.lower()))
        return docs

    def list_drive_folder_files(self, folder_id: str) -> List[DriveFileInfo]:
        drive = self._drive()
        out: List[DriveFileInfo] = []
        page_token = None
        while True:
            resp = drive.files().list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="nextPageToken, files(id, name, mimeType, parents)",
                pageToken=page_token,
                pageSize=200,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()
            for item in resp.get("files", []):
                out.append(
                    DriveFileInfo(
                        file_id=item["id"],
                        name=item["name"],
                        mime_type=item["mimeType"],
                        parents=item.get("parents", []),
                    )
                )
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        out.sort(key=lambda item: item.name.lower())
        return out

    def read_drive_file_text(self, file_id: str, mime_type: str) -> str:
        if mime_type == GOOGLE_DOC_MIME:
            return self._export_plain_text(file_id)
        if mime_type in {TEXT_PLAIN_MIME, TEXT_MARKDOWN_MIME}:
            return decode_google_export_text(self._download_file_bytes(file_id))
        if mime_type == DOCX_MIME:
            return decode_docx_bytes(self._download_file_bytes(file_id))
        raise ValueError(f"Unsupported Drive file type: {mime_type}")

    def _infer_core_folder(self, title: str) -> str:
        if title in ("Campaign Bible", "Glossary", "Rules And Tone"):
            return "01 Bible"
        if title == "Thread Tracker":
            return "06 Threads"
        return "unknown"

    def _resolve_doc_ref(self, doc_ref: DocumentRef) -> DocumentRef:
        if doc_ref.doc_id:
            return doc_ref
        found = self.find_doc(folder=doc_ref.folder, title=doc_ref.title)
        if not found or not found.doc_id:
            raise FileNotFoundError(f"Document not found: {doc_ref.folder}/{doc_ref.title}")
        return DocumentRef(folder=found.folder, title=found.title, doc_id=found.doc_id, path_hint=found.path_hint)

    def read_doc(self, doc_ref: DocumentRef) -> str:
        doc_ref = self._resolve_doc_ref(doc_ref)
        return self._export_plain_text(doc_ref.doc_id)

    def create_doc(self, folder: str, title: str, content: str, entity_type: WorldEntityType = WorldEntityType.other) -> WorldDocInfo:
        drive = self._drive()
        docs = self._docs()
        folder_id = self.folder_map.get(folder)
        body = {"name": title, "mimeType": GOOGLE_DOC_MIME}
        if folder_id:
            body["parents"] = [folder_id]

        created = drive.files().create(
            body=body,
            fields="id, name, parents",
            supportsAllDrives=True,
        ).execute()
        doc_id = created["id"]

        if content.strip():
            docs.documents().batchUpdate(
                documentId=doc_id,
                body={"requests": [{"insertText": {"location": {"index": 1}, "text": content}}]},
            ).execute()

        return WorldDocInfo(
            folder=folder,
            title=title,
            doc_id=doc_id,
            path_hint=f"{folder}/{title}",
            entity_type=entity_type,
        )

    def append_doc(self, doc_ref: DocumentRef, content: str) -> None:
        if not content:
            return
        docs = self._docs()
        doc_ref = self._resolve_doc_ref(doc_ref)

        doc = docs.documents().get(documentId=doc_ref.doc_id).execute()
        end_index = doc.get("body", {}).get("content", [{}])[-1].get("endIndex", 1)
        insert_at = max(1, end_index - 1)
        text = content
        if not text.startswith("\n"):
            text = "\n\n" + text
        docs.documents().batchUpdate(
            documentId=doc_ref.doc_id,
            body={"requests": [{"insertText": {"location": {"index": insert_at}, "text": text}}]},
        ).execute()

    def replace_doc(self, doc_ref: DocumentRef, content: str) -> None:
        docs = self._docs()
        doc_ref = self._resolve_doc_ref(doc_ref)

        doc = docs.documents().get(documentId=doc_ref.doc_id).execute()
        end_index = doc.get("body", {}).get("content", [{}])[-1].get("endIndex", 1)
        requests = []
        if end_index > 2:
            requests.append({"deleteContentRange": {"range": {"startIndex": 1, "endIndex": end_index - 1}}})
        if content:
            requests.append({"insertText": {"location": {"index": 1}, "text": content}})
        if requests:
            docs.documents().batchUpdate(documentId=doc_ref.doc_id, body={"requests": requests}).execute()

    def replace_section(self, doc_ref: DocumentRef, section: str, content: str) -> None:
        docs = self._docs()
        doc_ref = self._resolve_doc_ref(doc_ref)
        document = docs.documents().get(documentId=doc_ref.doc_id).execute()
        body_content = document.get("body", {}).get("content", [])
        document_end = max(1, (body_content[-1].get("endIndex", 1) if body_content else 1) - 1)
        normalized_section = normalize_section_name(section)
        cleaned_content = (content or "").strip()

        found_block = None
        found_level = None
        for block in body_content:
            details = _heading_details(block)
            if not details:
                continue
            level, normalized_title, _ = details
            if normalized_title == normalized_section:
                found_block = block
                found_level = level
                break

        requests: list[dict[str, Any]] = []
        if found_block is not None and found_level is not None:
            body_start = found_block.get("endIndex", 1)
            body_end = document_end
            for block in body_content:
                if block.get("startIndex", 0) <= found_block.get("startIndex", 0):
                    continue
                details = _heading_details(block)
                if not details:
                    continue
                level, _, _ = details
                if level <= found_level:
                    body_end = block.get("startIndex", document_end)
                    break
            if body_end > body_start:
                requests.append(
                    {
                        "deleteContentRange": {
                            "range": {
                                "startIndex": body_start,
                                "endIndex": body_end,
                            }
                        }
                    }
                )
            insertion = "\n"
            if cleaned_content:
                insertion += cleaned_content
            if not insertion.endswith("\n"):
                insertion += "\n"
            requests.append(
                {
                    "insertText": {
                        "location": {"index": body_start},
                        "text": insertion,
                    }
                }
            )
        else:
            heading = section.strip().strip("# ").strip()
            insertion = f"\n\n## {heading}\n\n"
            if cleaned_content:
                insertion += cleaned_content
                if not insertion.endswith("\n"):
                    insertion += "\n"
            requests.append(
                {
                    "insertText": {
                        "location": {"index": document_end},
                        "text": insertion,
                    }
                }
            )

        if requests:
            docs.documents().batchUpdate(documentId=doc_ref.doc_id, body={"requests": requests}).execute()

    def find_doc(self, *, folder: Optional[str] = None, title: Optional[str] = None, doc_id: Optional[str] = None) -> Optional[WorldDocInfo]:
        docs = self.list_world_docs()
        for doc in docs:
            if doc_id and doc.doc_id == doc_id:
                return doc
            if folder and title and doc.folder == folder and doc.title == title:
                return doc
        return None
