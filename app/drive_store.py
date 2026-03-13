from __future__ import annotations

import os
from dataclasses import dataclass
from html import unescape
from typing import Dict, List, Optional

import google.auth
from googleapiclient.discovery import build

from .models_v2 import DocumentRef, WorldDocInfo, WorldEntityType

GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
GOOGLE_FOLDER_MIME = "application/vnd.google-apps.folder"


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

    def _drive(self):
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive"])
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    def _docs(self):
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/documents", "https://www.googleapis.com/auth/drive"])
        return build("docs", "v1", credentials=creds, cache_discovery=False)

    def _export_plain_text(self, file_id: str) -> str:
        drive = self._drive()
        data = drive.files().export(fileId=file_id, mimeType="text/plain").execute()
        return data.decode("utf-8", errors="ignore")

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

    def _infer_core_folder(self, title: str) -> str:
        if title in ("Campaign Bible", "Glossary", "Rules And Tone"):
            return "01 Bible"
        if title == "Thread Tracker":
            return "06 Threads"
        return "unknown"

    def read_doc(self, doc_ref: DocumentRef) -> str:
        if not doc_ref.doc_id:
            found = self.find_doc(folder=doc_ref.folder, title=doc_ref.title)
            if not found or not found.doc_id:
                raise FileNotFoundError(f"Document not found: {doc_ref.folder}/{doc_ref.title}")
            doc_ref = DocumentRef(folder=found.folder, title=found.title, doc_id=found.doc_id, path_hint=found.path_hint)
        return self._export_plain_text(doc_ref.doc_id)

    def create_doc(self, folder: str, title: str, content: str, entity_type: WorldEntityType = WorldEntityType.other) -> WorldDocInfo:
        drive = self._drive()
        docs = self._docs()

        created = docs.documents().create(body={"title": title}).execute()
        doc_id = created["documentId"]

        # optional move to target folder
        folder_id = self.folder_map.get(folder)
        if folder_id:
            meta = drive.files().get(fileId=doc_id, fields="parents", supportsAllDrives=True).execute()
            prev_parents = ",".join(meta.get("parents", []))
            drive.files().update(
                fileId=doc_id,
                addParents=folder_id,
                removeParents=prev_parents,
                fields="id, parents",
                supportsAllDrives=True,
            ).execute()

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
        if not doc_ref.doc_id:
            found = self.find_doc(folder=doc_ref.folder, title=doc_ref.title)
            if not found or not found.doc_id:
                raise FileNotFoundError(f"Document not found: {doc_ref.folder}/{doc_ref.title}")
            doc_ref = DocumentRef(folder=found.folder, title=found.title, doc_id=found.doc_id, path_hint=found.path_hint)

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
        if not doc_ref.doc_id:
            found = self.find_doc(folder=doc_ref.folder, title=doc_ref.title)
            if not found or not found.doc_id:
                raise FileNotFoundError(f"Document not found: {doc_ref.folder}/{doc_ref.title}")
            doc_ref = DocumentRef(folder=found.folder, title=found.title, doc_id=found.doc_id, path_hint=found.path_hint)

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
        # Safe MVP: append a clearly marked section update instead of brittle in-place surgery.
        normalized = section.strip().strip("# ")
        block = f"\n\n## {normalized}\n\n{content.strip()}\n"
        self.append_doc(doc_ref, block)

    def find_doc(self, *, folder: Optional[str] = None, title: Optional[str] = None, doc_id: Optional[str] = None) -> Optional[WorldDocInfo]:
        docs = self.list_world_docs()
        for doc in docs:
            if doc_id and doc.doc_id == doc_id:
                return doc
            if folder and title and doc.folder == folder and doc.title == title:
                return doc
        return None
