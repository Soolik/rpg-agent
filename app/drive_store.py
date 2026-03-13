from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .models_v2 import DocumentRef, WorldDocInfo, WorldEntityType


@dataclass
class DriveStore:
    """
    Adapter warstwy Google Drive / Docs.

    Na start to jest świadomy stub z interfejsem do podmiany.
    W prawdziwej implementacji podepnij tu Google Drive API i Google Docs API.
    """

    folder_map: Dict[str, str]

    def list_world_docs(self) -> List[WorldDocInfo]:
        # TODO: Replace with actual Drive folder traversal.
        return []

    def read_doc(self, doc_ref: DocumentRef) -> str:
        # TODO: Replace with actual Google Docs export/read.
        raise NotImplementedError("Implement DriveStore.read_doc() using Google Docs API")

    def create_doc(self, folder: str, title: str, content: str, entity_type: WorldEntityType = WorldEntityType.other) -> WorldDocInfo:
        # TODO: Create Google Doc in the target folder and return created metadata.
        raise NotImplementedError("Implement DriveStore.create_doc() using Google Docs + Drive API")

    def append_doc(self, doc_ref: DocumentRef, content: str) -> None:
        # TODO: Append text to the end of Google Doc.
        raise NotImplementedError("Implement DriveStore.append_doc() using batchUpdate")

    def replace_doc(self, doc_ref: DocumentRef, content: str) -> None:
        # TODO: Replace full document content.
        raise NotImplementedError("Implement DriveStore.replace_doc()")

    def replace_section(self, doc_ref: DocumentRef, section: str, content: str) -> None:
        # TODO: Optional - replace a named markdown-like section in the doc body.
        raise NotImplementedError("Implement DriveStore.replace_section()")

    def find_doc(self, *, folder: Optional[str] = None, title: Optional[str] = None, doc_id: Optional[str] = None) -> Optional[WorldDocInfo]:
        docs = self.list_world_docs()
        for doc in docs:
            if doc_id and doc.doc_id == doc_id:
                return doc
            if folder and title and doc.folder == folder and doc.title == title:
                return doc
        return None
