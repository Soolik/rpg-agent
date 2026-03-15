from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .drive_store import DriveStore


class DriveWriteAccessError(RuntimeError):
    pass


@dataclass
class RoutedDriveStore:
    read_store: DriveStore
    write_store_factory: Optional[Callable[[], Optional[DriveStore]]] = None
    require_write_store: bool = False

    def _write_store(self) -> DriveStore:
        if self.write_store_factory is None:
            if self.require_write_store:
                raise DriveWriteAccessError("Google Drive user write access is required for this operation.")
            return self.read_store
        candidate = self.write_store_factory()
        if candidate is not None:
            return candidate
        if self.require_write_store:
            raise DriveWriteAccessError("Google Drive user write access is required for this operation.")
        return self.read_store

    def list_world_docs(self):
        return self.read_store.list_world_docs()

    def list_drive_folder_files(self, folder_id: str):
        return self.read_store.list_drive_folder_files(folder_id)

    def read_drive_file_text(self, file_id: str, mime_type: str):
        return self.read_store.read_drive_file_text(file_id, mime_type)

    def read_doc(self, doc_ref):
        return self.read_store.read_doc(doc_ref)

    def find_doc(self, **kwargs):
        return self.read_store.find_doc(**kwargs)

    def create_doc(self, *args, **kwargs):
        return self._write_store().create_doc(*args, **kwargs)

    def append_doc(self, *args, **kwargs):
        return self._write_store().append_doc(*args, **kwargs)

    def replace_doc(self, *args, **kwargs):
        return self._write_store().replace_doc(*args, **kwargs)

    def replace_section(self, *args, **kwargs):
        return self._write_store().replace_section(*args, **kwargs)
