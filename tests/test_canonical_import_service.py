import tempfile
import unittest
import zipfile
from pathlib import Path

from app.canonical_import_service import (
    CanonicalImportService,
    classify_canonical_file,
    read_local_canonical_text,
)
from app.models_v2 import DocumentRef


class FakeDriveStore:
    def __init__(self):
        self.docs = {}
        self.created = []
        self.replaced = []

    def find_doc(self, *, folder=None, title=None, doc_id=None):
        if folder and title:
            return self.docs.get((folder, title))
        return None

    def create_doc(self, *, folder, title, content, entity_type):
        doc = DocumentRef(folder=folder, title=title, doc_id=f"doc-{len(self.created) + 1}", path_hint=f"{folder}/{title}")
        self.docs[(folder, title)] = doc
        self.created.append({"folder": folder, "title": title, "content": content, "entity_type": entity_type})
        return doc

    def replace_doc(self, doc_ref, content):
        self.replaced.append({"doc_ref": doc_ref, "content": content})


def build_minimal_docx(path: Path, body_xml: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>""",
        )
        archive.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>""",
        )
        archive.writestr("word/document.xml", body_xml)


class CanonicalImportServiceTest(unittest.TestCase):
    def test_classify_known_files_into_expected_folders(self):
        folder, title, entity_type = classify_canonical_file(Path("Campaign Bible_2026_03_08_2046.docx"))
        self.assertEqual(folder, "01 Bible")
        self.assertEqual(title, "Campaign Bible")
        self.assertEqual(entity_type.value, "bible")

        folder, title, entity_type = classify_canonical_file(Path("NPCs_2026_03_08_2046.docx"))
        self.assertEqual(folder, "03 NPC")
        self.assertEqual(title, "NPCs")
        self.assertEqual(entity_type.value, "npc")

        folder, title, entity_type = classify_canonical_file(Path("Krew Na Gwiazdach - Rozdzial 1 - Cienie w Port Peril_2026_03_08_2046.docx"))
        self.assertEqual(folder, "02 Sessions")
        self.assertIn("Rozdzial 1", title)
        self.assertEqual(entity_type.value, "session")

    def test_read_local_docx_extracts_paragraphs_and_table_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            docx_path = Path(tmp) / "Thread Tracker_2026_03_08_2046.docx"
            build_minimal_docx(
                docx_path,
                """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Thread Tracker</w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Thread ID</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Title</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>T01</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Red Blade</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
    <w:sectPr/>
  </w:body>
</w:document>""",
            )

            text = read_local_canonical_text(docx_path)

        self.assertIn("Thread Tracker", text)
        self.assertIn("Thread ID | Title", text)
        self.assertIn("T01 | Red Blade", text)

    def test_import_folder_creates_and_replaces_docs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Campaign Bible_2026_03_08_2046.txt").write_text("Bible text", encoding="utf-8")
            (root / "NPCs_2026_03_08_2046.txt").write_text("Captain Mira", encoding="utf-8")

            drive_store = FakeDriveStore()
            drive_store.docs[("01 Bible", "Campaign Bible")] = DocumentRef(
                folder="01 Bible",
                title="Campaign Bible",
                doc_id="doc-existing",
                path_hint="01 Bible/Campaign Bible",
            )
            service = CanonicalImportService(drive_store=drive_store, reindex_fn=lambda targets: {"ok": True, "count": len(targets)})

            result = service.import_folder(
                source_path=str(root),
                dry_run=False,
                replace_existing=True,
                reindex_after_import=True,
            )

        self.assertEqual(result.updated_count, 1)
        self.assertEqual(result.created_count, 1)
        self.assertEqual(result.imported_count, 2)
        self.assertEqual(len(drive_store.replaced), 1)
        self.assertEqual(len(drive_store.created), 1)
        self.assertEqual(result.reindex_result["count"], 2)


if __name__ == "__main__":
    unittest.main()
