from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main
from app.canonical_import_service import CanonicalImportService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import lokalnych plikow kanonu do Google Docs.")
    parser.add_argument("--path", required=True, help="Sciezka do folderu z plikami txt/md/docx.")
    parser.add_argument("--apply", action="store_true", help="Wykonaj import. Bez tego dziala dry-run.")
    parser.add_argument("--no-replace", action="store_true", help="Nie nadpisuj istniejacych dokumentow.")
    parser.add_argument("--no-reindex", action="store_true", help="Nie odpalaj reindeksu po imporcie.")
    return parser.parse_args()


def main_cli() -> int:
    args = parse_args()
    drive_store = main.build_drive_store()
    service = CanonicalImportService(
        drive_store=drive_store,
        reindex_fn=main.reindex_after_apply_default,
    )
    result = service.import_folder(
        source_path=args.path,
        dry_run=not args.apply,
        replace_existing=not args.no_replace,
        reindex_after_import=not args.no_reindex,
    )
    print(
        json.dumps(
            {
                "source_path": result.source_path,
                "dry_run": result.dry_run,
                "imported_count": result.imported_count,
                "created_count": result.created_count,
                "updated_count": result.updated_count,
                "skipped_count": result.skipped_count,
                "warnings": result.warnings or [],
                "reindex_result": result.reindex_result,
                "results": [item.__dict__ for item in result.results],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
