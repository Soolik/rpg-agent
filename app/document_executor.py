from __future__ import annotations

from dataclasses import dataclass
from difflib import unified_diff
from typing import Optional

from .drive_store import DriveStore, replace_section_content
from .models_v2 import (
    ActionType,
    AppliedActionResult,
    DocumentAction,
    DocumentExecutionPreview,
    DocumentRef,
    WorldEntityType,
)
from .text_normalization import normalize_text_artifacts


def _normalize_text(value: str) -> str:
    return normalize_text_artifacts(value or "").strip()


def _excerpt(value: str, *, limit: int = 300) -> str:
    compact = (value or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _diff_text(current_text: str, proposed_text: str) -> str:
    diff_lines = list(
        unified_diff(
            current_text.splitlines(),
            proposed_text.splitlines(),
            fromfile="current",
            tofile="proposed",
            lineterm="",
        )
    )
    if len(diff_lines) > 160:
        diff_lines = diff_lines[:160] + ["... diff truncated ..."]
    return "\n".join(diff_lines)


def _merge_append_text(current_text: str, content: str) -> str:
    if not current_text.strip():
        return content.strip()
    addition = content.strip()
    if not addition:
        return current_text.rstrip()
    return current_text.rstrip() + "\n\n" + addition


def _update_tracker_rows(current_text: str, content: str, metadata: dict) -> str:
    incoming = [line.rstrip() for line in (content or "").splitlines() if line.strip()]
    if not incoming:
        return current_text
    current_lines = current_text.splitlines()
    thread_id = str(metadata.get("thread_id") or "").strip()
    title = str(metadata.get("title") or "").strip().lower()
    replacement_line = incoming[0]
    target_index = None
    for index, line in enumerate(current_lines):
        normalized = line.strip().lower()
        if thread_id and normalized.startswith(thread_id.lower()):
            target_index = index
            break
        if title and title in normalized:
            target_index = index
            break
    if target_index is None:
        if current_text.rstrip():
            return current_text.rstrip() + "\n" + replacement_line + "\n"
        return replacement_line + "\n"
    current_lines[target_index] = replacement_line
    return "\n".join(current_lines).rstrip() + "\n"


@dataclass(frozen=True)
class PreparedDocumentAction:
    action: DocumentAction
    target: Optional[DocumentRef]
    current_text: str
    proposed_text: str
    preview: DocumentExecutionPreview


@dataclass
class DocumentExecutor:
    drive_store: DriveStore

    def _resolve_target(self, action: DocumentAction) -> Optional[DocumentRef]:
        if not action.target:
            return None
        if action.target.doc_id:
            return action.target
        found = self.drive_store.find_doc(folder=action.target.folder, title=action.target.title)
        if not found:
            return action.target
        return DocumentRef(
            folder=found.folder,
            title=found.title,
            doc_id=found.doc_id,
            path_hint=found.path_hint,
        )

    def preview_action(self, action: DocumentAction) -> PreparedDocumentAction:
        target = self._resolve_target(action)
        current_text = ""
        if action.action_type not in {ActionType.create_doc, ActionType.reindex} and target and target.doc_id:
            current_text = self.drive_store.read_doc(target)

        if action.action_type == ActionType.create_doc:
            proposed_text = (action.content or "").strip()
            summary = "Preview create document"
        elif action.action_type == ActionType.append_doc:
            proposed_text = _merge_append_text(current_text, action.content or "")
            summary = "Preview append document"
        elif action.action_type == ActionType.replace_doc:
            proposed_text = action.content or ""
            summary = "Preview replace document"
        elif action.action_type == ActionType.replace_section:
            proposed_text = replace_section_content(current_text, action.section or "", action.content or "")
            summary = f"Preview replace section {action.section or ''}".strip()
        elif action.action_type == ActionType.create_if_missing:
            proposed_text = current_text if current_text.strip() else (action.content or "").strip()
            summary = "Preview create if missing"
        elif action.action_type == ActionType.update_tracker_row:
            proposed_text = _update_tracker_rows(current_text, action.content or "", action.metadata)
            summary = "Preview update tracker row"
        else:
            proposed_text = current_text
            summary = "Preview reindex"

        preview = DocumentExecutionPreview(
            action_type=action.action_type,
            target=target,
            summary=summary,
            current_excerpt=_excerpt(current_text),
            proposed_excerpt=_excerpt(proposed_text),
            diff_text=_diff_text(current_text, proposed_text) if action.action_type != ActionType.reindex else "",
        )
        return PreparedDocumentAction(
            action=action,
            target=target,
            current_text=current_text,
            proposed_text=proposed_text,
            preview=preview,
        )

    def verify_action(self, prepared: PreparedDocumentAction, target: Optional[DocumentRef]) -> tuple[bool, str]:
        if prepared.action.action_type == ActionType.reindex:
            return True, "Reindex delegated"
        if not target or not target.doc_id:
            return prepared.action.action_type == ActionType.create_if_missing, "No target document to verify"
        read_back = self.drive_store.read_doc(target)
        normalized_read = _normalize_text(read_back)
        normalized_expected = _normalize_text(prepared.proposed_text)

        if prepared.action.action_type in {ActionType.replace_doc, ActionType.create_doc, ActionType.update_tracker_row}:
            if normalized_read == normalized_expected:
                return True, "Document content verified"
            if normalized_expected and normalized_expected in normalized_read:
                return True, "Document contains expected content"
            return False, "Document verification mismatch"

        if prepared.action.action_type == ActionType.append_doc:
            normalized_addition = _normalize_text(prepared.action.content or "")
            if normalized_addition and normalized_addition in normalized_read:
                return True, "Appended content verified"
            return False, "Appended content not found after write"

        if prepared.action.action_type == ActionType.replace_section:
            expected = _normalize_text(prepared.action.content or "")
            if expected and expected in normalized_read:
                return True, "Section content verified"
            return False, "Section content not found after write"

        if prepared.action.action_type == ActionType.create_if_missing:
            if normalized_expected and normalized_expected in normalized_read:
                return True, "Create-if-missing content verified"
            return True, "Document already existed"

        return True, "Verification skipped"

    def apply_prepared(self, prepared: PreparedDocumentAction) -> AppliedActionResult:
        action = prepared.action
        target = prepared.target
        effective_target = target

        if action.action_type == ActionType.create_doc:
            if not action.target:
                raise ValueError("create_doc requires target")
            created = self.drive_store.create_doc(
                folder=action.target.folder,
                title=action.target.title,
                content=action.content or "",
                entity_type=action.entity_type or WorldEntityType.other,
            )
            effective_target = DocumentRef(
                folder=created.folder,
                title=created.title,
                doc_id=created.doc_id,
                path_hint=created.path_hint,
            )
        elif action.action_type == ActionType.append_doc:
            if not target:
                raise ValueError("append_doc requires target")
            self.drive_store.append_doc(target, action.content or "")
        elif action.action_type == ActionType.replace_doc:
            if not target:
                raise ValueError("replace_doc requires target")
            self.drive_store.replace_doc(target, action.content or "")
        elif action.action_type == ActionType.replace_section:
            if not target:
                raise ValueError("replace_section requires target")
            if not action.section:
                raise ValueError("replace_section requires section")
            self.drive_store.replace_section(target, action.section, action.content or "")
        elif action.action_type == ActionType.create_if_missing:
            if not action.target:
                raise ValueError("create_if_missing requires target")
            if target and target.doc_id:
                effective_target = target
            else:
                created = self.drive_store.create_doc(
                    folder=action.target.folder,
                    title=action.target.title,
                    content=action.content or "",
                    entity_type=action.entity_type or WorldEntityType.other,
                )
                effective_target = DocumentRef(
                    folder=created.folder,
                    title=created.title,
                    doc_id=created.doc_id,
                    path_hint=created.path_hint,
                )
        elif action.action_type == ActionType.update_tracker_row:
            if not target:
                raise ValueError("update_tracker_row requires target")
            self.drive_store.replace_doc(target, prepared.proposed_text)
        elif action.action_type == ActionType.reindex:
            return AppliedActionResult(
                action_type=action.action_type,
                success=True,
                message="Reindex delegated to apply step.",
                target=target,
                details={"preview": prepared.preview.model_dump(mode="json"), "verified": True},
            )
        else:
            raise ValueError(f"Unsupported action type: {action.action_type}")

        verified, verify_message = self.verify_action(prepared, effective_target)
        return AppliedActionResult(
            action_type=action.action_type,
            success=verified,
            message=verify_message,
            target=effective_target,
            details={"preview": prepared.preview.model_dump(mode="json"), "verified": verified},
        )
