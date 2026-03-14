from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Type

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from .api_models import (
    AssistantMode,
    ChatArtifact,
    ContinuityReport,
    NextAction,
    RequestTrace,
    SavedOutputRef,
    V1ChatResponse,
)
from .canon_guard import build_continuity_report
from .chat_models import ChatRequest, ChatResponse
from .conversation_store import ConversationMessageRecord, ConversationRecord, ConversationStore, NullConversationStore
from .routes_v2 import build_context_for_planner
from .world_model_store import NullWorldModelStore, WorldModelStore


MAX_HISTORY_MESSAGES = 8
MAX_HISTORY_CHARS = 4000
STREAM_CHUNK_CHARS = 500
SUMMARY_TRIGGER_MESSAGES = 10
SUMMARY_KEEP_RECENT_MESSAGES = 6
SUMMARY_MAX_LINES = 8
SUMMARY_MAX_CHARS = 1200


def _api_error(status_code: int, *, request_trace: RequestTrace, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
            "request_id": request_trace.request_id,
            "trace_id": request_trace.trace_id,
        },
    )


@dataclass
class DirectChatStream:
    chunks: Iterable[str]
    kind: str = "answer"
    artifact_type: Optional[str] = None
    references: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _saved_output_from_chat(response: ChatResponse) -> Optional[SavedOutputRef]:
    if not (response.output_doc_id and response.output_title and response.output_path):
        return None
    return SavedOutputRef(
        doc_id=response.output_doc_id,
        title=response.output_title,
        path=response.output_path,
    )


def _artifact_from_chat(response: ChatResponse) -> Optional[ChatArtifact]:
    if not (response.artifact_type and response.artifact_text):
        return None
    return ChatArtifact(
        artifact_type=response.artifact_type,
        text=response.artifact_text,
        format="markdown",
    )


def _continuity_for_response(
    *,
    message: str,
    response: ChatResponse,
    world_model_store: WorldModelStore | NullWorldModelStore,
) -> Optional[ContinuityReport]:
    text = response.artifact_text or response.reply
    if not text:
        return None
    entities = world_model_store.list_entities(limit=200)
    threads = world_model_store.list_threads(limit=200)
    allow_new_names = response.artifact_type == "npc_brief"
    return build_continuity_report(
        message=message,
        generated_text=text,
        known_entity_names=[entity.name for entity in entities],
        known_thread_names=[thread.title for thread in threads],
        extra_allowed_names=response.references,
        allow_proposed_new_names=allow_new_names,
    )


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    return ""


def _compact_title(value: str, *, fallback: str) -> str:
    text = " ".join((value or "").strip().split())
    if not text:
        return fallback
    text = text.lstrip("#*- ").strip()
    if ":" in text and len(text.split(":", 1)[0]) <= 24:
        text = text.split(":", 1)[1].strip() or text
    return text[:96].rstrip(" .:-") or fallback


def derive_conversation_title(message: str, explicit_title: Optional[str]) -> str:
    if explicit_title and explicit_title.strip():
        return explicit_title.strip()[:96]
    return _compact_title(_first_nonempty_line(message), fallback="Nowa rozmowa")


def _derive_response_title(response: ChatResponse) -> str:
    if response.artifact_text:
        return _compact_title(_first_nonempty_line(response.artifact_text), fallback="Odpowiedz")
    if response.artifact_type:
        return response.artifact_type.replace("_", " ").title()
    return _compact_title(_first_nonempty_line(response.reply), fallback="Odpowiedz")


def _next_actions_for_response(
    *,
    response: ChatResponse,
    continuity: Optional[ContinuityReport],
    conversation_id: Optional[str],
) -> list[NextAction]:
    actions: list[NextAction] = [
        NextAction(
            type="continue_conversation",
            label="Kontynuuj rozmowe",
            payload={"conversation_id": conversation_id} if conversation_id else {},
        )
    ]

    if response.kind == "proposal" and response.proposal_id is not None:
        actions.append(
            NextAction(
                type="accept_world_change",
                label="Zaakceptuj zmiane",
                payload={"proposal_id": response.proposal_id},
            )
        )
        actions.append(
            NextAction(
                type="reject_world_change",
                label="Odrzuc zmiane",
                payload={"proposal_id": response.proposal_id},
            )
        )

    if response.artifact_type:
        actions.append(
            NextAction(
                type="revise",
                label="Przerob odpowiedz",
                payload={"artifact_type": response.artifact_type},
            )
        )

    if continuity and not continuity.ok:
        actions.append(
            NextAction(
                type="review_continuity",
                label="Sprawdz ciaglosc",
                payload={"issue_count": len(continuity.issues)},
            )
        )

    if response.output_doc_id and response.output_path:
        actions.append(
            NextAction(
                type="open_output_doc",
                label="Otworz zapisany output",
                payload={"doc_id": response.output_doc_id, "path": response.output_path},
            )
        )

    deduped: list[NextAction] = []
    seen: set[tuple[str, str]] = set()
    for action in actions:
        key = (action.type, json.dumps(action.payload, ensure_ascii=False, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped


def _guard_context_instruction(message: str, candidate_text: str) -> str:
    return (
        f"{message.strip()}\n\n"
        "KANDYDAT DO WALIDACJI:\n"
        f"{candidate_text.strip()}"
    ).strip()


def _render_guard_reply(
    *,
    candidate_text: str,
    continuity: ContinuityReport,
    planner_notes: Optional[str],
) -> str:
    lines = ["# Guard Report", ""]

    if continuity.ok:
        lines.extend(
            [
                "## Werdykt",
                "",
                "- Nie wykryto oczywistych konfliktow z kanonem ani nowych nazw wlasnych wymagajacych decyzji.",
            ]
        )
    else:
        lines.extend(
            [
                "## Werdykt",
                "",
                f"- Wykryto {len(continuity.issues)} sygnalow do sprawdzenia.",
            ]
        )
        for issue in continuity.issues[:8]:
            lines.append(f"- [{issue.severity}] {issue.message}")

    if continuity.source_backed_names:
        lines.extend(["", "## Nazwy Potwierdzone", ""])
        lines.extend(f"- {name}" for name in continuity.source_backed_names[:10])

    if continuity.proposed_new_names:
        lines.extend(["", "## Nowe Nazwy", ""])
        lines.extend(f"- {name}" for name in continuity.proposed_new_names[:10])

    if continuity.possible_conflicts:
        lines.extend(["", "## Mozliwe Konflikty", ""])
        lines.extend(f"- {item}" for item in continuity.possible_conflicts[:10])

    if planner_notes:
        lines.extend(["", "## Notatki Guard", "", planner_notes.strip()])

    lines.extend(["", "## Sprawdzany Tekst", "", candidate_text.strip()])
    return "\n".join(lines).strip()


def _conversation_storage_enabled(store: ConversationStore | NullConversationStore) -> bool:
    return not isinstance(store, NullConversationStore)


def _prompt_history(messages: list[ConversationMessageRecord]) -> list[ConversationMessageRecord]:
    if not messages:
        return []

    selected: list[ConversationMessageRecord] = []
    total_chars = 0
    for message in reversed(messages):
        rendered = f"{message.role}: {message.content}"
        if selected and total_chars + len(rendered) > MAX_HISTORY_CHARS:
            break
        selected.append(message)
        total_chars += len(rendered)
        if len(selected) >= MAX_HISTORY_MESSAGES:
            break
    selected.reverse()
    return selected


def _compact_summary_text(text: str, *, limit: int = 160) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _summarize_messages(messages: list[ConversationMessageRecord]) -> str:
    if not messages:
        return ""

    selected = messages
    if len(messages) > SUMMARY_MAX_LINES:
        head = messages[:2]
        tail = messages[-(SUMMARY_MAX_LINES - 3) :]
        selected = head + [ConversationMessageRecord(
            message_id=0,
            conversation_id=messages[0].conversation_id,
            campaign_id=messages[0].campaign_id,
            role="system",
            content="Pominieto starsze wiadomosci.",
            created_at=messages[0].created_at,
            metadata={},
        )] + tail

    lines = ["PODSUMOWANIE ROZMOWY:"]
    for item in selected:
        if item.role == "system":
            lines.append("- " + item.content)
            continue
        role_label = "U" if item.role == "user" else "A" if item.role == "assistant" else item.role[:1].upper()
        kind_suffix = f" ({item.kind})" if item.kind else ""
        lines.append(f"- {role_label}{kind_suffix}: {_compact_summary_text(item.content)}")

    summary = "\n".join(lines).strip()
    if len(summary) <= SUMMARY_MAX_CHARS:
        return summary
    return summary[: SUMMARY_MAX_CHARS - 3].rstrip() + "..."


def _compose_message_with_history(
    message: str,
    history: list[ConversationMessageRecord],
    *,
    summary_text: Optional[str] = None,
) -> str:
    if not history and not summary_text:
        return message

    if not history and len(message) > MAX_HISTORY_CHARS:
        return message

    lines = []
    if summary_text:
        lines.extend([summary_text.strip(), ""])

    lines.append("KONTEKST ROZMOWY:")
    for item in history:
        role_label = "Uzytkownik" if item.role == "user" else "Asystent" if item.role == "assistant" else item.role.title()
        lines.append(f"{role_label}: {item.content}")
    lines.extend(
        [
            "",
            "NOWA WIADOMOSC UZYTKOWNIKA:",
            message,
            "",
            "Odpowiedz na ostatnia wiadomosc, zachowujac ciaglosc rozmowy i kanonu.",
        ]
    )
    return "\n".join(lines)


def _stream_chunks(text: str) -> list[str]:
    compact = text or ""
    if not compact:
        return []
    chunks: list[str] = []
    cursor = 0
    while cursor < len(compact):
        chunks.append(compact[cursor : cursor + STREAM_CHUNK_CHARS])
        cursor += STREAM_CHUNK_CHARS
    return chunks


def _sse_event(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


class ChatService:
    def __init__(
        self,
        *,
        chat_request_cls: Type[ChatRequest],
        chat_fn: Callable[[ChatRequest], ChatResponse],
        chat_stream_fn: Optional[Callable[[ChatRequest], Optional[DirectChatStream]]] = None,
        drive_store,
        planner,
        consistency_planner=None,
        world_model_store: Optional[WorldModelStore | NullWorldModelStore] = None,
        conversation_store: Optional[ConversationStore | NullConversationStore] = None,
    ):
        self.chat_request_cls = chat_request_cls
        self.chat_fn = chat_fn
        self.chat_stream_fn = chat_stream_fn
        self.drive_store = drive_store
        self.planner = planner
        self.consistency_planner = consistency_planner or planner
        self.world_model_store = world_model_store or NullWorldModelStore()
        self.conversation_store = conversation_store or NullConversationStore()

    def conversation_storage_enabled(self) -> bool:
        return _conversation_storage_enabled(self.conversation_store)

    def get_conversation(self, conversation_id: str) -> Optional[ConversationRecord]:
        return self.conversation_store.get_conversation(conversation_id)

    def create_conversation(self, *, title: Optional[str], seed_message: str = "", metadata: Optional[dict] = None) -> Optional[ConversationRecord]:
        if not self.conversation_storage_enabled():
            return None
        return self.conversation_store.create_conversation(
            title=derive_conversation_title(seed_message, title),
            metadata=metadata or {"source": "v1_chat"},
        )

    def list_conversations(self, limit: int = 20) -> list[ConversationRecord]:
        return self.conversation_store.list_conversations(limit=limit)

    def list_messages(self, conversation_id: str, limit: int = 100) -> list[ConversationMessageRecord]:
        return self.conversation_store.list_messages(conversation_id, limit=limit)

    def _conversation_summary_state(self, conversation: ConversationRecord) -> tuple[str, int]:
        metadata = conversation.metadata or {}
        summary_text = str(metadata.get("summary_text") or "").strip()
        try:
            summary_message_count = int(metadata.get("summary_message_count") or 0)
        except (TypeError, ValueError):
            summary_message_count = 0
        return summary_text, max(0, summary_message_count)

    def _prepare_conversation_context(
        self,
        *,
        trace: RequestTrace,
        message: str,
        conversation_id: Optional[str],
        conversation_title: Optional[str],
    ) -> tuple[Optional[ConversationRecord], str]:
        conversation = self._resolve_conversation(
            trace=trace,
            conversation_id=conversation_id,
            conversation_title=conversation_title,
            seed_message=message,
        )
        if not conversation:
            return None, message

        existing_messages = self.conversation_store.list_messages(conversation.conversation_id, limit=200)
        summary_text, summary_message_count = self._conversation_summary_state(conversation)
        if summary_message_count > 0 and summary_message_count < len(existing_messages):
            recent_messages = existing_messages[summary_message_count:]
        elif summary_message_count >= len(existing_messages):
            recent_messages = []
        else:
            recent_messages = existing_messages

        history = _prompt_history(recent_messages)
        composed_message = _compose_message_with_history(
            message,
            history,
            summary_text=summary_text or None,
        )
        return conversation, composed_message

    def _append_user_message(
        self,
        conversation: Optional[ConversationRecord],
        *,
        message: str,
        artifact_type: Optional[str],
        source_title: Optional[str],
        assistant_mode: AssistantMode,
        candidate_text: Optional[str],
    ) -> None:
        if not conversation:
            return
        self.conversation_store.append_message(
            conversation.conversation_id,
            role="user",
            content=message,
            kind="input",
            artifact_type=artifact_type,
            metadata={
                "source_title": source_title,
                "assistant_mode": assistant_mode.value,
                "candidate_text": candidate_text,
            },
        )

    def _append_assistant_message(
        self,
        conversation: Optional[ConversationRecord],
        *,
        response: V1ChatResponse,
    ) -> None:
        if not conversation:
            return
        self.conversation_store.append_message(
            conversation.conversation_id,
            role="assistant",
            content=response.reply_markdown,
            kind=response.kind,
            artifact_type=response.artifact_type,
            metadata={
                "proposal_id": response.proposal_id,
                "session_id": response.session_id,
                "continuity_ok": response.continuity.ok if response.continuity else None,
                "assistant_mode": response.mode.value,
            },
        )

    def _refresh_conversation_summary(self, conversation: Optional[ConversationRecord]) -> None:
        if not conversation or not self.conversation_storage_enabled():
            return
        messages = self.conversation_store.list_messages(conversation.conversation_id, limit=200)
        if len(messages) <= SUMMARY_TRIGGER_MESSAGES:
            if (conversation.metadata or {}).get("summary_text"):
                self.conversation_store.update_conversation_metadata(
                    conversation.conversation_id,
                    metadata_patch={"summary_text": "", "summary_message_count": 0},
                )
            return

        summary_cutoff = max(0, len(messages) - SUMMARY_KEEP_RECENT_MESSAGES)
        summary_text = _summarize_messages(messages[:summary_cutoff])
        self.conversation_store.update_conversation_metadata(
            conversation.conversation_id,
            metadata_patch={
                "summary_text": summary_text,
                "summary_message_count": summary_cutoff,
            },
        )

    def _resolve_conversation(
        self,
        *,
        trace: RequestTrace,
        conversation_id: Optional[str],
        conversation_title: Optional[str],
        seed_message: str,
    ) -> Optional[ConversationRecord]:
        if not conversation_id:
            if not self.conversation_storage_enabled():
                return None
            return self.create_conversation(title=conversation_title, seed_message=seed_message, metadata={"source": "v1_chat"})

        if not self.conversation_storage_enabled():
            raise _api_error(
                503,
                request_trace=trace,
                code="conversation_store_unavailable",
                message="Conversation storage is not configured for this deployment.",
            )

        conversation = self.conversation_store.get_conversation(conversation_id)
        if not conversation:
            raise _api_error(
                404,
                request_trace=trace,
                code="conversation_not_found",
                message="Conversation not found.",
            )
        return conversation

    def _response_from_chat(
        self,
        *,
        trace: RequestTrace,
        message: str,
        response: ChatResponse,
        mode: AssistantMode,
        conversation_id: Optional[str] = None,
        conversation_title: Optional[str] = None,
    ) -> V1ChatResponse:
        continuity = _continuity_for_response(
            message=message,
            response=response,
            world_model_store=self.world_model_store,
        )
        reply_markdown = response.artifact_text or response.reply
        return V1ChatResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            kind=response.kind,
            mode=mode,
            reply=response.reply,
            reply_markdown=reply_markdown,
            title=_derive_response_title(response),
            conversation_id=conversation_id,
            conversation_title=conversation_title,
            artifact_type=response.artifact_type,
            artifact_text=response.artifact_text,
            artifact=_artifact_from_chat(response),
            proposal_id=response.proposal_id,
            session_id=response.session_id,
            citations=response.references,
            warnings=response.warnings,
            next_actions=_next_actions_for_response(
                response=response,
                continuity=continuity,
                conversation_id=conversation_id,
            ),
            output=_saved_output_from_chat(response),
            telemetry=response.telemetry,
            continuity=continuity,
        )

    def _build_guard_response(
        self,
        *,
        trace: RequestTrace,
        message: str,
        candidate_text: str,
        planner_notes: Optional[str],
        conversation_id: Optional[str] = None,
        conversation_title: Optional[str] = None,
    ) -> V1ChatResponse:
        entities = self.world_model_store.list_entities(limit=200)
        threads = self.world_model_store.list_threads(limit=200)
        continuity = build_continuity_report(
            message=message,
            generated_text=candidate_text,
            known_entity_names=[entity.name for entity in entities],
            known_thread_names=[thread.title for thread in threads],
            extra_allowed_names=[],
            allow_proposed_new_names=False,
        )
        reply_markdown = _render_guard_reply(
            candidate_text=candidate_text,
            continuity=continuity,
            planner_notes=planner_notes,
        )
        pseudo_response = ChatResponse(
            kind="answer",
            reply=reply_markdown,
            references=[],
            warnings=[],
        )
        return V1ChatResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            kind="answer",
            mode=AssistantMode.guard,
            reply=reply_markdown,
            reply_markdown=reply_markdown,
            title="Guard Report",
            conversation_id=conversation_id,
            conversation_title=conversation_title,
            citations=[],
            warnings=[],
            next_actions=_next_actions_for_response(
                response=pseudo_response,
                continuity=continuity,
                conversation_id=conversation_id,
            ),
            output=None,
            telemetry=None,
            continuity=continuity,
        )

    def run(
        self,
        *,
        trace: RequestTrace,
        message: str,
        assistant_mode: AssistantMode,
        intent: str,
        artifact_type: Optional[str],
        source_title: Optional[str],
        candidate_text: Optional[str],
        include_sources: bool,
        include_telemetry: bool,
        save_output: bool,
        output_title: Optional[str],
        conversation_id: Optional[str],
        conversation_title: Optional[str],
    ) -> V1ChatResponse:
        conversation, composed_message = self._prepare_conversation_context(
            trace=trace,
            message=message,
            conversation_id=conversation_id,
            conversation_title=conversation_title,
        )
        self._append_user_message(
            conversation,
            message=message,
            artifact_type=artifact_type,
            source_title=source_title,
            assistant_mode=assistant_mode,
            candidate_text=candidate_text,
        )

        if assistant_mode == AssistantMode.guard:
            checked_text = (candidate_text or message).strip()
            planner_notes = None
            if self.consistency_planner and hasattr(self.consistency_planner, "consistency_check"):
                planner_notes = self.consistency_planner.consistency_check(
                    instruction=_guard_context_instruction(composed_message, checked_text),
                    world_context=build_context_for_planner(self.drive_store),
                )
            rendered = self._build_guard_response(
                trace=trace,
                message=message,
                candidate_text=checked_text,
                planner_notes=planner_notes,
                conversation_id=conversation.conversation_id if conversation else None,
                conversation_title=conversation.title if conversation else None,
            )
        else:
            effective_intent = "proposal" if assistant_mode == AssistantMode.editor else intent
            response = self.chat_fn(
                self.chat_request_cls(
                    message=composed_message,
                    intent=effective_intent,
                    artifact_type=artifact_type,
                    source_title=source_title,
                    conversation_id=conversation.conversation_id if conversation else None,
                    conversation_title=conversation.title if conversation else conversation_title,
                    include_sources=include_sources,
                    include_telemetry=include_telemetry,
                    save_output=save_output,
                    output_title=output_title,
                )
            )
            rendered = self._response_from_chat(
                trace=trace,
                message=message,
                response=response,
                mode=assistant_mode,
                conversation_id=conversation.conversation_id if conversation else None,
                conversation_title=conversation.title if conversation else None,
            )

        self._append_assistant_message(conversation, response=rendered)
        self._refresh_conversation_summary(conversation)
        return rendered

    def stream_run(
        self,
        *,
        trace: RequestTrace,
        message: str,
        assistant_mode: AssistantMode,
        intent: str,
        artifact_type: Optional[str],
        source_title: Optional[str],
        candidate_text: Optional[str],
        include_sources: bool,
        include_telemetry: bool,
        save_output: bool,
        output_title: Optional[str],
        conversation_id: Optional[str],
        conversation_title: Optional[str],
    ) -> StreamingResponse:
        conversation, composed_message = self._prepare_conversation_context(
            trace=trace,
            message=message,
            conversation_id=conversation_id,
            conversation_title=conversation_title,
        )

        handle: Optional[DirectChatStream] = None
        if self.chat_stream_fn and assistant_mode == AssistantMode.create:
            handle = self.chat_stream_fn(
                self.chat_request_cls(
                    message=composed_message,
                    intent=intent,
                    artifact_type=artifact_type,
                    source_title=source_title,
                    conversation_id=conversation.conversation_id if conversation else None,
                    conversation_title=conversation.title if conversation else conversation_title,
                    include_sources=include_sources,
                    include_telemetry=include_telemetry,
                    save_output=save_output,
                    output_title=output_title,
                )
            )

        if not handle:
            response = self.run(
                trace=trace,
                message=message,
                assistant_mode=assistant_mode,
                intent=intent,
                artifact_type=artifact_type,
                source_title=source_title,
                candidate_text=candidate_text,
                include_sources=include_sources,
                include_telemetry=include_telemetry,
                save_output=save_output,
                output_title=output_title,
                conversation_id=conversation_id,
                conversation_title=conversation_title,
            )
            return self.stream(response)

        self._append_user_message(
            conversation,
            message=message,
            artifact_type=artifact_type,
            source_title=source_title,
            assistant_mode=assistant_mode,
            candidate_text=candidate_text,
        )

        def iterator():
            yield _sse_event(
                "start",
                {
                    "request_id": trace.request_id,
                    "trace_id": trace.trace_id,
                    "conversation_id": conversation.conversation_id if conversation else None,
                    "title": "Odpowiedz",
                    "stream_mode": "direct",
                },
            )

            try:
                parts: list[str] = []
                for chunk in handle.chunks:
                    if not chunk:
                        continue
                    parts.append(chunk)
                    yield _sse_event("delta", {"text": chunk})

                final_text = "".join(parts).strip()
                final_response = self._response_from_chat(
                    trace=trace,
                    message=message,
                    response=ChatResponse(
                        kind=handle.kind,  # type: ignore[arg-type]
                        reply=final_text,
                        artifact_type=handle.artifact_type,
                        artifact_text=final_text if handle.artifact_type else None,
                        references=handle.references,
                        warnings=handle.warnings,
                    ),
                    mode=assistant_mode,
                    conversation_id=conversation.conversation_id if conversation else None,
                    conversation_title=conversation.title if conversation else None,
                )
                self._append_assistant_message(conversation, response=final_response)
                self._refresh_conversation_summary(conversation)
                yield _sse_event("complete", final_response.model_dump(mode="json"))
            except Exception as exc:
                yield _sse_event(
                    "error",
                    {
                        "request_id": trace.request_id,
                        "trace_id": trace.trace_id,
                        "message": str(exc),
                    },
                )

        return StreamingResponse(iterator(), media_type="text/event-stream")

    def stream(self, response: V1ChatResponse) -> StreamingResponse:
        payload = response.model_dump(mode="json")

        def iterator():
            yield _sse_event(
                "start",
                {
                    "request_id": response.request_id,
                    "trace_id": response.trace_id,
                    "conversation_id": response.conversation_id,
                    "title": response.title,
                    "stream_mode": "buffered",
                },
            )
            for chunk in _stream_chunks(response.reply_markdown):
                yield _sse_event("delta", {"text": chunk})
            yield _sse_event("complete", payload)

        return StreamingResponse(iterator(), media_type="text/event-stream")
