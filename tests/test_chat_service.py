import unittest

from app.api_models import AssistantMode, RequestTrace
from app.chat_models import ChatRequest, ChatResponse
from app.chat_service import ChatService
from app.conversation_store import ConversationMessageRecord, ConversationRecord


class FakeDriveStore:
    pass


class FakePlanner:
    def consistency_check(self, instruction, world_context):
        return "OK"


class FakeWorldModelStore:
    def list_entities(self, limit=200):
        return []

    def list_threads(self, limit=200):
        return []


class FakeConversationStore:
    def __init__(self):
        self.conversations = {}
        self.messages = {}
        self.next_id = 1
        self.metadata_updates = []

    def create_conversation(self, *, title="", metadata=None):
        conversation_id = f"conv-{self.next_id}"
        self.next_id += 1
        record = ConversationRecord(
            conversation_id=conversation_id,
            campaign_id="kng",
            title=title or "Nowa rozmowa",
            message_count=0,
            created_at="2026-03-15T00:00:00+00:00",
            updated_at="2026-03-15T00:00:00+00:00",
            last_message_at=None,
            metadata=metadata or {},
        )
        self.conversations[conversation_id] = record
        self.messages[conversation_id] = []
        return record

    def get_conversation(self, conversation_id):
        return self.conversations.get(conversation_id)

    def list_conversations(self, limit=20):
        return list(self.conversations.values())[:limit]

    def append_message(self, conversation_id, *, role, content, kind=None, artifact_type=None, metadata=None):
        items = self.messages.setdefault(conversation_id, [])
        message = ConversationMessageRecord(
            message_id=len(items) + 1,
            conversation_id=conversation_id,
            campaign_id="kng",
            role=role,
            content=content,
            kind=kind,
            artifact_type=artifact_type,
            created_at="2026-03-15T00:00:00+00:00",
            metadata=metadata or {},
        )
        items.append(message)
        conversation = self.conversations[conversation_id]
        conversation.message_count = len(items)
        conversation.last_message_at = message.created_at
        conversation.updated_at = message.created_at
        return message

    def list_messages(self, conversation_id, limit=100):
        return self.messages.get(conversation_id, [])[:limit]

    def update_conversation_metadata(self, conversation_id, *, metadata_patch):
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return None
        self.metadata_updates.append({"conversation_id": conversation_id, "metadata_patch": metadata_patch})
        conversation.metadata = {
            **(conversation.metadata or {}),
            **(metadata_patch or {}),
        }
        return conversation


class ChatServiceTest(unittest.TestCase):
    def build_service(self, chat_fn, conversation_store):
        return ChatService(
            chat_request_cls=ChatRequest,
            chat_fn=chat_fn,
            drive_store=FakeDriveStore(),
            planner=FakePlanner(),
            consistency_planner=FakePlanner(),
            world_model_store=FakeWorldModelStore(),
            conversation_store=conversation_store,
        )

    def test_run_includes_summary_and_recent_history_in_prompt(self):
        seen = {}

        def fake_chat(req):
            seen["message"] = req.message
            return ChatResponse(kind="answer", reply="Kontynuacja.", references=[])

        store = FakeConversationStore()
        conversation = store.create_conversation(
            title="Red Blade",
            metadata={
                "summary_text": "PODSUMOWANIE ROZMOWY:\n- U: Ustalono, ze Red Blade naciska na Captain Mira.",
                "summary_message_count": 2,
            },
        )
        store.append_message(conversation.conversation_id, role="user", content="Kim jest Captain Mira?", kind="input")
        store.append_message(conversation.conversation_id, role="assistant", content="Dowodzi garnizonem.", kind="answer")
        store.append_message(conversation.conversation_id, role="user", content="Co planuje Red Blade?", kind="input")
        store.append_message(conversation.conversation_id, role="assistant", content="Chce wymusic lojalnosc.", kind="answer")

        service = self.build_service(fake_chat, store)
        service.run(
            trace=RequestTrace(request_id="req-1", trace_id="req-1"),
            message="A jak to uderzy w Captain Mira?",
            assistant_mode=AssistantMode.create,
            intent="answer",
            artifact_type=None,
            source_title=None,
            candidate_text=None,
            include_sources=False,
            include_telemetry=False,
            save_output=False,
            output_title=None,
            conversation_id=conversation.conversation_id,
            conversation_title=None,
        )

        self.assertIn("PODSUMOWANIE ROZMOWY:", seen["message"])
        self.assertIn("Co planuje Red Blade?", seen["message"])
        self.assertIn("A jak to uderzy w Captain Mira?", seen["message"])

    def test_run_refreshes_summary_after_long_conversation(self):
        def fake_chat(_req):
            return ChatResponse(kind="answer", reply="Nowa odpowiedz.", references=[])

        store = FakeConversationStore()
        conversation = store.create_conversation(title="Dluga rozmowa", metadata={})
        for idx in range(10):
            role = "user" if idx % 2 == 0 else "assistant"
            kind = "input" if role == "user" else "answer"
            store.append_message(
                conversation.conversation_id,
                role=role,
                content=f"Wiadomosc {idx}",
                kind=kind,
            )

        service = self.build_service(fake_chat, store)
        service.run(
            trace=RequestTrace(request_id="req-2", trace_id="req-2"),
            message="Dopisujemy kolejny krok.",
            assistant_mode=AssistantMode.create,
            intent="answer",
            artifact_type=None,
            source_title=None,
            candidate_text=None,
            include_sources=False,
            include_telemetry=False,
            save_output=False,
            output_title=None,
            conversation_id=conversation.conversation_id,
            conversation_title=None,
        )

        updated = store.get_conversation(conversation.conversation_id)
        self.assertTrue(updated.metadata["summary_text"].startswith("PODSUMOWANIE ROZMOWY:"))
        self.assertGreater(updated.metadata["summary_message_count"], 0)

    def test_run_does_not_refresh_summary_when_only_small_unsummarized_delta_exists(self):
        def fake_chat(_req):
            return ChatResponse(kind="answer", reply="Krotka odpowiedz.", references=[])

        store = FakeConversationStore()
        conversation = store.create_conversation(
            title="Red Blade",
            metadata={
                "summary_text": "PODSUMOWANIE ROZMOWY:\n- U: Red Blade naciska.",
                "summary_message_count": 10,
            },
        )
        for idx in range(10):
            role = "user" if idx % 2 == 0 else "assistant"
            kind = "input" if role == "user" else "answer"
            store.append_message(conversation.conversation_id, role=role, content=f"Stara wiadomosc {idx}", kind=kind)

        service = self.build_service(fake_chat, store)
        service.run(
            trace=RequestTrace(request_id="req-3", trace_id="req-3"),
            message="Krotki dopisek.",
            assistant_mode=AssistantMode.create,
            intent="answer",
            artifact_type=None,
            source_title=None,
            candidate_text=None,
            include_sources=False,
            include_telemetry=False,
            save_output=False,
            output_title=None,
            conversation_id=conversation.conversation_id,
            conversation_title=None,
        )

        summary_updates = [item for item in store.metadata_updates if "summary_text" in item["metadata_patch"]]
        self.assertEqual(summary_updates, [])


if __name__ == "__main__":
    unittest.main()
