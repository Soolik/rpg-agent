import unittest

import main
from app.models_v2 import ChangeProposal, DocumentRef, WorldEntityRecord, WorldThreadRecord


class ChatFlowTest(unittest.TestCase):
    def test_detect_chat_intent_prefers_proposal_markers(self):
        intent = main.detect_chat_intent(
            "W dokumencie Campaign Bible podmien sekcje Test Automation na nowy tekst."
        )
        self.assertEqual(intent, "proposal")

    def test_detect_chat_intent_recognizes_creative_request(self):
        intent = main.detect_chat_intent(
            "Wymysl 3 hooki na nastepna sesje zwiazane z Red Blade."
        )
        self.assertEqual(intent, "creative")

    def test_infer_artifact_type_recognizes_pre_session_brief(self):
        artifact_type = main.infer_artifact_type(
            "Przygotuj briefing przed sesja o Red Blade i Captain Mira.",
            None,
        )
        self.assertEqual(artifact_type, "pre_session_brief")

    def test_infer_artifact_type_recognizes_pre_session_brief_with_polish_inflection(self):
        artifact_type = main.infer_artifact_type(
            "Potrzebny brief przed sesją o Red Blade i Captain Mira.",
            None,
        )
        self.assertEqual(artifact_type, "pre_session_brief")

    def test_chat_answer_returns_human_text_with_sources(self):
        original_ask = main.ask
        try:
            main.ask = lambda req: main.AskResponse(
                answer="Ten wpis potwierdza, ze apply_changes dziala poprawnie.",
                sources=[
                    {
                        "folder": "01 Bible",
                        "title": "Campaign Bible",
                        "doc_id": "doc-1",
                    }
                ],
            )

            response = main.chat(
                main.ChatRequest(
                    message="Co mowi sekcja Test Automation?",
                    intent="answer",
                    include_sources=True,
                )
            )
        finally:
            main.ask = original_ask

        self.assertEqual(response.kind, "answer")
        self.assertIn("Ten wpis potwierdza", response.reply)
        self.assertIn("Zrodla:", response.reply)
        self.assertEqual(response.references, ["01 Bible / Campaign Bible"])

    def test_chat_can_return_telemetry_when_requested(self):
        original_ask = main.ask

        def fake_ask(req):
            main.record_telemetry("gemini_calls", {"label": "answer:test", "finish_reason": "STOP"})
            return main.AskResponse(answer="Gotowy tekst.", sources=[])

        try:
            main.ask = fake_ask
            response = main.chat(
                main.ChatRequest(
                    message="Odpowiedz krotko.",
                    intent="answer",
                    include_telemetry=True,
                )
            )
        finally:
            main.ask = original_ask

        self.assertIsNotNone(response.telemetry)
        self.assertEqual(response.telemetry["gemini_calls"][0]["label"], "answer:test")

    def test_chat_answer_can_render_gm_brief_artifact(self):
        original_ask = main.ask
        try:
            main.ask = lambda req: main.AskResponse(
                answer="Ten wpis potwierdza, ze apply_changes dziala poprawnie.",
                sources=[
                    {
                        "folder": "01 Bible",
                        "title": "Campaign Bible",
                        "doc_id": "doc-1",
                    }
                ],
            )

            response = main.chat(
                main.ChatRequest(
                    message="Co mowi sekcja Test Automation?",
                    intent="answer",
                    include_sources=True,
                    artifact_type="gm_brief",
                )
            )
        finally:
            main.ask = original_ask

        self.assertEqual(response.artifact_type, "gm_brief")
        self.assertIn("# GM Brief", response.artifact_text)
        self.assertIn("## Sources", response.artifact_text)
        self.assertIn("01 Bible / Campaign Bible", response.artifact_text)

    def test_chat_text_appends_telemetry_when_present(self):
        original_chat = main.chat
        try:
            main.chat = lambda req: main.ChatResponse(
                kind="answer",
                reply="Gotowy tekst.",
                telemetry={"gemini_calls": [{"label": "debug:test"}]},
            )
            text = main.chat_text(
                main.ChatRequest(
                    message="Odpowiedz krotko.",
                    include_telemetry=True,
                )
            )
        finally:
            main.chat = original_chat

        self.assertIn("Telemetry:", text)
        self.assertIn("\"debug:test\"", text)

    def test_chat_creative_returns_generated_artifact(self):
        original_generate_creative_artifact = main.generate_creative_artifact
        try:
            main.generate_creative_artifact = lambda **kwargs: (
                "Tytul:\nCienie Red Blade\n\nHook 1:\nKupiec znika tej samej nocy co poslaniec frakcji.",
                ["01 Bible / Campaign Bible", "06 Threads / Thread Tracker"],
            )

            response = main.chat(
                main.ChatRequest(
                    message="Wymysl 3 hooki na nastepna sesje zwiazane z Red Blade.",
                    artifact_type="session_hooks",
                )
            )
        finally:
            main.generate_creative_artifact = original_generate_creative_artifact

        self.assertEqual(response.kind, "creative")
        self.assertEqual(response.artifact_type, "session_hooks")
        self.assertIn("Cienie Red Blade", response.reply)
        self.assertEqual(response.references, ["01 Bible / Campaign Bible", "06 Threads / Thread Tracker"])

    def test_chat_creative_defaults_to_session_hooks_when_artifact_not_given(self):
        original_generate_creative_artifact = main.generate_creative_artifact
        captured = {}

        def fake_generate_creative_artifact(**kwargs):
            captured.update(kwargs)
            return "Tytul:\nCienie Red Blade", []

        try:
            main.generate_creative_artifact = fake_generate_creative_artifact

            response = main.chat(
                main.ChatRequest(
                    message="Wymysl 3 hooki na nastepna sesje zwiazane z Red Blade.",
                )
            )
        finally:
            main.generate_creative_artifact = original_generate_creative_artifact

        self.assertEqual(response.kind, "creative")
        self.assertEqual(response.artifact_type, "session_hooks")
        self.assertEqual(captured["artifact_type"], "session_hooks")

    def test_collect_canonical_names_uses_world_model_matches(self):
        original_store = main.world_model_store_v2

        class FakeStore:
            def list_entities(self, limit=100, kind=None):
                return [
                    WorldEntityRecord(
                        id=1,
                        campaign_id="kng",
                        entity_kind="npc",
                        name="Captain Mira",
                        description="Desc",
                        tags=[],
                        last_session_id=None,
                        updated_at="2026-03-14T00:00:00+00:00",
                    )
                ]

            def list_threads(self, limit=100, status=None):
                return [
                    WorldThreadRecord(
                        id=1,
                        campaign_id="kng",
                        thread_key="T01",
                        thread_id="T01",
                        title="Red Blade",
                        status="active",
                        last_change="Change",
                        last_session_id=None,
                        updated_at="2026-03-14T00:00:00+00:00",
                    )
                ]

        try:
            main.world_model_store_v2 = FakeStore()
            names = main.collect_canonical_names(
                "Wymysl 3 hooki na sesje zwiazane z Red Blade i Captain Mira.",
                [],
            )
        finally:
            main.world_model_store_v2 = original_store

        self.assertEqual(names, ["Captain Mira", "Red Blade"])

    def test_generate_creative_artifact_session_hooks_uses_structured_sections(self):
        original_generate = main.gemini_generate
        original_build_world_model_context = main.build_world_model_context
        original_build_recent_sessions_context = main.build_recent_sessions_context
        original_build_context_for_planner = main.build_context_for_planner
        original_vector_search = main.vector_search
        original_render_source_labels = main.render_source_labels
        original_store = main.world_model_store_v2
        prompts = []

        class FakeStore:
            def list_entities(self, limit=100, kind=None):
                return [
                    WorldEntityRecord(
                        id=1,
                        campaign_id="kng",
                        entity_kind="npc",
                        name="Captain Mira",
                        description="Desc",
                        tags=[],
                        last_session_id=None,
                        updated_at="2026-03-14T00:00:00+00:00",
                    )
                ]

            def list_threads(self, limit=100, status=None):
                return [
                    WorldThreadRecord(
                        id=1,
                        campaign_id="kng",
                        thread_key="T01",
                        thread_id="T01",
                        title="Red Blade",
                        status="active",
                        last_change="Change",
                        last_session_id=None,
                        updated_at="2026-03-14T00:00:00+00:00",
                    )
                ]

        outputs = iter(
            [
                "Cienie Red Blade",
                "Captain Mira prosi BG o dyskretne spotkanie z emisariuszem Red Blade. To spotkanie moze uruchomic otwarty konflikt z wladzami.",
                "Dowody wskazuja, ze Red Blade testuje lojalnosc Captain Mira. Bohaterowie musza zdecydowac, czy wejda w te gre.",
                "BG odkrywaja, ze magazyn zaopatrzenia Red Blade zostal okradziony przez kogos z otoczenia Miry. Kradziez grozi eskalacja paniki i przemocy.",
                "* Utrata zaufania do Captain Mira.\n* Eskalacja konfliktu o zasoby.\n* Red Blade zyska przewage polityczna.\n* Bohaterowie straca wiarygodnosc wobec wladz.",
                "* Przygotuj emisariusza Red Blade.\n* Przygotuj magazyn i tropy po wlamaniu.\n* Przygotuj reakcje strazy miejskiej.\n* Przygotuj dowody laczace Mirę z Red Blade.",
            ]
        )

        def fake_generate(prompt, **kwargs):
            prompts.append(prompt)
            return next(outputs)

        try:
            main.world_model_store_v2 = FakeStore()
            main.gemini_generate = fake_generate
            main.build_world_model_context = lambda limit=30: "KNOWN ENTITIES:\n- npc: Captain Mira\nKNOWN THREADS:\n- T01 | Red Blade"
            main.build_recent_sessions_context = lambda limit=5: "RECENT SESSIONS:\n- session_id=1 | source=Session 05 | summary=Mira ujawnila kontakt."
            main.build_context_for_planner = lambda drive_store: "Campaign context about Red Blade."
            main.vector_search = lambda message, top_k: []
            main.render_source_labels = lambda hits: []

            artifact_text, references = main.generate_creative_artifact(
                message="Wymysl 3 hooki na nastepna sesje zwiazane z Red Blade i Captain Mira.",
                artifact_type="session_hooks",
            )
        finally:
            main.world_model_store_v2 = original_store
            main.gemini_generate = original_generate
            main.build_world_model_context = original_build_world_model_context
            main.build_recent_sessions_context = original_build_recent_sessions_context
            main.build_context_for_planner = original_build_context_for_planner
            main.vector_search = original_vector_search
            main.render_source_labels = original_render_source_labels

        self.assertEqual(references, [])
        self.assertIn("Tytul: Cienie Red Blade", artifact_text)
        self.assertIn("Hook 1:\nCaptain Mira prosi BG", artifact_text)
        self.assertIn("Hook 2:\nDowody wskazuja, ze Red Blade", artifact_text)
        self.assertIn("Hook 3:\nBG odkrywaja", artifact_text)
        self.assertIn("Stawki:\n* Utrata zaufania do Captain Mira.", artifact_text)
        self.assertIn("Co przygotowac:\n* Przygotuj emisariusza Red Blade.", artifact_text)
        self.assertNotIn("Do doprecyzowania.", artifact_text)
        self.assertTrue(any("Nie tlumacz nazw kanonicznych." in prompt for prompt in prompts))

    def test_generate_creative_artifact_npc_brief_uses_structured_sections(self):
        original_generate = main.gemini_generate
        original_build_world_model_context = main.build_world_model_context
        original_build_recent_sessions_context = main.build_recent_sessions_context
        original_build_context_for_planner = main.build_context_for_planner
        original_vector_search = main.vector_search
        original_render_source_labels = main.render_source_labels
        outputs = iter(
            [
                "Kael",
                "Dowodca polowy Red Blade, ktory wchodzi w spor z Captain Mira o metode przetrwania.",
                "Surowy, spokojny i stale gotowy do wydania rozkazu.",
                "Wierzy, ze tylko brutalna skutecznosc uratuje ludzi przed katastrofa.",
                "Ukrywa, ze juz poswiecil niewinnych w imie planu Red Blade.",
                "* Captain Mira: uwaza ja za zbyt miekka.\n* Red Blade: ma w frakcji lojalistow.\n* BG: chce wykorzystac ich jako narzedzie nacisku.",
                "* Moze zlecic BG moralnie watpliwa misje.\n* Moze podwazyc autorytet Captain Mira.\n* Moze doprowadzic do otwartego konfliktu w Red Blade.",
            ]
        )

        def fake_generate(prompt, **kwargs):
            return next(outputs)

        try:
            main.gemini_generate = fake_generate
            main.build_world_model_context = lambda limit=30: "KNOWN ENTITIES:\n- npc: Captain Mira\nKNOWN THREADS:\n- T01 | Red Blade"
            main.build_recent_sessions_context = lambda limit=5: "RECENT SESSIONS:\n- session_id=1 | source=Session 05 | summary=Mira ujawnila kontakt."
            main.build_context_for_planner = lambda drive_store: "Campaign context about Red Blade."
            main.vector_search = lambda message, top_k: []
            main.render_source_labels = lambda hits: []

            artifact_text, references = main.generate_creative_artifact(
                message="Stworz nowego NPC powiazanego z Red Blade, ktory moze wejsc w konflikt z Captain Mira.",
                artifact_type="npc_brief",
            )
        finally:
            main.gemini_generate = original_generate
            main.build_world_model_context = original_build_world_model_context
            main.build_recent_sessions_context = original_build_recent_sessions_context
            main.build_context_for_planner = original_build_context_for_planner
            main.vector_search = original_vector_search
            main.render_source_labels = original_render_source_labels

        self.assertEqual(references, [])
        self.assertIn("Imie: Kael", artifact_text)
        self.assertIn("Rola w kampanii:\nDowodca polowy Red Blade", artifact_text)
        self.assertIn("Sekret:\nUkrywa, ze juz poswiecil niewinnych", artifact_text)
        self.assertIn("Relacje:\n* Captain Mira:", artifact_text)
        self.assertIn("Jak uzyc tej postaci na sesji:\n* Moze zlecic BG", artifact_text)
        self.assertNotIn("Do doprecyzowania.", artifact_text)

    def test_ensure_artifact_shape_appends_missing_sections(self):
        original_generate = main.gemini_generate
        try:
            main.gemini_generate = lambda *args, **kwargs: ""
            shaped = main.ensure_artifact_shape(
                artifact_type="session_hooks",
                text="Tytul:\nCienie Red Blade\n\nHook 1:\nZaczepka na rynku.",
                repair_context="Kontekst testowy.",
            )
        finally:
            main.gemini_generate = original_generate

        self.assertIn("Hook 2:", shaped)
        self.assertIn("Hook 3:", shaped)
        self.assertIn("Stawki:", shaped)
        self.assertIn("Co przygotowac:", shaped)

    def test_ensure_artifact_shape_falls_back_when_repair_fails(self):
        original_generate = main.gemini_generate
        try:
            def fail_generate(*args, **kwargs):
                raise RuntimeError("repair failed")

            main.gemini_generate = fail_generate
            shaped = main.ensure_artifact_shape(
                artifact_type="npc_brief",
                text="Imie:\nVarek Krwawy Szpon",
                repair_context="Kontekst testowy.",
            )
        finally:
            main.gemini_generate = original_generate

        self.assertIn("Rola w kampanii:", shaped)
        self.assertIn("Sekret:", shaped)
        self.assertIn("Jak uzyc tej postaci na sesji:", shaped)

    def test_ensure_artifact_shape_uses_generated_missing_sections(self):
        original_generate = main.gemini_generate
        calls = {"count": 0}

        def fake_generate(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return "Tytul: Cienie Red Blade\n\nHook 1:\nZaczepka na rynku."
            return (
                "Hook 2:\nPoslaniec Red Blade znika przed spotkaniem.\n\n"
                "Hook 3:\nMira dostaje ultimatum od wlasnych ludzi.\n\n"
                "Stawki:\n- Miasto traci zaufanie do Miry.\n\n"
                "Co przygotowac:\n- Straznikow portowych.\n- Plotki o Red Blade."
            )

        try:
            main.gemini_generate = fake_generate
            shaped = main.ensure_artifact_shape(
                artifact_type="session_hooks",
                text="Tytul: Cienie Red Blade\n\nHook 1:\nZaczepka na rynku.",
                repair_context="Kontekst testowy.",
            )
        finally:
            main.gemini_generate = original_generate

        self.assertIn("Hook 2:\nPoslaniec Red Blade znika", shaped)
        self.assertIn("Hook 3:\nMira dostaje ultimatum", shaped)
        self.assertIn("Stawki:\n- Miasto traci zaufanie", shaped)
        self.assertIn("Co przygotowac:\n- Straznikow portowych.", shaped)

    def test_merge_artifact_sections_replaces_placeholder_content(self):
        merged = main.merge_artifact_sections(
            "Tytul: Cienie Red Blade\n\nHook 1:\nZaczepka.\n\nHook 2:\nDo doprecyzowania.",
            "Hook 2:\nPelny drugi hook.",
            "session_hooks",
        )

        self.assertIn("Hook 2:\nPelny drugi hook.", merged)
        self.assertNotIn("Hook 2:\nDo doprecyzowania.", merged)

    def test_markers_requiring_fill_detect_placeholder_sections(self):
        markers = main.markers_requiring_fill(
            "# Pre-Session Brief\n\n## Campaign State\nNapiecie rosnie.\n\n## Active Threads\n- Do doprecyzowania.",
            "pre_session_brief",
        )

        self.assertIn("## Active Threads", markers)
        self.assertIn("## Prep Checklist", markers)

    def test_markers_requiring_fill_detects_truncated_session_hooks_sections(self):
        markers = main.markers_requiring_fill(
            "Tytul:\nZa dlugi\nw dwoch liniach.\n\n"
            "Hook 1:\nCaptain Mira prosi BG o pomoc w sprawie Red Blade i\n\n"
            "Stawki:\n* Jedna stawka.",
            "session_hooks",
        )

        self.assertIn("Tytul:", markers)
        self.assertIn("Hook 1:", markers)
        self.assertIn("Stawki:", markers)

    def test_markers_requiring_fill_detects_incomplete_npc_lists(self):
        markers = main.markers_requiring_fill(
            "Imie: Kael\n\n"
            "Rola w kampanii:\nDowodca Red Blade, ktory naciska Captain Mira.\n\n"
            "Relacje:\n* Captain Mira: uwaza ja za zbyt miekka.",
            "npc_brief",
        )

        self.assertIn("Relacje:", markers)
        self.assertIn("Jak uzyc tej postaci na sesji:", markers)

    def test_markers_requiring_fill_detects_empty_bullet_items(self):
        markers = main.markers_requiring_fill(
            "Imie: Kael\n\n"
            "Rola w kampanii:\nDowodca Red Blade, ktory naciska Captain Mira.\n\n"
            "Relacje:\n* Captain Mira: uwaza ja za zbyt miekka.\n*",
            "npc_brief",
        )

        self.assertIn("Relacje:", markers)

    def test_sanitize_generated_section_flattens_multiline_title(self):
        sanitized = main.sanitize_generated_section(
            "Tytul:",
            "Tytul:\nMira i Red Blade: trudne wybory.\nRed Blade zada swojej ceny.",
        )

        self.assertEqual(sanitized, "Mira i Red Blade: trudne wybory.")

    def test_generate_creative_section_repairs_truncated_hook(self):
        original_generate = main.gemini_generate
        outputs = iter(
            [
                "Captain Mira odkrywa zdrade i prz",
                "Captain Mira odkrywa zdrade i prz",
                "Captain Mira odkrywa, ze Red Blade testuje jej lojalnosc. Bohaterowie musza szybko wybrac strone konfliktu.",
            ]
        )

        try:
            main.gemini_generate = lambda *args, **kwargs: next(outputs)
            section = main.generate_creative_section(
                artifact_type="session_hooks",
                marker="Hook 3:",
                instruction="Napisz 2-4 zdania. Ten hook ma byc wyraznie inny od poprzednich.",
                message="Wymysl 3 hooki na sesje zwiazane z Red Blade i Captain Mira.",
                world_context="Kontekst kampanii o Red Blade.",
                structured_context="KNOWN ENTITIES:\n- Captain Mira\nKNOWN THREADS:\n- T01 | Red Blade",
                recent_sessions_context="Session 05: Mira ujawnila kontakt.",
                canonical_names=["Captain Mira", "Red Blade"],
                prior_sections_text="Tytul: Cienie Red Blade",
                require_canonical_name=True,
            )
        finally:
            main.gemini_generate = original_generate

        self.assertIn("Captain Mira", section)
        self.assertIn("Red Blade", section)
        self.assertTrue(section.endswith("."))

    def test_generate_creative_section_repairs_incomplete_npc_list(self):
        original_generate = main.gemini_generate
        outputs = iter(
            [
                "* Captain Mira: uwaza ja za przeszkode.\n*",
                "* Captain Mira: uwaza ja za przeszkode.\n*",
                "* Captain Mira: uwaza ja za przeszkode w planach Red Blade.\n* Red Blade: ma tam lojalistow gotowych wykonywac rozkazy.",
                "BG: chce ich ustawic przeciwko Captain Mira i wymusic szybki wybor lojalnosci.",
            ]
        )

        try:
            main.gemini_generate = lambda *args, **kwargs: next(outputs)
            section = main.generate_creative_section(
                artifact_type="npc_brief",
                marker="Relacje:",
                instruction="Daj 3-5 bulletow zaczynajacych sie od '* ' o relacjach postaci.",
                message="Stworz nowego NPC powiazanego z Red Blade, ktory moze wejsc w konflikt z Captain Mira.",
                world_context="Kontekst kampanii o Red Blade.",
                structured_context="KNOWN ENTITIES:\n- Captain Mira\nKNOWN THREADS:\n- T01 | Red Blade",
                recent_sessions_context="Session 05: Mira ujawnila kontakt.",
                canonical_names=["Captain Mira", "Red Blade"],
                prior_sections_text="Imie: Kael",
                require_canonical_name=True,
            )
        finally:
            main.gemini_generate = original_generate

        self.assertIn("* Captain Mira:", section)
        self.assertIn("* Red Blade:", section)
        self.assertIn("* BG:", section)

    def test_chat_answer_can_save_output_doc(self):
        original_ask = main.ask
        original_drive_store = main.drive_store_v2

        class FakeDriveStore:
            def __init__(self):
                self.created = None

            def find_doc(self, folder=None, title=None, doc_id=None):
                return None

            def create_doc(self, folder, title, content, entity_type=None):
                self.created = {
                    "folder": folder,
                    "title": title,
                    "content": content,
                    "entity_type": entity_type,
                }
                return main.WorldDocInfo(
                    folder=folder,
                    title=title,
                    doc_id="out-1",
                    path_hint=f"{folder}/{title}",
                    entity_type=main.WorldEntityType.output,
                )

        fake_drive_store = FakeDriveStore()

        try:
            main.ask = lambda req: main.AskResponse(answer="Gotowy tekst.", sources=[])
            main.drive_store_v2 = fake_drive_store

            response = main.chat(
                main.ChatRequest(
                    message="Odpowiedz krotko na pytanie.",
                    intent="answer",
                    save_output=True,
                    output_title="Answer 01",
                )
            )
        finally:
            main.ask = original_ask
            main.drive_store_v2 = original_drive_store

        self.assertEqual(response.output_doc_id, "out-1")
        self.assertEqual(response.output_title, "Answer 01")
        self.assertEqual(fake_drive_store.created["folder"], "08 Outputs")
        self.assertEqual(fake_drive_store.created["title"], "Answer 01")
        self.assertEqual(fake_drive_store.created["content"], "Gotowy tekst.")

    def test_chat_answer_saves_artifact_content_when_requested(self):
        original_ask = main.ask
        original_drive_store = main.drive_store_v2

        class FakeDriveStore:
            def __init__(self):
                self.created = None

            def find_doc(self, folder=None, title=None, doc_id=None):
                return None

            def create_doc(self, folder, title, content, entity_type=None):
                self.created = {
                    "folder": folder,
                    "title": title,
                    "content": content,
                    "entity_type": entity_type,
                }
                return main.WorldDocInfo(
                    folder=folder,
                    title=title,
                    doc_id="out-2",
                    path_hint=f"{folder}/{title}",
                    entity_type=main.WorldEntityType.output,
                )

        fake_drive_store = FakeDriveStore()

        try:
            main.ask = lambda req: main.AskResponse(answer="Gotowy tekst.", sources=[])
            main.drive_store_v2 = fake_drive_store

            response = main.chat(
                main.ChatRequest(
                    message="Odpowiedz krotko na pytanie.",
                    intent="answer",
                    artifact_type="player_summary",
                    save_output=True,
                    output_title="Player Summary 01",
                )
            )
        finally:
            main.ask = original_ask
            main.drive_store_v2 = original_drive_store

        self.assertEqual(response.output_doc_id, "out-2")
        self.assertEqual(response.output_title, "Player Summary 01")
        self.assertIn("# Player Summary", fake_drive_store.created["content"])
        self.assertNotEqual(fake_drive_store.created["content"], "Gotowy tekst.")

    def test_chat_answer_returns_warning_when_output_save_fails(self):
        original_ask = main.ask
        original_drive_store = main.drive_store_v2

        class FakeDriveStore:
            def find_doc(self, folder=None, title=None, doc_id=None):
                return None

            def create_doc(self, folder, title, content, entity_type=None):
                raise RuntimeError("storageQuotaExceeded")

        try:
            main.ask = lambda req: main.AskResponse(answer="Gotowy tekst.", sources=[])
            main.drive_store_v2 = FakeDriveStore()

            response = main.chat(
                main.ChatRequest(
                    message="Odpowiedz krotko na pytanie.",
                    intent="answer",
                    save_output=True,
                    output_title="Answer 01",
                )
            )
        finally:
            main.ask = original_ask
            main.drive_store_v2 = original_drive_store

        self.assertEqual(response.kind, "answer")
        self.assertEqual(response.reply, "Gotowy tekst.")
        self.assertEqual(response.output_doc_id, None)
        self.assertEqual(len(response.warnings), 1)
        self.assertIn("storageQuotaExceeded", response.warnings[0])

    def test_chat_answer_falls_back_to_rollup_doc_on_storage_quota(self):
        original_ask = main.ask
        original_drive_store = main.drive_store_v2
        original_rollup_doc_id = main.OUTPUT_ROLLUP_DOC_ID
        original_rollup_doc_title = main.OUTPUT_ROLLUP_DOC_TITLE
        original_rollup_mode = main.OUTPUT_ROLLUP_MODE

        class FakeDriveStore:
            def __init__(self):
                self.replaced = None

            def find_doc(self, folder=None, title=None, doc_id=None):
                if doc_id == "rollup-1":
                    return main.WorldDocInfo(
                        folder="08 Outputs",
                        title="Agent Inbox",
                        doc_id="rollup-1",
                        path_hint="08 Outputs/Agent Inbox",
                        entity_type=main.WorldEntityType.output,
                    )
                return None

            def create_doc(self, folder, title, content, entity_type=None):
                raise RuntimeError("storageQuotaExceeded")

            def replace_doc(self, doc_ref, content):
                self.replaced = {"doc_ref": doc_ref, "content": content}

        fake_drive_store = FakeDriveStore()

        try:
            main.ask = lambda req: main.AskResponse(answer="Gotowy tekst.", sources=[])
            main.drive_store_v2 = fake_drive_store
            main.OUTPUT_ROLLUP_DOC_ID = "rollup-1"
            main.OUTPUT_ROLLUP_DOC_TITLE = "Agent Inbox"
            main.OUTPUT_ROLLUP_MODE = "replace"

            response = main.chat(
                main.ChatRequest(
                    message="Odpowiedz krotko na pytanie.",
                    intent="answer",
                    save_output=True,
                    output_title="Answer 01",
                )
            )
        finally:
            main.ask = original_ask
            main.drive_store_v2 = original_drive_store
            main.OUTPUT_ROLLUP_DOC_ID = original_rollup_doc_id
            main.OUTPUT_ROLLUP_DOC_TITLE = original_rollup_doc_title
            main.OUTPUT_ROLLUP_MODE = original_rollup_mode

        self.assertEqual(response.output_doc_id, "rollup-1")
        self.assertEqual(response.output_path, "08 Outputs/Agent Inbox")
        self.assertEqual(len(response.warnings), 1)
        self.assertIn("fallback dokumentu 08 Outputs/Agent Inbox", response.warnings[0])
        self.assertEqual(fake_drive_store.replaced["doc_ref"].doc_id, "rollup-1")
        self.assertEqual(fake_drive_store.replaced["content"], "Gotowy tekst.")

    def test_chat_session_sync_returns_human_summary(self):
        original_sync = main.ingest_session_and_sync
        try:
            main.ingest_session_and_sync = lambda req: main.IngestAndSyncSessionResponse(
                patch=main.SessionPatch(
                    session_summary="Captain Mira ujawnila tajny kontakt z Red Blade.",
                    thread_tracker_patch=[
                        main.ThreadPatch(
                            thread_id="T01",
                            title="Red Blade",
                            status="Updated",
                            change="Ujawniono tajny kontakt Captain Miry.",
                        )
                    ],
                    entities_patch=[
                        main.EntityPatch(
                            kind="npc",
                            name="Captain Mira",
                            description="Tajny kontakt Red Blade.",
                            tags=[],
                        )
                    ],
                    rag_additions=[],
                ),
                sync=main.SyncSessionPatchResponse(
                    session_id=11,
                    campaign_id="kng",
                    summary="Session patch synced into world model",
                    entity_count=1,
                    thread_count=1,
                ),
            )

            response = main.chat(
                main.ChatRequest(
                    message="Captain Mira ujawnila tajny kontakt z Red Blade.\nTo zmienia watek frakcji.",
                    intent="session_sync",
                    source_title="Session 06",
                )
            )
        finally:
            main.ingest_session_and_sync = original_sync

        self.assertEqual(response.kind, "session_sync")
        self.assertEqual(response.session_id, 11)
        self.assertIn("Zaktualizowalem model swiata z notatek.", response.reply)
        self.assertIn("T01 / Red Blade", response.reply)

    def test_chat_session_sync_can_render_session_report_artifact(self):
        original_sync = main.ingest_session_and_sync
        try:
            main.ingest_session_and_sync = lambda req: main.IngestAndSyncSessionResponse(
                patch=main.SessionPatch(
                    session_summary="Captain Mira ujawnila tajny kontakt z Red Blade.",
                    thread_tracker_patch=[
                        main.ThreadPatch(
                            thread_id="T01",
                            title="Red Blade",
                            status="Updated",
                            change="Ujawniono tajny kontakt Captain Miry.",
                        )
                    ],
                    entities_patch=[
                        main.EntityPatch(
                            kind="npc",
                            name="Captain Mira",
                            description="Tajny kontakt Red Blade.",
                            tags=[],
                        )
                    ],
                    rag_additions=["Captain Mira ma tajny kontakt z Red Blade."],
                ),
                sync=main.SyncSessionPatchResponse(
                    session_id=12,
                    campaign_id="kng",
                    summary="Session patch synced into world model",
                    entity_count=1,
                    thread_count=1,
                ),
            )

            response = main.chat(
                main.ChatRequest(
                    message="Captain Mira ujawnila tajny kontakt z Red Blade.\nTo zmienia watek frakcji.",
                    intent="session_sync",
                    source_title="Session 06",
                    artifact_type="session_report",
                )
            )
        finally:
            main.ingest_session_and_sync = original_sync

        self.assertEqual(response.artifact_type, "session_report")
        self.assertIn("# Session Report", response.artifact_text)
        self.assertIn("## Executive Summary", response.artifact_text)
        self.assertIn("## Threads", response.artifact_text)
        self.assertIn("## Facts For Retrieval", response.artifact_text)
        self.assertIn("## Suggested Document Follow-ups", response.artifact_text)
        self.assertIn("## Prep For Next Session", response.artifact_text)

    def test_chat_pre_session_brief_uses_brief_generator(self):
        original_generate_pre_session_brief = main.generate_pre_session_brief
        try:
            main.generate_pre_session_brief = lambda message: (
                "# Pre-Session Brief\n\n## Campaign State\nNapiecie rosnie.\n\n## Active Threads\n- T01 / Red Blade",
                ["06 Threads / Thread Tracker"],
            )

            response = main.chat(
                main.ChatRequest(
                    message="Przygotuj briefing przed sesja o Red Blade i Captain Mira.",
                )
            )
        finally:
            main.generate_pre_session_brief = original_generate_pre_session_brief

        self.assertEqual(response.kind, "answer")
        self.assertEqual(response.artifact_type, "pre_session_brief")
        self.assertIn("# Pre-Session Brief", response.artifact_text)
        self.assertEqual(response.references, ["06 Threads / Thread Tracker"])

    def test_generate_pre_session_brief_uses_structured_sections(self):
        original_generate = main.gemini_generate
        original_build_world_model_context = main.build_world_model_context
        original_build_recent_sessions_context = main.build_recent_sessions_context
        original_vector_search = main.vector_search
        original_build_context_for_planner = main.build_context_for_planner
        original_render_source_labels = main.render_source_labels

        outputs = iter(
            [
                "* T01 / Red Blade nabiera znaczenia.\n* Captain Mira jest pod presja polityczna.\n* Bohaterowie sa w centrum konfliktu lojalnosci.",
                "* T01 / Red Blade nabiera znaczenia.\n* T02 / Walka o zasoby eskaluje.\n* T05 / Peknieta przysiega wraca jako zagrozenie.",
                "* Captain Mira jest kluczowa dla kolejnej sesji.\n* Red Blade naciska na szybkie decyzje.\n* Wladze miasta obserwuja bohaterow.",
                "* Wyciek informacji rozbije sojusze.\n* Red Blade moze wymusic brutalne ruchy.\n* Bohaterowie zaplaca za zly wybor polityczny.",
                "* Przesluchanie poslanca Red Blade.\n* Konfrontacja z Captain Mira.\n* Spotkanie z rada miasta po ujawnieniu kontaktu.",
                "* Przygotuj reakcje Red Blade.\n* Przygotuj konsekwencje polityczne.\n* Przygotuj sceny nacisku na bohaterow.",
            ]
        )

        def fake_generate(prompt, **kwargs):
            return next(outputs)

        try:
            main.gemini_generate = fake_generate
            main.build_world_model_context = lambda limit=40: "WORLD MODEL"
            main.build_recent_sessions_context = lambda limit=6: "RECENT SESSIONS"
            main.vector_search = lambda message, top_k: []
            main.build_context_for_planner = lambda drive_store: "CAMPAIGN CONTEXT"
            main.render_source_labels = lambda hits: []

            artifact_text, references = main.generate_pre_session_brief(
                "Przygotuj briefing przed sesja o Red Blade."
            )
        finally:
            main.gemini_generate = original_generate
            main.build_world_model_context = original_build_world_model_context
            main.build_recent_sessions_context = original_build_recent_sessions_context
            main.vector_search = original_vector_search
            main.build_context_for_planner = original_build_context_for_planner
            main.render_source_labels = original_render_source_labels

        self.assertEqual(references, [])
        self.assertIn("# Pre-Session Brief", artifact_text)
        self.assertIn("## Active Threads", artifact_text)
        self.assertIn("## Key NPCs and Factions", artifact_text)
        self.assertIn("## Risks and Pressure Points", artifact_text)
        self.assertIn("## Scene Opportunities", artifact_text)
        self.assertIn("## Prep Checklist", artifact_text)
        self.assertIn("T01 / Red Blade nabiera znaczenia.", artifact_text)
        self.assertNotIn("Do doprecyzowania.", artifact_text)

    def test_chat_proposal_returns_human_summary(self):
        original_drive_store = main.drive_store_v2
        original_planner = main.planner_v2
        original_workflow_store = main.workflow_store_v2
        original_build_context = main.build_context_for_planner

        class FakeDriveStore:
            def list_world_docs(self):
                return []

        class FakePlanner:
            def propose(self, request, world_docs, world_context):
                return ChangeProposal(
                    summary="Podmien sekcje Test Automation w Campaign Bible.",
                    user_goal=request.instruction,
                    impacted_docs=[DocumentRef(folder="01 Bible", title="Campaign Bible")],
                    actions=[],
                    needs_confirmation=True,
                )

        class FakeWorkflowStore:
            def save_proposal(self, request, proposal):
                return 17

        try:
            main.drive_store_v2 = FakeDriveStore()
            main.planner_v2 = FakePlanner()
            main.workflow_store_v2 = FakeWorkflowStore()
            main.build_context_for_planner = lambda drive_store: "world context"

            response = main.chat(
                main.ChatRequest(
                    message="W dokumencie Campaign Bible podmien sekcje Test Automation na nowy tekst.",
                    intent="proposal",
                )
            )
        finally:
            main.drive_store_v2 = original_drive_store
            main.planner_v2 = original_planner
            main.workflow_store_v2 = original_workflow_store
            main.build_context_for_planner = original_build_context

        self.assertEqual(response.kind, "proposal")
        self.assertEqual(response.proposal_id, 17)
        self.assertIn("Przygotowalem propozycje zmiany.", response.reply)
        self.assertIn("Proposal ID: 17", response.reply)


if __name__ == "__main__":
    unittest.main()
