import unittest

from app.canon_guard import build_continuity_report


class CanonGuardTest(unittest.TestCase):
    def test_build_continuity_report_flags_unknown_proper_noun(self):
        report = build_continuity_report(
            message="Przygotuj briefing przed sesja o Red Blade i Captain Mira.",
            generated_text=(
                "# Pre-Session Brief\n\n"
                "## Key NPCs and Factions\n\n"
                "* **Captain Mira** - jest pod presja.\n"
                "* **Red Blade** - eskaluje konflikt.\n"
                "* **Skup** - wchodzi do gry politycznej."
            ),
            known_entity_names=["Captain Mira"],
            known_thread_names=["Red Blade"],
        )

        self.assertFalse(report.ok)
        self.assertEqual(report.source_backed_names, ["Captain Mira", "Red Blade"])
        self.assertIn("Skup", report.proposed_new_names)
        self.assertTrue(any(issue.related_name == "Skup" for issue in report.issues))

    def test_build_continuity_report_allows_new_name_when_requested(self):
        report = build_continuity_report(
            message="Stworz nowego NPC powiazanego z Red Blade.",
            generated_text="Imie: Kaelen\n\nRola w kampanii:\nKaelen pracuje dla Red Blade.",
            known_entity_names=["Captain Mira"],
            known_thread_names=["Red Blade"],
            allow_proposed_new_names=True,
        )

        self.assertTrue(report.ok)
        self.assertIn("Kaelen", report.proposed_new_names)
        self.assertTrue(all(issue.severity == "info" for issue in report.issues))


if __name__ == "__main__":
    unittest.main()
