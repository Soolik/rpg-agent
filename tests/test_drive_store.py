import unittest

from app.drive_store import replace_section_content


class DriveStoreSectionReplaceTest(unittest.TestCase):
    def test_replace_existing_section_body(self):
        original = (
            "# NPC\n\n"
            "## Identity\n\n"
            "- Name: Mira\n\n"
            "## Secrets\n\n"
            "Old secret\n\n"
            "## Relationships\n\n"
            "Trusted crew\n"
        )

        updated = replace_section_content(original, "Secrets", "New secret line")

        self.assertIn("## Secrets\n\nNew secret line\n", updated)
        self.assertNotIn("Old secret", updated)
        self.assertIn("## Relationships\n\nTrusted crew\n", updated)

    def test_append_section_when_missing(self):
        original = "# NPC\n\n## Identity\n\n- Name: Mira\n"

        updated = replace_section_content(original, "Motivations", "Protect the crew")

        self.assertTrue(updated.endswith("## Motivations\n\nProtect the crew\n"))

    def test_keep_nested_headings_inside_section(self):
        original = (
            "# NPC\n\n"
            "## Secrets\n\n"
            "Old secret\n\n"
            "### Known by\n\n"
            "Nobody\n\n"
            "## Relationships\n\n"
            "Trusted crew\n"
        )

        updated = replace_section_content(original, "Secrets", "Replaced secret")

        self.assertIn("## Secrets\n\nReplaced secret\n", updated)
        self.assertNotIn("### Known by", updated)
        self.assertIn("## Relationships\n\nTrusted crew\n", updated)


if __name__ == "__main__":
    unittest.main()
