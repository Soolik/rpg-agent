import unittest

from app.routed_drive_store import RoutedDriveStore


class RoutedDriveStoreTest(unittest.TestCase):
    def test_writes_use_user_store_when_available(self):
        class FakeStore:
            def __init__(self, label):
                self.label = label
                self.calls = []

            def create_doc(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return self.label

            def list_world_docs(self):
                return [self.label]

        read_store = FakeStore("read")
        user_store = FakeStore("user")
        store = RoutedDriveStore(read_store=read_store, write_store_factory=lambda: user_store)

        created = store.create_doc(folder="03 NPC", title="Captain Mira", content="...", entity_type="npc")

        self.assertEqual(created, "user")
        self.assertEqual(len(user_store.calls), 1)
        self.assertEqual(store.list_world_docs(), ["read"])

    def test_writes_fall_back_to_read_store_when_user_store_missing(self):
        class FakeStore:
            def __init__(self):
                self.calls = []

            def create_doc(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return "read"

        read_store = FakeStore()
        store = RoutedDriveStore(read_store=read_store, write_store_factory=lambda: None)

        created = store.create_doc(folder="03 NPC", title="Captain Mira", content="...", entity_type="npc")

        self.assertEqual(created, "read")
        self.assertEqual(len(read_store.calls), 1)


if __name__ == "__main__":
    unittest.main()
