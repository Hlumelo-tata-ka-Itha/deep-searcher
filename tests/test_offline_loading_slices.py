import unittest
from unittest.mock import MagicMock, patch

from deepsearcher import offline_loading


class TestEmbedAndInsertSlices(unittest.TestCase):
    """A large load must embed+insert in bounded slices, not hold every vector at once."""

    def test_slices_bound_memory_and_insert_everything(self):
        embed_sizes, inserted = [], []
        model = MagicMock()
        model.embed_chunks.side_effect = lambda c, batch_size: embed_sizes.append(len(c)) or c
        db = MagicMock()
        db.insert_data.side_effect = lambda collection, chunks: inserted.extend(chunks)

        with patch.object(offline_loading, "EMBED_INSERT_SLICE", 2):
            offline_loading._embed_and_insert(list(range(5)), model, db, "c", 256)

        self.assertEqual(embed_sizes, [2, 2, 1])
        self.assertEqual(inserted, list(range(5)))


if __name__ == "__main__":
    unittest.main()
