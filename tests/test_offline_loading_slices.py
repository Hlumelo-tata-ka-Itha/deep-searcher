import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from deepsearcher import offline_loading


class TestEmbedAndInsertSlices(unittest.TestCase):
    """A large load must embed+insert in bounded slices, not hold every vector at once."""

    def test_slices_bound_memory_and_insert_everything(self):
        chunks = [SimpleNamespace(id=i, embedding=None) for i in range(5)]
        embed_sizes, inserted, held = [], [], []

        def embed(part, batch_size):
            embed_sizes.append(len(part))
            held.append(sum(c.embedding is not None for c in chunks))  # vectors alive before this slice
            for c in part:
                c.embedding = [0.0]
            return part

        model = MagicMock()
        model.embed_chunks.side_effect = embed
        db = MagicMock()
        db.insert_data.side_effect = lambda collection, chunks: inserted.extend(c.id for c in chunks)

        with patch.object(offline_loading, "EMBED_INSERT_SLICE", 2):
            offline_loading._embed_and_insert(chunks, model, db, "c", 256)

        self.assertEqual(embed_sizes, [2, 2, 1])
        self.assertEqual(inserted, list(range(5)))
        self.assertEqual(held, [0, 0, 0])  # earlier slices' vectors were released
        self.assertTrue(all(c.embedding is None for c in chunks))


if __name__ == "__main__":
    unittest.main()
