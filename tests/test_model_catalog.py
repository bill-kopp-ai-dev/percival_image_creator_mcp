import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_catalog import (  # noqa: E402
    find_alternatives,
    get_catalog_metadata,
    get_model_card,
    list_model_cards,
    load_catalog,
)


class TestModelCatalog(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = load_catalog(use_cache=False)

    def test_catalog_schema_and_metadata(self) -> None:
        metadata = get_catalog_metadata(self.catalog)
        self.assertEqual(metadata["schema_version"], "2.1")
        self.assertEqual(metadata["provider"], "venice.ai")
        self.assertGreater(metadata["model_count"], 0)
        self.assertIn("text_to_image", metadata["supported_task_types"])
        self.assertIn("image_edit", metadata["supported_task_types"])

    def test_task_filters(self) -> None:
        generation = list_model_cards("text_to_image", catalog=self.catalog)
        edits = list_model_cards("image_edit", catalog=self.catalog)
        self.assertTrue(generation)
        self.assertTrue(edits)
        self.assertTrue(all("text_to_image" in card["task_types"] for card in generation))
        self.assertTrue(all("image_edit" in card["task_types"] for card in edits))

    def test_card_lookup_and_alternatives(self) -> None:
        card = get_model_card("venice-sd35", catalog=self.catalog)
        self.assertIsNotNone(card)
        assert card is not None
        self.assertIn("text_to_image", card["task_types"])
        self.assertIn("recommended_api_params", card)
        self.assertIsInstance(card["recommended_api_params"], dict)

        alternatives = find_alternatives(
            model_id="venice-sd35",
            task_type="text_to_image",
            max_results=3,
            catalog=self.catalog,
        )
        self.assertLessEqual(len(alternatives), 3)
        self.assertTrue(all(alt["id"] != "venice-sd35" for alt in alternatives))


if __name__ == "__main__":
    unittest.main()
