import unittest
from LocalGeoLite import load_model, code, text, unload_model


class exampleUsages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_model(cache_dir="D:/HuggingFace/Models")

    def test_code(self):
        prompt = "How to calculate NDVI using rasterio?"
        code(prompt, 512)

    def test_text(self):
        prompt = "What is NDVI and how is it used in remote sensing?"
        text(prompt)


if __name__ == '__main__':
    unittest.main()
