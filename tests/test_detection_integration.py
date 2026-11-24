import numpy as np

from analysis.detection import detect_poster_elements


class FakeDetector:
    def __init__(self, weights_path=None, conf_threshold=0.25):
        self.is_available = True

    def predict_array(self, array, conf=0.25):
        h, w = array.shape[:2]
        # Return one box in pixel coords
        return [
            {
                "class_id": 3,
                "class_name": "Product",
                "confidence": 0.9,
                "x1": int(0.1 * w),
                "y1": int(0.2 * h),
                "x2": int(0.4 * w),
                "y2": int(0.5 * h),
            }
        ]

    def predict(self, image_path, conf=0.25):
        # create a dummy image for predict()
        return self.predict_array(np.zeros((600, 800, 3), dtype=np.uint8), conf=conf)


def test_detect_poster_elements_with_fake_detector(monkeypatch):
    # Patch the symbol referenced inside analysis.detection
    monkeypatch.setattr("analysis.detection.PosterDetector", FakeDetector)

    img = np.zeros((600, 800, 3), dtype=np.uint8)
    records, source = detect_poster_elements(img, image_path=None, weights_path=None)

    assert source == "model"
    assert len(records) == 1
    rec = records[0]
    assert rec.class_id == 3
    assert rec.class_name.lower() == "product"
    assert rec.confidence == 0.9


def test_detector_is_cached_and_duplicates_merged(monkeypatch):
    from analysis import detection

    created = []

    class CountingDetector(FakeDetector):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

        def predict_array(self, array, conf=0.25):
            box = {"class_id": 4, "class_name": "logo", "x1": 10, "y1": 10, "x2": 100, "y2": 100}
            return [{**box, "confidence": 0.9}, {**box, "x2": 105, "confidence": 0.5},
                    {"class_id": 3, "class_name": "Product", "confidence": 0.8, "x1": 12, "y1": 12, "x2": 98, "y2": 98}]

    monkeypatch.setattr(detection, "PosterDetector", CountingDetector)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    for _ in range(3):
        records, _ = detect_poster_elements(img)
    assert len(created) == 1  # loaded once, reused
    assert sorted(r.class_name for r in records) == ["Product", "logo"]  # duplicate logo merged, other class kept
