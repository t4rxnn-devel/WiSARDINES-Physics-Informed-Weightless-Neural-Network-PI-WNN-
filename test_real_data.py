import tempfile
import unittest
from pathlib import Path
import zipfile

import numpy as np

from real_data import iter_higgs_chunks, load_csv, stratified_split


class RealDataTests(unittest.TestCase):
    def test_csv_loader_and_stratified_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.csv"
            path.write_text(
                "p1x,p1y,p2x,p2y,label,group\n"
                "1,0,0,1,a,g1\n2,0,0,2,a,g1\n"
                "3,0,0,3,b,g2\n4,0,0,4,b,g2\n",
                encoding="utf-8",
            )
            dataset = load_csv(path, ["p1x", "p1y", "p2x", "p2y"], "label", "group")
            train, test = stratified_split(dataset, test_fraction=0.5, seed=2)
            self.assertEqual(train.features.shape[1], 4)
            self.assertTrue(set(train.groups).isdisjoint(set(test.groups)))

    def test_loader_rejects_nonfinite_values(self) -> None:
        with self.assertRaises(ValueError):
            from real_data import _validate
            _validate(np.array([[np.nan, 0, 0, 1]] * 2), np.array([0, 1]))

    def test_higgs_stream_selects_all_28_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "higgs.zip"
            row = ",".join(["1"] + [str(index) for index in range(1, 29)]) + "\n"
            row += ",".join(["0"] + [str(index) for index in range(1, 29)]) + "\n"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("HIGGS.csv", row)
            features, labels = next(iter_higgs_chunks(path, chunk_size=1))
            self.assertEqual(features.shape, (1, 28))
            np.testing.assert_array_equal(features[0], np.arange(1, 29))
            np.testing.assert_array_equal(labels, [1])


if __name__ == "__main__":
    unittest.main()