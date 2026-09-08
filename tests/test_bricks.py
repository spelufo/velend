import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "vlend_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules.setdefault(PACKAGE, package)
sys.modules.setdefault(PACKAGE + ".state", types.ModuleType(PACKAGE + ".state"))
spec = importlib.util.spec_from_file_location(PACKAGE + ".bricks", ROOT / "bricks.py")
bricks = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bricks
spec.loader.exec_module(bricks)


class FakeVolume:
	def __init__(self):
		self.requests = []

	def __getitem__(self, index):
		self.requests.append(index)
		z, y, x, _level = index
		shape = (
			z.stop - z.start,
			y.stop - y.start,
			x.stop - x.start,
		)
		return np.full(shape, 255.0, dtype=np.float32)


class BrickTests(unittest.TestCase):
	def test_read_brick_uses_requested_level_and_boundary_padding(self):
		volume = FakeVolume()
		brick = bricks.read_brick(volume, (6, 5, 4), (0, 0, 0), level=1)

		self.assertEqual(volume.requests[0][-1], 1)
		self.assertEqual(brick.shape, (66, 66, 66))
		np.testing.assert_array_equal(brick[1:5, 1:6, 1:7], 1.0)
		self.assertEqual(float(brick[0, 0, 0]), 0.0)

	def test_l1_is_empty_until_l0_capacity_is_exceeded(self):
		chunks = np.stack(
			[np.arange(20), np.zeros(20, dtype=int), np.zeros(20, dtype=int)],
			axis=1,
		)
		(l0_keys, _), (l1_keys, _) = bricks.select_lod_chunks(
			chunks, (64, 64, 64), (32, 32, 32), (0, 0, 0)
		)
		self.assertEqual(len(l0_keys), 20)
		self.assertEqual(len(l1_keys), 0)

	def test_overflow_collapses_to_l1_and_obeys_both_capacities(self):
		parents = np.indices((16, 16, 16)).reshape(3, -1).T
		chunks = parents * 2
		(l0_keys, _), (l1_keys, l1_coords) = bricks.select_lod_chunks(
			chunks, (32, 32, 32), (16, 16, 16), (0, 0, 0)
		)
		self.assertEqual(len(l0_keys), 12 ** 3)
		self.assertEqual(len(l1_keys), 10 ** 3)
		self.assertEqual(len(np.unique(l1_coords, axis=0)), len(l1_coords))

	def test_residency_capacity_and_failed_upload_rollback(self):
		residency = bricks.Residency((2, 2, 2), slot_count=1)
		self.assertEqual(residency.place(3), 0)
		self.assertIsNone(residency.place(4))
		residency.release(3)
		self.assertEqual(float(residency.flat[3]), 0.0)
		self.assertEqual(residency.place(4), 0)


if __name__ == "__main__":
	unittest.main()
