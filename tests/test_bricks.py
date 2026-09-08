import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "velend_test"
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


class EmptyVolume(FakeVolume):
	def __getitem__(self, index):
		data = super().__getitem__(index)
		data.fill(0.0)
		return data


class BrickTests(unittest.TestCase):
	def test_read_brick_uses_requested_level_and_boundary_padding(self):
		volume = FakeVolume()
		brick = bricks.read_brick(volume, (6, 5, 4), (0, 0, 0), level=4)

		self.assertEqual(volume.requests[0][-1], 4)
		self.assertEqual(brick.shape, (66, 66, 66))
		np.testing.assert_array_equal(brick[1:5, 1:6, 1:7], 1.0)
		self.assertEqual(float(brick[0, 0, 0]), 0.0)

	def test_fully_black_read_is_reported_as_empty(self):
		brick = bricks.read_brick(EmptyVolume(), (6, 5, 4), (0, 0, 0), level=2)
		self.assertIs(brick, bricks.EMPTY_BRICK)

	def test_coarser_levels_are_empty_until_finer_capacity_is_exceeded(self):
		chunks = np.stack(
			[np.arange(20), np.zeros(20, dtype=int), np.zeros(20, dtype=int)],
			axis=1,
		)
		selected = bricks.select_lod_chunks(
			chunks,
			{0: (64, 64, 64), 1: (32, 32, 32), 2: (16, 16, 16)},
			(0, 0, 0),
			slot_counts=(20, 10, 5),
		)
		self.assertEqual([len(selected[level][0]) for level in range(3)], [20, 0, 0])

	def test_overflow_cascades_through_all_levels(self):
		parents = np.indices((16, 16, 16)).reshape(3, -1).T
		chunks = parents * 2
		selected = bricks.select_lod_chunks(
			chunks,
			{
				0: (32, 32, 32),
				1: (16, 16, 16),
				2: (8, 8, 8),
				3: (4, 4, 4),
				4: (2, 2, 2),
			},
			(0, 0, 0),
			slot_counts=(2, 2, 2, 2, 2),
		)
		self.assertEqual([len(selected[level][0]) for level in range(5)], [2, 2, 2, 2, 2])
		for level in range(5):
			coords = selected[level][1]
			self.assertEqual(len(np.unique(coords, axis=0)), len(coords))

	def test_default_memory_budgets_derive_expected_atlas_sizes(self):
		self.assertEqual(bricks.SLOTS_PER_AXIS, (12, 10, 8, 6, 4))
		for budget, slots in zip(bricks.ATLAS_MEMORY_MB, bricks.SLOTS_PER_AXIS):
			used = slots ** 3 * bricks.BRICK_SIZE ** 3
			next_size = (slots + 1) ** 3 * bricks.BRICK_SIZE ** 3
			self.assertLessEqual(used, budget * 1_000_000)
			self.assertGreater(next_size, budget * 1_000_000)

	def test_memory_budget_must_fit_one_padded_brick(self):
		with self.assertRaises(ValueError):
			bricks.atlas_slots_per_axis(0)

	def test_residency_capacity_and_failed_upload_rollback(self):
		residency = bricks.Residency((2, 2, 2), slot_count=1)
		self.assertEqual(residency.place(3), 0)
		self.assertIsNone(residency.place(4))
		residency.release(3)
		self.assertEqual(float(residency.flat[3]), 0.0)
		self.assertEqual(residency.place(4), 0)


if __name__ == "__main__":
	unittest.main()
