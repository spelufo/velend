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
spec = importlib.util.spec_from_file_location(PACKAGE + ".bricks", ROOT / "bricks.py")
bricks = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bricks
spec.loader.exec_module(bricks)


class FakeArray:
	"""Stands in for one level's zarr array."""

	def __init__(self, shape, fill=255):
		self.shape = shape
		self.fill = fill
		self.requests = []

	def __getitem__(self, index):
		self.requests.append(index)
		z, y, x = index
		shape = (
			z.stop - z.start,
			y.stop - y.start,
			x.stop - x.start,
		)
		return np.full(shape, self.fill, dtype=np.uint8)


class VoxelArray:
	"""Stands in for a level whose voxels are not all the same."""

	def __init__(self, voxels_zyx):
		self.voxels = voxels_zyx
		self.shape = voxels_zyx.shape

	def __getitem__(self, index):
		return self.voxels[index]


class BrickTests(unittest.TestCase):
	def test_read_brick_clips_to_the_array_and_pads_the_boundary(self):
		array = FakeArray((4, 5, 6))
		brick = bricks.read_brick(array, (0, 0, 0))

		self.assertEqual(
			array.requests[0], (slice(0, 4), slice(0, 5), slice(0, 6))
		)
		self.assertEqual(brick.shape, (66, 66, 66))
		np.testing.assert_array_equal(brick[1:5, 1:6, 1:7], 1.0)
		self.assertEqual(float(brick[0, 0, 0]), 0.0)

	def test_fully_black_read_is_reported_as_empty(self):
		brick = bricks.read_brick(FakeArray((4, 5, 6), fill=0), (0, 0, 0))
		self.assertIs(brick, bricks.EMPTY_BRICK)

	def test_black_core_is_empty_even_when_a_neighbour_holds_data(self):
		# A chunk absent from a sparse mirror, sitting right next to one that
		# was downloaded: only the padding sees the neighbour's voxels, and
		# uploading it would hide the coarser levels behind a black brick.
		core = bricks.BRICK_CORE
		voxels = np.zeros((core, core, 2 * core), dtype=np.uint8)
		voxels[:, :, :core] = 255
		brick = bricks.read_brick(VoxelArray(voxels), (1, 0, 0))
		self.assertIs(brick, bricks.EMPTY_BRICK)

	def test_a_single_lit_core_voxel_still_loads(self):
		core = bricks.BRICK_CORE
		voxels = np.zeros((core, core, 2 * core), dtype=np.uint8)
		voxels[0, 0, core] = 255
		brick = bricks.read_brick(VoxelArray(voxels), (1, 0, 0))
		self.assertIsNot(brick, bricks.EMPTY_BRICK)
		self.assertEqual(float(brick[bricks.BRICK_PAD, bricks.BRICK_PAD, bricks.BRICK_PAD]), 1.0)

	def test_every_level_takes_what_its_atlas_holds(self):
		# Twenty chunks in a row. Each level sees all of them on its own grid,
		# so none is left empty waiting for a finer one to overflow.
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
		self.assertEqual([len(selected[level][0]) for level in range(3)], [20, 10, 5])

	def test_levels_cover_overlapping_ground(self):
		# What the finest level holds must also be covered by the coarser ones,
		# or a chunk still in flight at L0 would have nothing behind it.
		chunks = np.indices((8, 8, 8)).reshape(3, -1).T
		selected = bricks.select_lod_chunks(
			chunks,
			{0: (8, 8, 8), 1: (4, 4, 4), 2: (2, 2, 2)},
			(0, 0, 0),
			slot_counts=(8, 64, 8),
		)
		for level in (1, 2):
			parents = {tuple(coord) for coord in selected[level][1]}
			for coord in selected[0][1]:
				self.assertIn(tuple(coord >> level), parents)

	def test_each_level_keeps_its_selection_distinct(self):
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

	def test_levels_collapse_by_their_own_distance_from_level_zero(self):
		# Five chunks in a row. On a grid two levels coarser they fall into two
		# chunks, and four levels coarser into one, whatever the levels in
		# between hold.
		chunks = np.array([[i, 0, 0] for i in range(5)])
		selected = bricks.select_lod_chunks(
			chunks,
			{0: (32, 32, 32), 2: (8, 8, 8), 4: (2, 2, 2)},
			(0, 0, 0),
			slot_counts=(1, 5, 5),
		)
		self.assertEqual([len(selected[level][0]) for level in (0, 2, 4)], [1, 2, 1])
		np.testing.assert_array_equal(
			np.sort(selected[2][1], axis=0), [[0, 0, 0], [1, 0, 0]]
		)

	def test_levels_need_not_start_at_zero(self):
		# The caller always hands over level 0 chunks, so every level shifts
		# them onto its own grid first.
		chunks = np.array([[0, 0, 0], [4, 0, 0], [8, 0, 0]])
		selected = bricks.select_lod_chunks(
			chunks,
			{2: (16, 16, 16), 3: (8, 8, 8)},
			(0, 0, 0),
			slot_counts=(2, 5),
		)
		np.testing.assert_array_equal(selected[2][1], [[0, 0, 0], [1, 0, 0]])
		np.testing.assert_array_equal(selected[3][1], [[0, 0, 0], [1, 0, 0]])

	def test_next_level_walks_the_configured_levels(self):
		levels = bricks.LEVELS
		self.assertEqual(
			[bricks.next_level(level) for level in levels],
			list(levels[1:]) + [None],
		)

	def test_default_memory_budgets_derive_expected_atlas_sizes(self):
		self.assertEqual(
			bricks.SLOTS_PER_AXIS, {0: 12, 1: 10, 2: 8, 3: 6, 4: 4, 5: 6}
		)
		for level, slots in bricks.SLOTS_PER_AXIS.items():
			budget = bricks.ATLAS_MEMORY_MB[level]
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
