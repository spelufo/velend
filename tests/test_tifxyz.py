import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "velend_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules.setdefault(PACKAGE, package)
spec = importlib.util.spec_from_file_location(PACKAGE + ".tifxyz", ROOT / "tifxyz.py")
tifxyz = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tifxyz
spec.loader.exec_module(tifxyz)

try:
	import tifffile
	import imagecodecs
except ImportError:
	# Both are wheels the extension bundles, so Blender's Python has them and a
	# bare `python3 tests/...` may not. What needs them skips.
	tifffile = None

# The surfaces volume-cartographer's own tests are built against, if the villa
# checkout the extension is developed alongside is there.
VILLA_SEGMENT = (
	ROOT / "villa" / "volume-cartographer" / "core" / "test" / "data"
	/ "segments" / "20241113070770"
)


def grid(height, width):
	"""A grid of distinguishable, valid positions."""
	rows, cols = np.mgrid[0:height, 0:width]
	x = (100.0 + cols).astype(np.float32)
	y = (200.0 + rows).astype(np.float32)
	z = np.full((height, width), 300.0, dtype=np.float32)
	return x, y, z


class ValidityTest(unittest.TestCase):
	def test_the_unwritten_point_is_invalid(self):
		x, y, z = grid(2, 2)
		x[0, 0], y[0, 0], z[0, 0] = -1.0, -1.0, -1.0
		valid = tifxyz.valid_points(x, y, z)
		self.assertFalse(valid[0, 0])
		self.assertTrue(valid[1, 1])

	def test_points_at_or_behind_the_volume_front_are_invalid(self):
		# VC drops z <= 0 outright, not just the -1 sentinel.
		x, y, z = grid(1, 3)
		z[0, 0] = 0.0
		z[0, 1] = -0.5
		self.assertEqual(list(tifxyz.valid_points(x, y, z)[0]), [False, False, True])

	def test_nan_is_invalid(self):
		x, y, z = grid(1, 2)
		y[0, 0] = np.nan
		self.assertEqual(list(tifxyz.valid_points(x, y, z)[0]), [False, True])


class MaskTest(unittest.TestCase):
	def test_only_a_full_mask_pixel_keeps_its_point(self):
		mask = np.array([[255, 254], [0, 255]], dtype=np.uint8)
		reduced = tifxyz.reduce_mask(mask, (2, 2))
		self.assertEqual(reduced.tolist(), [[True, False], [False, True]])

	def test_a_finer_mask_covers_a_block_per_point(self):
		# Twice the grid in each direction: four mask pixels per grid point, and
		# any one of them being unset drops the point.
		mask = np.full((4, 4), 255, dtype=np.uint8)
		mask[3, 3] = 0
		reduced = tifxyz.reduce_mask(mask, (2, 2))
		self.assertEqual(reduced.tolist(), [[True, True], [True, False]])

	def test_a_mask_that_does_not_divide_the_grid_is_ignored(self):
		mask = np.full((3, 4), 255, dtype=np.uint8)
		self.assertIsNone(tifxyz.reduce_mask(mask, (2, 2)))


class QuadTest(unittest.TestCase):
	def test_a_full_grid_becomes_a_full_quad_mesh(self):
		valid = np.ones((3, 3), dtype=bool)
		used, quads = tifxyz.build_quads(valid)
		self.assertEqual(used.sum(), 9)
		self.assertEqual(len(quads), 4)
		# Wound around the grid cell, so the faces all face the same way.
		self.assertEqual(quads[0].tolist(), [0, 1, 4, 3])

	def test_an_invalid_point_drops_the_quads_that_need_it(self):
		valid = np.ones((3, 3), dtype=bool)
		valid[1, 1] = False
		used, quads = tifxyz.build_quads(valid)
		self.assertEqual(len(quads), 0)
		# With no quad left there is nothing to hold any vertex either.
		self.assertEqual(used.sum(), 0)

	def test_points_no_quad_uses_are_left_out(self):
		valid = np.ones((2, 3), dtype=bool)
		valid[1, 2] = False
		used, quads = tifxyz.build_quads(valid)
		self.assertEqual(len(quads), 1)
		# The far column has no complete quad, so its remaining point is not a
		# vertex; four of the six grid points are.
		self.assertEqual(used.sum(), 4)
		self.assertEqual(used[:, 2].tolist(), [False, False])

	def test_faces_index_the_vertices_that_were_kept(self):
		valid = np.ones((4, 4), dtype=bool)
		valid[0, 0] = False
		used, quads = tifxyz.build_quads(valid)
		self.assertTrue((quads >= 0).all())
		self.assertTrue((quads < used.sum()).all())
		self.assertEqual(len(set(map(tuple, quads.tolist()))), len(quads))


class SurfaceArraysTest(unittest.TestCase):
	def test_positions_uvs_and_channels_line_up(self):
		x, y, z = grid(3, 4)
		valid = np.ones((3, 4), dtype=bool)
		channel = np.arange(12, dtype=np.float32).reshape(3, 4)
		positions, quads, uvs, values = tifxyz.surface_arrays(
			x, y, z, valid, {"generations": channel}
		)
		self.assertEqual(positions.shape, (12, 3))
		self.assertEqual(len(quads), 6)
		# Vertices come out in row major order, so the first is the grid's
		# top left corner and the last its bottom right.
		self.assertEqual(positions[0].tolist(), [100.0, 200.0, 300.0])
		self.assertEqual(positions[-1].tolist(), [103.0, 202.0, 300.0])
		self.assertEqual(values["generations"].tolist(), list(range(12)))
		# U runs along the columns, V up the rows.
		self.assertEqual(uvs[0].tolist(), [0.0, 1.0])
		self.assertEqual(uvs[-1].tolist(), [1.0, 0.0])


def write_surface(directory, x, y, z, mask=None, channels=None, meta=None):
	"""A tifxyz surface on disk, written the way volume-cartographer does."""
	directory.mkdir(parents=True, exist_ok=True)
	for name, band in (("x", x), ("y", y), ("z", z)):
		tifffile.imwrite(directory / (name + ".tif"), band, compression=None)
	if mask is not None:
		# VC writes the mask tiled and LZW compressed, unlike the coordinates,
		# so this covers the tiled and the compressed paths too.
		tifffile.imwrite(
			directory / "mask.tif", mask, compression="lzw", tile=(16, 16)
		)
	for name, values in (channels or {}).items():
		tifffile.imwrite(directory / (name + ".tif"), values)
	full = {"format": "tifxyz", "type": "seg", "uuid": "test", "scale": [0.05, 0.05]}
	full.update(meta or {})
	(directory / "meta.json").write_text(json.dumps(full))
	return directory


@unittest.skipIf(tifffile is None, "tifffile and imagecodecs are not installed")
class ReadSurfaceTest(unittest.TestCase):
	def setUp(self):
		self.root = Path(tempfile.mkdtemp())
		self.addCleanup(shutil.rmtree, self.root)

	def test_reads_a_surface_into_mesh_arrays(self):
		x, y, z = grid(5, 5)
		path = write_surface(self.root / "seg", x, y, z)
		surface = tifxyz.read_surface(path)
		self.assertEqual(surface.shape, (5, 5))
		self.assertEqual(len(surface.positions), 25)
		self.assertEqual(len(surface.quads), 16)
		self.assertEqual(surface.uuid, "test")
		self.assertEqual(surface.scale, (0.05, 0.05))

	def test_applies_the_mask_and_reads_extra_channels(self):
		x, y, z = grid(4, 4)
		mask = np.full((8, 8), 255, dtype=np.uint8)
		mask[0, 0] = 0
		generations = np.arange(16, dtype=np.uint16).reshape(4, 4)
		path = write_surface(
			self.root / "seg", x, y, z, mask=mask,
			channels={"generations": generations},
		)
		surface = tifxyz.read_surface(path)
		# The one masked out corner takes its quad, and only that quad, away.
		self.assertEqual(len(surface.quads), 8)
		self.assertEqual(sorted(surface.channels), ["generations"])
		self.assertEqual(len(surface.channels["generations"]), len(surface.positions))

		ignored = tifxyz.read_surface(path, load_mask=False, load_channels=False)
		self.assertEqual(len(ignored.quads), 9)
		self.assertEqual(ignored.channels, {})

	def test_step_subsamples_the_grid_without_moving_it(self):
		x, y, z = grid(5, 5)
		path = write_surface(self.root / "seg", x, y, z)
		surface = tifxyz.read_surface(path, step=2)
		self.assertEqual(surface.shape, (3, 3))
		self.assertEqual(len(surface.quads), 4)
		# Same corners, four times fewer quads between them.
		self.assertEqual(surface.positions[0].tolist(), [100.0, 200.0, 300.0])
		self.assertEqual(surface.positions[-1].tolist(), [104.0, 204.0, 300.0])

	def test_a_directory_of_surfaces_is_found_from_its_parent(self):
		x, y, z = grid(3, 3)
		write_surface(self.root / "patches" / "a", x, y, z)
		write_surface(self.root / "patches" / "b", x, y, z)
		(self.root / "patches" / "not-a-surface").mkdir()
		found = tifxyz.surface_dirs(str(self.root / "patches"))
		self.assertEqual([Path(path).name for path in found], ["a", "b"])
		# A surface directory stands for itself.
		self.assertEqual(
			tifxyz.surface_dirs(str(self.root / "patches" / "a")),
			[str(self.root / "patches" / "a")],
		)

	def test_a_placeholder_without_tiffs_is_not_a_surface(self):
		placeholder = self.root / "lazy"
		placeholder.mkdir()
		(placeholder / "meta.json").write_text('{"format": "tifxyz"}')
		self.assertFalse(tifxyz.is_surface_dir(str(placeholder)))
		self.assertEqual(tifxyz.surface_dirs(str(placeholder)), [])


@unittest.skipIf(tifffile is None, "tifffile and imagecodecs are not installed")
@unittest.skipUnless(VILLA_SEGMENT.is_dir(), "villa checkout is not there")
class VillaFixtureTest(unittest.TestCase):
	"""Reads one of volume-cartographer's own test surfaces."""

	def test_reads_a_volume_cartographer_surface(self):
		surface = tifxyz.read_surface(str(VILLA_SEGMENT))
		self.assertEqual(surface.shape, (129, 129))
		self.assertEqual(surface.uuid, "20241113070770")
		self.assertTrue(len(surface.quads) > 0)
		self.assertEqual(len(surface.uvs), len(surface.positions))
		# Every position sits inside the bounding box the surface's meta claims.
		low, high = surface.meta["bbox"]
		self.assertTrue((surface.positions >= np.array(low) - 1e-3).all())
		self.assertTrue((surface.positions <= np.array(high) + 1e-3).all())


if __name__ == "__main__":
	unittest.main()
