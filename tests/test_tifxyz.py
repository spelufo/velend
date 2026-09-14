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
		# Tifxyz front-facing winding, with V following its rows.
		self.assertEqual(quads[0].tolist(), [0, 3, 4, 1])

	def test_faces_point_opposite_the_grid_coordinate_cross_product(self):
		x, y, z = grid(2, 2)
		positions, quads, _, _ = tifxyz.surface_arrays(
			x, y, z, np.ones((2, 2), dtype=bool), {}
		)
		quad = quads[0]
		normal = np.cross(positions[quad[1]] - positions[quad[0]],
			positions[quad[2]] - positions[quad[0]])
		self.assertLess(normal[2], 0.0)

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
		# U runs opposite the columns and V runs along the tifxyz rows.
		self.assertEqual(uvs[0].tolist(), [1.0, 0.0])
		self.assertEqual(uvs[-1].tolist(), [0.0, 1.0])


def write_surface(directory, x, y, z, mask=None, channels=None, meta=None):
	"""A tifxyz surface on disk, written the way volume-cartographer does."""
	directory.mkdir(parents=True, exist_ok=True)
	for name, band in (("x", x), ("y", y), ("z", z)):
		tifffile.imwrite(directory / (name + ".tif"), band, compression=None)
	if mask is not None:
		# Exercise Villa's tiled LZW form when the local codec supports writing it.
		try:
			tifffile.imwrite(
				directory / "mask.tif", mask, compression="lzw", tile=(16, 16)
			)
		except Exception:
			tifffile.imwrite(directory / "mask.tif", mask, compression=None)
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
		self.assertEqual(len(surface.quads), 8)
		self.assertEqual(len(surface.positions), 15)
		self.assertEqual(sorted(surface.channels), ["generations"])
		self.assertEqual(len(surface.channels["generations"]), len(surface.positions))

		without_channels = tifxyz.read_surface(path, load_channels=False)
		self.assertEqual(len(without_channels.quads), 8)
		self.assertEqual(without_channels.channels, {})

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


class UVGridTest(unittest.TestCase):
	def test_orders_a_grid_by_descending_u_and_ascending_v(self):
		uvs = np.array([
			[1, 0], [0, 1], [1, 1], [0, 0], [0.5, 1], [0.5, 0],
		], dtype=float)
		faces = [(1, 4, 5, 3), (4, 2, 0, 5)]
		grid_indices = tifxyz.grid_from_uvs(uvs, faces)
		self.assertEqual(grid_indices.tolist(), [[0, 5, 3], [2, 4, 1]])

	def test_rejects_a_missing_quad(self):
		uvs = np.array([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=float)
		with self.assertRaisesRegex(ValueError, "complete rectangular"):
			tifxyz.grid_from_uvs(uvs, [])

	def test_crops_to_the_remaining_uv_bounds(self):
		uvs = np.array([[0.2, 0.8], [0.5, 0.8], [0.2, 0.4], [0.5, 0.4]])
		partial = tifxyz.partial_grid_from_uvs(uvs, [(0, 1, 3, 2)])
		self.assertEqual(partial.tolist(), [[3, 2], [1, 0]])

	def test_keeps_holes_inside_the_uv_crop(self):
		uvs = np.array([
			[0, 1], [0.25, 1], [0.75, 1], [1, 1],
			[0, 0], [0.25, 0], [0.75, 0], [1, 0],
		])
		partial = tifxyz.partial_grid_from_uvs(
			uvs, [(0, 1, 5, 4), (2, 3, 7, 6)]
		)
		self.assertEqual(partial.tolist(), [
			[7, 6, -1, 5, 4], [3, 2, -1, 1, 0],
		])

	def test_ignores_orphan_vertices_without_uv_loops(self):
		uvs = np.array([
			[0, 1], [0.5, 1], [0, 0], [0.5, 0], [np.nan, np.nan],
		])
		partial = tifxyz.partial_grid_from_uvs(uvs, [(0, 1, 3, 2)])
		self.assertEqual(partial.tolist(), [[3, 2], [1, 0]])

	def test_reports_the_face_and_edge_that_do_not_follow_the_grid(self):
		uvs = np.array([[0, 0], [1, 0], [0.9, 1], [0, 1]], dtype=float)
		with self.assertRaises(tifxyz.UVGridError) as raised:
			tifxyz.partial_grid_from_uvs(uvs, [(0, 1, 2, 3)])
		self.assertEqual(raised.exception.vertices, (1, 2))
		self.assertEqual(raised.exception.faces, (0,))

	def test_missing_grid_values_get_the_invalid_point_sentinel(self):
		grid_indices = np.array([[0, 1, -1], [2, 3, -1]])
		points = np.arange(12, dtype=float).reshape(4, 3)
		completed = tifxyz.scatter_grid(points, grid_indices, (-1, -1, -1))
		self.assertEqual(completed[0, 2].tolist(), [-1.0, -1.0, -1.0])
		self.assertEqual(completed[1, 1].tolist(), points[3].tolist())


@unittest.skipIf(tifffile is None, "tifffile and imagecodecs are not installed")
class WriteSurfaceTest(unittest.TestCase):
	def setUp(self):
		self.root = Path(tempfile.mkdtemp())
		self.addCleanup(shutil.rmtree, self.root)

	def test_invalid_completed_samples_do_not_reimport_as_geometry(self):
		x, y, z = grid(2, 3)
		points = np.stack((x, y, z), axis=2)
		points[:, 2] = -1.0
		mask = np.ones((2, 3), dtype=np.float32)
		mask[:, 2] = 0.0
		tifxyz.write_surface(
			self.root / "completed", points, mask, {}, {}, "completed", (1, 1)
		)
		surface = tifxyz.read_surface(self.root / "completed")
		self.assertEqual(len(surface.positions), 4)
		self.assertEqual(len(surface.quads), 1)

	def test_writes_villa_mask_channels_and_metadata(self):
		x, y, z = grid(2, 3)
		points = np.stack((x, y, z), axis=2)
		mask = np.array([[1.0, 0.7, 0.49], [0.0, 1.0, 1.0]], dtype=np.float32)
		tifxyz.write_surface(
			self.root / "out", points, mask,
			{"score": np.arange(6, dtype=np.float32).reshape(2, 3)},
			{"note": "kept"}, "exported", (0.1, 0.2),
		)
		self.assertEqual(
			tifxyz.read_page(self.root / "out" / "mask.tif").tolist(),
			[[255, 255, 0], [0, 255, 255]],
		)
		meta = json.loads((self.root / "out" / "meta.json").read_text())
		self.assertEqual(meta["uuid"], "exported")
		self.assertEqual(meta["scale"], [0.1, 0.2])
		self.assertEqual(meta["tiff_dimensions"], [3, 2])
		self.assertEqual(meta["note"], "kept")
		self.assertEqual(
			tifxyz.read_page(self.root / "out" / "score.tif").shape, (2, 3)
		)


if __name__ == "__main__":
	unittest.main()
