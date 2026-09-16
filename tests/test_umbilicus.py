import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
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
spec = importlib.util.spec_from_file_location(PACKAGE + ".umbilicus", ROOT / "umbilicus.py")
umbilicus = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = umbilicus
spec.loader.exec_module(umbilicus)


def write(name, text):
	"""`text` in a file called `name`, in a directory that outlives the test."""
	directory = tempfile.mkdtemp()
	path = Path(directory) / name
	path.write_text(text)
	return str(path)


def write_json(name, document):
	return write(name, json.dumps(document))


class JsonPointsTest(unittest.TestCase):
	def test_objects_are_read_by_their_keys(self):
		points = umbilicus.json_points(
			{"control_points": [{"x": 1, "y": 2, "z": 3, "score": 100}]}
		)
		self.assertEqual(points, [[1.0, 2.0, 3.0]])

	def test_arrays_are_read_as_z_y_x(self):
		# The opposite order from the objects above, which is what VC writes.
		self.assertEqual(umbilicus.json_points([[3, 2, 1]]), [[1.0, 2.0, 3.0]])

	def test_either_key_holds_the_points(self):
		for key in umbilicus.POINT_KEYS:
			self.assertEqual(umbilicus.json_points({key: [[3, 2, 1]]}), [[1.0, 2.0, 3.0]])

	def test_a_fourth_value_is_ignored(self):
		self.assertEqual(umbilicus.json_points([[3, 2, 1, 100]]), [[1.0, 2.0, 3.0]])

	def test_a_document_holding_no_points_is_refused(self):
		with self.assertRaises(ValueError):
			umbilicus.json_points({"metadata": {}})

	def test_a_point_missing_a_coordinate_is_refused(self):
		with self.assertRaises(ValueError):
			umbilicus.json_points([{"x": 1, "y": 2}])

	def test_a_coordinate_that_is_not_a_number_is_refused(self):
		for value in ("3", None, True, float("nan")):
			with self.assertRaises(ValueError):
				umbilicus.json_points([{"x": 1, "y": 2, "z": value}])


class TextPointsTest(unittest.TestCase):
	def test_columns_are_z_y_x(self):
		self.assertEqual(umbilicus.text_points(["3, 2, 1"]), [[1.0, 2.0, 3.0]])

	def test_blanks_and_comments_are_skipped(self):
		lines = ["# an umbilicus", "", "3, 2, 1", "  ", "6, 5, 4"]
		self.assertEqual(
			umbilicus.text_points(lines), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
		)

	def test_a_short_line_is_refused(self):
		with self.assertRaises(ValueError):
			umbilicus.text_points(["3, 2"])


class ArraysTest(unittest.TestCase):
	def test_points_are_sorted_up_the_scroll(self):
		curve, edges = umbilicus.umbilicus_arrays([[0, 0, 9], [0, 0, 1], [0, 0, 5]])
		self.assertEqual(list(curve[:, 2]), [1.0, 5.0, 9.0])
		self.assertEqual(edges.tolist(), [[0, 1], [1, 2]])

	def test_a_single_point_has_no_edges(self):
		curve, edges = umbilicus.umbilicus_arrays([[1, 2, 3]])
		self.assertEqual(len(curve), 1)
		self.assertEqual(edges.shape, (0, 2))

	def test_the_edges_index_the_points(self):
		curve, edges = umbilicus.umbilicus_arrays([[0, 0, z] for z in range(5)])
		self.assertTrue((edges < len(curve)).all())
		self.assertEqual(len(edges), len(curve) - 1)


class ReadTest(unittest.TestCase):
	def test_the_format_the_herculaneum_umbilici_repo_publishes(self):
		path = write_json(
			"PHerc1203_umbilicus.json",
			{
				"control_points": [
					{"x": 3452, "y": 2992, "z": 2320, "score": 100},
					{"x": 3552, "y": 3040, "z": 2576, "score": 100},
				],
				"metadata": {"total_points": 2, "source_volume": "PHerc1203/volumes/x.zarr"},
			},
		)
		curve = umbilicus.read_umbilicus(path)
		self.assertEqual(curve.name, "PHerc1203_umbilicus")
		np.testing.assert_allclose(curve.positions[0], [3452.0, 2992.0, 2320.0])
		self.assertEqual(curve.edges.tolist(), [[0, 1]])
		self.assertEqual(curve.scores.tolist(), [100.0, 100.0])
		# Nothing in that metadata pins the frame the coordinates are in.
		self.assertIsNone(curve.voxel_size_um)
		self.assertIsNone(curve.volume_shape)

	def test_a_text_file_is_read_by_its_extension(self):
		curve = umbilicus.read_umbilicus(write("umbilicus.txt", "3, 2, 1\n6, 5, 4\n"))
		np.testing.assert_allclose(curve.positions, [[1, 2, 3], [4, 5, 6]])

	def test_a_stated_frame_is_read(self):
		path = write_json(
			"umbilicus.json",
			{
				"points": [[1, 2, 3]],
				"metadata": {
					"voxelsize_um": 7.91,
					"volume_width": 8096,
					"volume_height": 7888,
					"volume_slices": 14376,
				},
			},
		)
		curve = umbilicus.read_umbilicus(path)
		self.assertAlmostEqual(curve.voxel_size_um, 7.91)
		self.assertEqual(curve.volume_shape, (8096.0, 7888.0, 14376.0))

	def test_an_incomplete_triplet_states_no_shape(self):
		path = write_json(
			"umbilicus.json",
			{"points": [[1, 2, 3]], "metadata": {"volume_width": 8096}},
		)
		self.assertIsNone(umbilicus.read_umbilicus(path).volume_shape)

	def test_a_nonsense_voxel_size_states_nothing(self):
		for value in (0, -1, "7.91", None):
			path = write_json(
				"umbilicus.json",
				{"points": [[1, 2, 3]], "metadata": {"voxelsize_um": value}},
			)
			self.assertIsNone(umbilicus.read_umbilicus(path).voxel_size_um)

	def test_a_project_umbilicus_is_named_after_its_directory(self):
		path = write_json("umbilicus.json", [[1, 2, 3]])
		directory = Path(path).parent.name
		self.assertEqual(umbilicus.read_umbilicus(path).name, "%s umbilicus" % directory)

	def test_an_empty_file_is_refused(self):
		with self.assertRaises(ValueError):
			umbilicus.read_umbilicus(write_json("umbilicus.json", []))


class ExportTest(unittest.TestCase):
	def test_export_metadata_replaces_source_and_all_timestamps(self):
		now = datetime(2026, 9, 16, 14, 51, 5, tzinfo=timezone.utc)
		result = umbilicus.export_metadata(
			{"source_volume": "old", "timestamp": "old", "other": 42},
			"PHerc1203/volumes/scan.zarr", now,
		)
		stamp = "2026-09-16T14:51:05Z"
		self.assertEqual([result[key] for key in ("timestamp", "created", "modified")],
			[stamp, stamp, stamp])
		self.assertEqual(result["source_volume"], "PHerc1203/volumes/scan.zarr")
		self.assertEqual(result["annotator_note"], "exported from blender using velend")
		self.assertEqual(result["other"], 42)
		with self.assertRaisesRegex(ValueError, "scene volume"):
			umbilicus.export_metadata({}, "", now)

	def test_line_order_comes_from_edges_even_when_vertex_indices_are_scrambled(self):
		points = [[30, 0, 3], [10, 0, 1], [20, 0, 2]]
		edges = [[0, 2], [1, 2]]
		ordered = umbilicus.ordered_polyline(points, edges)
		self.assertEqual(ordered[:, 0].tolist(), [10, 20, 30])

	def test_branch_cycle_disconnected_and_loose_vertex_are_refused(self):
		cases = [
			(4, [[0, 1], [1, 2], [1, 3]]),
			(4, [[0, 1], [1, 2], [2, 0]]),
			(5, [[0, 1], [1, 2], [2, 0], [3, 4]]),
			(3, [[0, 1]]),
		]
		for count, edges in cases:
			with self.subTest(edges=edges), self.assertRaises(ValueError):
				umbilicus.ordered_polyline([[0, 0, z] for z in range(count)], edges)

	def test_a_line_that_doubles_back_in_z_is_refused(self):
		with self.assertRaisesRegex(ValueError, "doubles back"):
			umbilicus.ordered_polyline(
				[[0, 0, 1], [0, 0, 3], [0, 0, 2]], [[0, 1], [1, 2]]
			)

	def test_tiny_z_roundoff_at_equal_height_keeps_edge_order(self):
		points = umbilicus.ordered_polyline(
			[[0, 0, 1], [1, 0, 1 - 1e-6], [2, 0, 2]], [[0, 1], [1, 2]]
		)
		self.assertEqual(points[:, 0].tolist(), [0, 1, 2])
		self.assertEqual(points[:2, 2].tolist(), [1, 1])

	def test_written_json_round_trips_and_keeps_metadata(self):
		path = Path(tempfile.mkdtemp()) / "umbilicus.json"
		umbilicus.write_umbilicus(
			path, [[30, 0, 3], [10, 0, 1], [20, 0, 2]], [[0, 2], [1, 2]],
			7.91, {"source_volume": "scan.zarr", "total_points": 99},
			scores=[30, 10, 20],
		)
		document = json.loads(path.read_text())
		self.assertEqual(document["control_points"], [
			{"x": 10.0, "y": 0.0, "z": 1.0, "score": 10},
			{"x": 20.0, "y": 0.0, "z": 2.0, "score": 20},
			{"x": 30.0, "y": 0.0, "z": 3.0, "score": 30},
		])
		self.assertEqual(document["metadata"], {
			"source_volume": "scan.zarr", "total_points": 3, "voxelsize_um": 7.91,
		})
		curve = umbilicus.read_umbilicus(path)
		self.assertEqual(curve.edges.tolist(), [[0, 1], [1, 2]])
		self.assertEqual(curve.positions[:, 0].tolist(), [10, 20, 30])
		self.assertEqual(curve.scores.tolist(), [10.0, 20.0, 30.0])

	def test_new_mesh_writes_no_unearned_score(self):
		path = Path(tempfile.mkdtemp()) / "umbilicus.json"
		umbilicus.write_umbilicus(path, [[1, 2, 3], [4, 5, 6]], [[0, 1]], 9.362)
		document = json.loads(path.read_text())
		self.assertEqual(document["control_points"], [
			{"x": 1.0, "y": 2.0, "z": 3.0},
			{"x": 4.0, "y": 5.0, "z": 6.0},
		])
		self.assertEqual(document["metadata"]["total_points"], 2)

	def test_invalid_metadata_does_not_replace_an_existing_file(self):
		path = Path(tempfile.mkdtemp()) / "umbilicus.json"
		path.write_text("old file")
		with self.assertRaises(TypeError):
			umbilicus.write_umbilicus(
				path, [[0, 0, 1], [0, 0, 2]], [[0, 1]], 7.91, {"bad": object()}
			)
		self.assertEqual(path.read_text(), "old file")


if __name__ == "__main__":
	unittest.main()
