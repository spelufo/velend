"""Reading and writing umbilicus files: the scroll's axis through the volume.

An umbilicus marks the centre of the spiral on each slice, so a handful of
control points up the scroll say where its core runs. The file is a list of
those points in full resolution voxels, either as JSON

	{"control_points": [{"x": 3452, "y": 2992, "z": 2320, "score": 100}, ...],
	 "metadata": {...}}

or as text, one `z, y, x` line per point. Both shapes turn up in the wild, and
`metadata` is optional: only when it states a voxel size or the dimensions of
the volume the points were placed in does the file pin its own frame.

	https://github.com/ScrollPrize/villa
	volume-cartographer/core/src/Umbilicus.cpp, LoadJsonFile and LoadTextFile
	https://github.com/AlexeyDrobkovStrikesBack/herculaneum-umbilici

Nothing here imports bpy, so all of it can be tested outside Blender.
The operators turn these points into a mesh of vertices joined by edges and back.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np


# Where a JSON file's points can sit. VC accepts either name, and a document
# that is itself the array.
POINT_KEYS = ("points", "control_points")

# The extensions read as one `z, y, x` line per point rather than as JSON.
TEXT_SUFFIXES = (".txt", ".csv")


def _number(value, what):
	"""One coordinate, as a float, refusing what is not a plain number.

	`bool` is an `int` in Python, and `float("nan")` is a float, so neither the
	type check nor the finiteness check can be left out.
	"""
	if isinstance(value, bool) or not isinstance(value, (int, float)):
		raise ValueError("%s is not a number: %r" % (what, value))
	value = float(value)
	if not np.isfinite(value):
		raise ValueError("%s is not finite: %r" % (what, value))
	return value


def json_points(document):
	"""The (x, y, z) points of a parsed umbilicus JSON document.

	An entry is either an object keyed by x, y and z, or a bare array, which VC
	writes and reads as [z, y, x] — the opposite order, so the two cannot be
	told apart by their numbers and the shape of the entry is what decides.
	"""
	entries = None
	if isinstance(document, list):
		entries = document
	elif isinstance(document, dict):
		for key in POINT_KEYS:
			if key in document:
				entries = document[key]
				if not isinstance(entries, list):
					raise ValueError("'%s' is not an array" % key)
				break
	if entries is None:
		raise ValueError(
			"root is not an array and holds no %s" % " or ".join(POINT_KEYS)
		)

	points = []
	for index, entry in enumerate(entries):
		where = "point %d" % index
		if isinstance(entry, dict):
			missing = [key for key in "xyz" if key not in entry]
			if missing:
				raise ValueError("%s has no %s" % (where, ", ".join(missing)))
			point = [_number(entry[key], "%s %s" % (where, key)) for key in "xyz"]
		elif isinstance(entry, (list, tuple)):
			if len(entry) < 3:
				raise ValueError("%s has fewer than three values" % where)
			z, y, x = (_number(entry[i], "%s [%d]" % (where, i)) for i in range(3))
			point = [x, y, z]
		else:
			raise ValueError("%s is neither an object nor an array" % where)
		points.append(point)
	return points


def text_points(lines):
	"""The (x, y, z) points of a text umbilicus, whose columns are z, y and x.

	Blank lines and `#` comments are skipped, the way VC reads them.
	"""
	points = []
	for number, line in enumerate(lines, start=1):
		line = line.strip()
		if not line or line.startswith("#"):
			continue
		fields = [field.strip() for field in line.split(",")]
		fields = [field for field in fields if field]
		if len(fields) != 3:
			raise ValueError("line %d does not hold three values" % number)
		try:
			z, y, x = (float(field) for field in fields)
		except ValueError:
			raise ValueError("line %d holds something that is not a number" % number)
		points.append([x, y, z])
	return points


def json_metadata(document):
	"""A JSON document's `metadata` object, or an empty one without it.

	A file that states nothing is as ordinary as one that does, so a missing or
	wrong-typed block is read as saying nothing rather than refused.
	"""
	if not isinstance(document, dict):
		return {}
	metadata = document.get("metadata")
	return metadata if isinstance(metadata, dict) else {}


def _positive(value):
	"""`value` as a float when it is a positive number, else None."""
	if isinstance(value, bool) or not isinstance(value, (int, float)):
		return None
	value = float(value)
	return value if np.isfinite(value) and value > 0.0 else None


class Umbilicus:
	"""An umbilicus file as the arrays a mesh is built from.

	`positions` are in voxels, one row per vertex, sorted up the scroll; `edges`
	join each to the next. `metadata` is what the file said about itself, as it
	was written. `scores` holds per-point confidence when every source point has it.
	"""

	def __init__(self, path, metadata, positions, edges, scores=None):
		self.path = path
		self.metadata = metadata
		self.positions = positions
		self.edges = edges
		self.scores = scores

	@property
	def name(self):
		"""What to call the object, from the file's own name: an
		`umbilicus.json` sitting in a project directory is named after the
		directory, and a `PHerc1203_umbilicus.json` after itself."""
		stem = os.path.splitext(os.path.basename(self.path))[0]
		if stem in ("umbilicus", "estimated_umbilicus"):
			parent = os.path.basename(os.path.dirname(os.path.abspath(self.path)))
			if parent:
				return "%s %s" % (parent, stem.replace("_", " "))
		return stem

	@property
	def voxel_size_um(self):
		"""The voxel size the file says its coordinates are in, or None.

		Only `voxelsize_um` is a statement about the frame; `source_volume` and
		the like name where the points came from without pinning their scale.
		"""
		return _positive(self.metadata.get("voxelsize_um"))

	@property
	def volume_shape(self):
		"""The (x, y, z) voxel counts the file says it was placed in, or None.

		Two of the three dimensions cannot describe a grid, so a partial
		triplet says nothing, the same way VC refuses it.
		"""
		shape = tuple(
			_positive(self.metadata.get(key))
			for key in ("volume_width", "volume_height", "volume_slices")
		)
		return shape if all(size is not None for size in shape) else None


def umbilicus_arrays(points):
	"""The positions and edges of a polyline through the control points.

	Sorted by z, as VC sorts them before interpolating between them: the file's
	own order is not guaranteed to run up the scroll, and an unsorted polyline
	would zigzag between the same points.
	"""
	positions = np.asarray(points, dtype=np.float32).reshape(-1, 3)
	positions = positions[np.argsort(positions[:, 2], kind="stable")]
	starts = np.arange(len(positions) - 1, dtype=np.int32)
	edges = np.stack((starts, starts + 1), axis=1)
	return positions, edges


def read_umbilicus(path):
	"""Read an umbilicus file into the arrays a mesh is built from."""
	metadata = {}
	scores = None
	if os.path.splitext(path)[1].lower() in TEXT_SUFFIXES:
		with open(path) as file:
			points = text_points(file)
	else:
		with open(path) as file:
			document = json.load(file)
		points = json_points(document)
		metadata = json_metadata(document)
		entries = document if isinstance(document, list) else next(
			(document[key] for key in POINT_KEYS if key in document), []
		)
		values = [entry.get("score") if isinstance(entry, dict) else None for entry in entries]
		if values and all(
			isinstance(value, (int, float)) and not isinstance(value, bool)
			and np.isfinite(value) for value in values
		):
			order = np.argsort([point[2] for point in points], kind="stable")
			scores = np.asarray(values, dtype=np.float32)[order]
	if not points:
		raise ValueError("no points")
	positions, edges = umbilicus_arrays(points)
	return Umbilicus(path, metadata, positions, edges, scores)


def _ordered_polyline(points, edges):
	"""Return a single edge-connected line's points and original vertex indices.

	VC sorts control points by z when reading them, so a line that doubles back
	in z cannot be written without changing its connections on the next import.
	"""
	points = np.asarray(points, dtype=np.float64)
	edges = np.asarray(edges)
	if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
		raise ValueError("the mesh needs at least two vertices")
	if not np.isfinite(points).all():
		raise ValueError("vertex coordinates must be finite")
	if edges.shape != (len(points) - 1, 2) or not np.issubdtype(edges.dtype, np.integer):
		raise ValueError("the mesh must have one edge between each pair of line vertices")
	if ((edges < 0) | (edges >= len(points))).any():
		raise ValueError("an edge refers to a vertex outside the mesh")

	neighbors = [[] for _ in points]
	for a, b in edges:
		if a == b or b in neighbors[a]:
			raise ValueError("the line has a repeated or self-connected edge")
		neighbors[a].append(b)
		neighbors[b].append(a)
	if any(len(adjacent) not in (1, 2) for adjacent in neighbors):
		raise ValueError("the mesh must be one line without branches or loose vertices")
	ends = [index for index, adjacent in enumerate(neighbors) if len(adjacent) == 1]
	if len(ends) != 2:
		raise ValueError("the mesh must be an open line")
	start = min(ends, key=lambda index: (points[index, 2], index))
	ordered = [start]
	previous = -1
	while True:
		following = [index for index in neighbors[ordered[-1]] if index != previous]
		if not following:
			break
		previous = ordered[-1]
		ordered.append(following[0])
	if len(ordered) != len(points):
		raise ValueError("the mesh contains disconnected lines or a cycle")
	result = points[ordered]
	if (np.diff(result[:, 2]) < -0.01).any():
		raise ValueError("the line doubles back in z; umbilicus files are sorted by z")
	# Inverting the import placement can leave tiny z reversals at equal-height
	# points. Keep their edge order stable under VC's z sort.
	result[:, 2] = np.maximum.accumulate(result[:, 2])
	return result, ordered


def ordered_polyline(points, edges):
	"""Return a single edge-connected line's points, from low to high z."""
	return _ordered_polyline(points, edges)[0]


def export_metadata(metadata, source_volume, now=None):
	"""Update provenance fields for one export, using one UTC instant."""
	if not source_volume:
		raise ValueError("set a scene volume before exporting an umbilicus")
	now = now or datetime.now(timezone.utc)
	stamp = now.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
	result = dict(metadata or {})
	result.update({
		"timestamp": stamp,
		"created": stamp,
		"modified": stamp,
		"source_volume": source_volume,
		"annotator_note": "exported from blender using velend",
	})
	return result


def write_umbilicus(path, points, edges, voxel_size, metadata=None, scores=None):
	"""Write a connected mesh line as VC-compatible JSON in full-resolution voxels."""
	ordered, indices = _ordered_polyline(points, edges)
	voxel_size = _positive(voxel_size)
	if voxel_size is None:
		raise ValueError("voxel size must be positive and finite")
	if scores is not None:
		scores = np.asarray(scores, dtype=np.float64)
		if scores.shape != (len(ordered),) or not np.isfinite(scores).all():
			raise ValueError("scores must be one finite number per vertex")
	metadata = dict(metadata or {})
	metadata["voxelsize_um"] = voxel_size
	metadata["total_points"] = len(ordered)
	control_points = []
	for row, vertex_index in zip(ordered, indices):
		point = dict(zip("xyz", row.tolist()))
		if scores is not None:
			score = float(scores[vertex_index])
			point["score"] = int(score) if score.is_integer() else score
		control_points.append(point)
	document = {"control_points": control_points, "metadata": metadata}
	contents = json.dumps(document, indent=2, allow_nan=False) + "\n"
	with open(path, "w") as file:
		file.write(contents)
