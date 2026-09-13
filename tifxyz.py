"""Reading volume-cartographer's tifxyz surfaces.

A tifxyz surface is a directory holding a grid of points on a segmented sheet:
`x.tif`, `y.tif` and `z.tif` give every grid point's position in the volume, in
full resolution voxels, and `meta.json` describes the grid. Alongside them can
sit `mask.tif`, which takes points out of the surface, and any number of extra
single channel TIFFs like `generations.tif` carrying a value per grid point.

	https://github.com/ScrollPrize/villa
	volume-cartographer/core/src/QuadSurface.cpp, load_quad_from_tifxyz_impl

Nothing here imports bpy, so all of it can be tested outside Blender, and only
the file reading needs tifffile. `commands.velend_OT_import_tifxyz` turns what
`read_surface` returns into a mesh.
"""

import json
import os
import shutil
import tempfile

import numpy as np


# The files every surface has, and the ones whose meaning we know. Any other
# `*.tif` in the directory is an extra channel, a value per grid point.
COORDINATE_FILES = ("x.tif", "y.tif", "z.tif")
REQUIRED_FILES = COORDINATE_FILES + ("meta.json",)
RESERVED_FILES = frozenset(COORDINATE_FILES + ("mask.tif",))

# A mask pixel keeps its grid point only when it is fully set. VC uses 255 for
# every sample format, not the maximum of the type, so this is a plain number.
MASK_KEEP = 255


def is_surface_dir(path):
	"""Whether `path` is a tifxyz surface rather than something else.

	The open data catalogue also ships directories holding a `meta.json` and no
	TIFFs, standing in for a surface that has not been downloaded, so the
	coordinate files are what makes a surface loadable.
	"""
	return all(os.path.isfile(os.path.join(path, name)) for name in REQUIRED_FILES)


def surface_dirs(path):
	"""The surfaces to import for a chosen directory.

	Segments are kept as one directory each under a folder like `patches/`, so
	a directory that is not itself a surface is taken to be a folder of them.
	"""
	if is_surface_dir(path):
		return [path]
	if not os.path.isdir(path):
		return []
	return sorted(
		entry.path
		for entry in os.scandir(path)
		if entry.is_dir() and is_surface_dir(entry.path)
	)


def read_meta(directory):
	"""The surface's `meta.json`, as it is written."""
	with open(os.path.join(directory, "meta.json")) as file:
		return json.load(file)


def meta_scale(meta):
	"""Grid samples per voxel, along the grid's columns and rows.

	Written as [x, y] the way the surface is parameterized, which is the
	opposite order from the (row, column) the arrays are indexed by.
	"""
	scale = meta.get("scale")
	if not isinstance(scale, (list, tuple)) or len(scale) < 2:
		return (1.0, 1.0)
	return (float(scale[0]), float(scale[1]))


def read_page(path):
	"""The first page of a TIFF, as a 2D array.

	Masks are sometimes written as a multipage file whose later pages are
	renderings of the surface, and OpenCV writes some channels with several
	identical samples per pixel. The surface data is page 0, channel 0.
	"""
	# Imported here so that everything but the file reading works without the
	# wheels, which is what the tests exercise.
	import tifffile

	with tifffile.TiffFile(path) as tif:
		image = tif.pages[0].asarray()
	if image.ndim == 3:
		image = image[..., 0]
	if image.ndim != 2:
		raise ValueError("%s is not a 2D image" % os.path.basename(path))
	return image


def read_coordinates(directory):
	"""The (x, y, z) grids of a surface, in voxels, at their stored size."""
	bands = []
	for name in COORDINATE_FILES:
		band = np.asarray(read_page(os.path.join(directory, name)), dtype=np.float32)
		if bands and band.shape != bands[0].shape:
			raise ValueError("%s does not match the size of x.tif" % name)
		bands.append(band)
	return tuple(bands)


def valid_points(x, y, z):
	"""Which grid points hold a real position.

	Points that were never filled in are written as (-1, -1, -1), and VC drops
	anything at or behind the volume's front face as well.
	"""
	return np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 0.0) & (x != -1.0)


def reduce_mask(mask, shape):
	"""One boolean per grid point from a mask that may be finer than the grid.

	A mask is allowed to be an exact multiple of the grid in each direction, so
	that it can carve at a finer resolution than the grid is stored at; a grid
	point survives only if every mask pixel covering it is set. Anything that
	does not divide evenly is not a mask for this grid, and is ignored.
	"""
	height, width = shape
	factor_y, remainder_y = divmod(mask.shape[0], height)
	factor_x, remainder_x = divmod(mask.shape[1], width)
	if remainder_y or remainder_x or factor_y < 1 or factor_x < 1:
		return None
	blocks = mask.reshape(height, factor_y, width, factor_x)
	return blocks.min(axis=(1, 3)) >= MASK_KEEP


def read_mask(directory, shape):
	"""`mask.tif` reduced to one boolean per grid point, or None without one."""
	path = os.path.join(directory, "mask.tif")
	if not os.path.isfile(path):
		return None
	return reduce_mask(read_page(path), shape)


def channel_names(directory):
	"""The extra per grid point channels the surface carries."""
	return sorted(
		name[:-len(".tif")]
		for name in os.listdir(directory)
		if name.endswith(".tif") and name not in RESERVED_FILES
	)


def read_channel(directory, name, shape):
	"""One extra channel as floats, or None when it is not the grid's size."""
	channel = np.asarray(
		read_page(os.path.join(directory, name + ".tif")), dtype=np.float32
	)
	return channel if channel.shape == shape else None


def build_quads(valid):
	"""The vertices and faces of the grid's valid quads.

	Returns (used, quads): `used` picks the grid points that become vertices,
	in row major order, and `quads` holds the four vertex numbers of each face.
	A quad needs all four of its corners; a grid point that no quad uses would
	be a loose vertex along the surface's ragged edge, so it is left out.
	"""
	quad_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, 1:] & valid[1:, :-1]
	used = np.zeros(valid.shape, dtype=bool)
	used[:-1, :-1] |= quad_valid
	used[:-1, 1:] |= quad_valid
	used[1:, 1:] |= quad_valid
	used[1:, :-1] |= quad_valid
	# Vertices are numbered in the order `used` walks the grid, which is the
	# order the coordinates are pulled out in below.
	indices = np.cumsum(used.ravel(), dtype=np.int32).reshape(valid.shape) - 1
	rows, cols = np.nonzero(quad_valid)
	quads = np.empty((len(rows), 4), dtype=np.int32)
	quads[:, 0] = indices[rows, cols]
	quads[:, 1] = indices[rows, cols + 1]
	quads[:, 2] = indices[rows + 1, cols + 1]
	quads[:, 3] = indices[rows + 1, cols]
	return used, quads


class Surface:
	"""A tifxyz surface as the arrays a mesh is built from.

	`positions` are in voxels, one row per vertex; `quads` index into them;
	`uvs` place each vertex on the grid, in 0..1; `channels` holds one value per
	vertex for each extra channel that was read.
	"""

	def __init__(self, path, meta, shape, positions, quads, uvs, channels):
		self.path = path
		self.meta = meta
		self.shape = shape
		self.positions = positions
		self.quads = quads
		self.uvs = uvs
		self.channels = channels

	@property
	def uuid(self):
		return str(self.meta.get("uuid") or os.path.basename(os.path.normpath(self.path)))

	@property
	def scale(self):
		return meta_scale(self.meta)


def surface_arrays(x, y, z, valid, channels):
	"""Build mesh arrays for the valid part of a coordinate grid."""
	used, quads = build_quads(valid)
	positions = np.stack((x[used], y[used], z[used]), axis=1).astype(np.float32)
	rows, cols = np.nonzero(used)
	height, width = valid.shape
	uvs = np.empty((len(rows), 2), dtype=np.float32)
	uvs[:, 0] = cols / max(width - 1, 1)
	# Row 0 is the top of the grid, and V runs up the image in Blender.
	uvs[:, 1] = 1.0 - rows / max(height - 1, 1)
	values = {name: channel[used] for name, channel in channels.items()}
	return positions, quads, uvs, values


def read_surface(directory, step=1, load_channels=True):
	"""Read a tifxyz surface directory into the arrays a mesh is built from.

	`step` keeps every nth row and column of the grid, which is how a segment
	whose grid is millions of quads is brought down to something a viewport can
	live with.
	"""
	meta = read_meta(directory)
	x, y, z = read_coordinates(directory)
	valid = valid_points(x, y, z)
	loaded_mask = read_mask(directory, valid.shape)
	if loaded_mask is not None:
		valid &= loaded_mask
	channels = {}
	if load_channels:
		for name in channel_names(directory):
			channel = read_channel(directory, name, valid.shape)
			if channel is not None:
				channels[name] = channel
	if step > 1:
		sample = (slice(None, None, step), slice(None, None, step))
		x, y, z, valid = x[sample], y[sample], z[sample], valid[sample]
		channels = {name: channel[sample] for name, channel in channels.items()}
	positions, quads, uvs, values = surface_arrays(x, y, z, valid, channels)
	return Surface(
		directory, meta, valid.shape, positions, quads, uvs, values
	)


def _levels(values, tolerance=1e-5):
	"""Cluster nearly equal UV coordinates and return levels and indices."""
	order = np.argsort(values)
	levels = []
	indices = np.empty(len(values), dtype=np.int32)
	for vertex in order:
		value = float(values[vertex])
		if not levels or abs(value - levels[-1]) > tolerance:
			levels.append(value)
		indices[vertex] = len(levels) - 1
	return np.asarray(levels), indices


def grid_from_uvs(vertex_uvs, faces, tolerance=1e-5):
	"""Return row-major indices for a complete rectangular UV quad grid."""
	uvs = np.asarray(vertex_uvs, dtype=np.float64)
	if uvs.ndim != 2 or uvs.shape[1] != 2 or len(uvs) < 4:
		raise ValueError("the active UV map does not define a grid")
	u_levels, cols = _levels(uvs[:, 0], tolerance)
	v_levels, bottom_rows = _levels(uvs[:, 1], tolerance)
	height, width = len(v_levels), len(u_levels)
	if height < 2 or width < 2 or height * width != len(uvs):
		raise ValueError("UV vertices do not form a complete rectangular lattice")
	rows = height - 1 - bottom_rows
	grid = np.full((height, width), -1, dtype=np.int32)
	for vertex, (row, col) in enumerate(zip(rows, cols)):
		if grid[row, col] != -1:
			raise ValueError("more than one vertex occupies a UV grid point")
		grid[row, col] = vertex
	if (grid < 0).any():
		raise ValueError("the UV grid has missing vertices")

	actual = set()
	for face in faces:
		if len(face) != 4:
			raise ValueError("the mesh contains a non-quad face")
		cells = {(int(rows[v]), int(cols[v])) for v in face}
		face_rows = {cell[0] for cell in cells}
		face_cols = {cell[1] for cell in cells}
		if len(cells) != 4 or len(face_rows) != 2 or len(face_cols) != 2:
			raise ValueError("a face does not follow the UV grid")
		r0, r1 = min(face_rows), max(face_rows)
		c0, c1 = min(face_cols), max(face_cols)
		if r1 != r0 + 1 or c1 != c0 + 1:
			raise ValueError("a face skips a row or column in the UV grid")
		actual.add((r0, c0))
	expected = {(row, col) for row in range(height - 1) for col in range(width - 1)}
	if actual != expected or len(faces) != len(expected):
		raise ValueError("the mesh is not a complete rectangular quad grid")
	return grid


def partial_grid_from_uvs(vertex_uvs, faces, tolerance=1e-5):
	"""Map surviving face vertices to their smallest rectangular UV crop.

	Vertices without face loops have no UV in Blender and are ignored. Holes
	inside the crop remain -1, so callers can export them as masked samples.
	"""
	uvs = np.asarray(vertex_uvs, dtype=np.float64)
	if uvs.ndim != 2 or uvs.shape[1] != 2:
		raise ValueError("the active UV map does not define a grid")
	mapped = np.isfinite(uvs).all(axis=1)
	if mapped.sum() < 4:
		raise ValueError("too few vertices remain in the active UV map")

	u_steps = []
	v_steps = []
	for original_face in faces:
		face = tuple(original_face)
		if len(face) != 4:
			raise ValueError("the mesh contains a non-quad face")
		if any(vertex < 0 or vertex >= len(uvs) or not mapped[vertex] for vertex in face):
			raise ValueError("a face has a vertex without an active UV coordinate")
		for first, second in zip(face, face[1:] + face[:1]):
			delta = np.abs(uvs[first] - uvs[second])
			if delta[0] > tolerance and delta[1] <= tolerance:
				u_steps.append(float(delta[0]))
			elif delta[1] > tolerance and delta[0] <= tolerance:
				v_steps.append(float(delta[1]))
			else:
				raise ValueError("a face edge does not follow the UV grid")
	if not u_steps or not v_steps:
		raise ValueError("the remaining faces do not establish a rectangular UV spacing")

	u_step = min(u_steps)
	v_step = min(v_steps)
	u_min, v_min = uvs[mapped].min(axis=0)
	u_max, v_max = uvs[mapped].max(axis=0)
	width = int(round((u_max - u_min) / u_step)) + 1
	height = int(round((v_max - v_min) / v_step)) + 1
	if width < 2 or height < 2:
		raise ValueError("the inferred UV grid is too small")

	vertices = np.flatnonzero(mapped)
	cols_float = (uvs[vertices, 0] - u_min) / u_step
	rows_float = (v_max - uvs[vertices, 1]) / v_step
	cols = np.rint(cols_float).astype(np.int32)
	rows = np.rint(rows_float).astype(np.int32)
	# Blender stores UVs as float32. Allow their error to grow slightly when
	# converted into grid-index units, but stay well below half a cell.
	index_tolerance = 0.1
	if (
		(np.abs(cols_float - cols) > index_tolerance).any()
		or (np.abs(rows_float - rows) > index_tolerance).any()
		or (cols < 0).any() or (cols >= width).any()
		or (rows < 0).any() or (rows >= height).any()
	):
		raise ValueError("UV vertices do not lie on the inferred tifxyz grid")

	vertex_cells = {
		int(vertex): (int(row), int(col))
		for vertex, row, col in zip(vertices, rows, cols)
	}
	grid = np.full((height, width), -1, dtype=np.int32)
	for vertex, (row, col) in vertex_cells.items():
		if grid[row, col] != -1:
			raise ValueError("more than one vertex occupies an inferred grid point")
		grid[row, col] = vertex

	seen = set()
	for original_face in faces:
		face = tuple(original_face)
		cells = {vertex_cells[vertex] for vertex in face}
		face_rows = {cell[0] for cell in cells}
		face_cols = {cell[1] for cell in cells}
		if len(cells) != 4 or len(face_rows) != 2 or len(face_cols) != 2:
			raise ValueError("a face does not follow the inferred UV grid")
		r0, r1 = min(face_rows), max(face_rows)
		c0, c1 = min(face_cols), max(face_cols)
		if r1 != r0 + 1 or c1 != c0 + 1 or (r0, c0) in seen:
			raise ValueError("the mesh has invalid or duplicate grid faces")
		seen.add((r0, c0))
	return grid


def scatter_grid(values, grid, fill):
	"""Place existing vertex values in a full grid and fill missing samples."""
	values = np.asarray(values)
	grid = np.asarray(grid)
	result = np.empty(grid.shape + values.shape[1:], dtype=values.dtype)
	result[...] = fill
	rows, cols = np.nonzero(grid >= 0)
	result[rows, cols] = values[grid[rows, cols]]
	return result


def surface_meta(meta, uuid, scale, points, mask):
	"""Return Villa-compatible metadata updated for exported arrays."""
	result = dict(meta or {})
	result.update({
		"format": "tifxyz", "type": "seg", "uuid": str(uuid),
		"scale": [float(scale[0]), float(scale[1])],
		"tiff_dimensions": [int(points.shape[1]), int(points.shape[0])],
	})
	valid = points[np.asarray(mask) >= 0.5]
	result["bbox"] = (
		[valid.min(axis=0).tolist(), valid.max(axis=0).tolist()]
		if len(valid) else [[-1.0] * 3, [-1.0] * 3]
	)
	return result


def write_surface(directory, points, mask, channels, meta, uuid, scale):
	"""Atomically publish a rectangular grid as a tifxyz directory."""
	import tifffile

	points = np.asarray(points, dtype=np.float32)
	if points.ndim != 3 or points.shape[2] != 3:
		raise ValueError("points must be a height by width by 3 array")
	mask = np.asarray(mask, dtype=np.float32)
	if mask.shape != points.shape[:2]:
		raise ValueError("mask does not match the coordinate grid")
	for name, values in channels.items():
		if np.asarray(values).shape != mask.shape:
			raise ValueError("channel %s does not match the coordinate grid" % name)

	directory = os.path.abspath(directory)
	parent = os.path.dirname(directory)
	os.makedirs(parent, exist_ok=True)
	staging = tempfile.mkdtemp(prefix=".velend-tifxyz-", dir=parent)
	try:
		for index, name in enumerate(("x", "y", "z")):
			tifffile.imwrite(os.path.join(staging, name + ".tif"), points[..., index])
		binary_mask = np.where(mask >= 0.5, MASK_KEEP, 0).astype(np.uint8)
		mask_path = os.path.join(staging, "mask.tif")
		try:
			tifffile.imwrite(
				mask_path, binary_mask, compression="lzw", tile=(1024, 1024)
			)
		except Exception:
			# Villa accepts an ordinary uint8 TIFF too. This also keeps export
			# usable on platforms whose imagecodecs wheel lacks LZW encoding.
			tifffile.imwrite(mask_path, binary_mask, compression=None)
		for name, values in channels.items():
			tifffile.imwrite(
				os.path.join(staging, name + ".tif"),
				np.asarray(values, dtype=np.float32),
			)
		with open(os.path.join(staging, "meta.json"), "w") as file:
			json.dump(surface_meta(meta, uuid, scale, points, mask), file, indent=2)
			file.write("\n")

		os.makedirs(directory, exist_ok=True)
		new_names = set(os.listdir(staging))
		for name in os.listdir(directory):
			if name.endswith(".tif") and name not in new_names:
				os.remove(os.path.join(directory, name))
		for name in new_names:
			os.replace(os.path.join(staging, name), os.path.join(directory, name))
	finally:
		shutil.rmtree(staging, ignore_errors=True)
