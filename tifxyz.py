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
	"""The vertex arrays of a grid: positions, quads, UVs and channel values."""
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


def read_surface(directory, step=1, load_mask=True, load_channels=True):
	"""Read a tifxyz surface directory into the arrays a mesh is built from.

	`step` keeps every nth row and column of the grid, which is how a segment
	whose grid is millions of quads is brought down to something a viewport can
	live with.
	"""
	meta = read_meta(directory)
	x, y, z = read_coordinates(directory)
	valid = valid_points(x, y, z)
	if load_mask:
		mask = read_mask(directory, valid.shape)
		if mask is not None:
			valid = valid & mask
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
	return Surface(directory, meta, valid.shape, positions, quads, uvs, values)
