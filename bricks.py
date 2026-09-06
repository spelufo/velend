"""CPU side of the brick streaming system.

Nothing in here touches `gpu`, so it can run from operators and worker threads,
where no GPU context exists. `atlas.py` owns the matching GPU resources.

The volume is cut into a grid of cubic chunks of `BRICK_CORE` level 0 voxels.
A chunk that is resident on the GPU occupies one slot of the brick atlas, and
is stored there padded by `BRICK_PAD` voxels of its neighbours on every side so
that hardware trilinear filtering stays seamless right up to the core boundary.
"""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import state


# Brick geometry, in level 0 voxels.
BRICK_CORE = 64
BRICK_PAD = 1
BRICK_SIZE = BRICK_CORE + 2 * BRICK_PAD

# The atlas is a cube of SLOTS_PER_AXIS^3 bricks: 12^3 bricks of 66^3 R8 texels
# is a 792^3 texture, about 497MB on the GPU.
SLOTS_PER_AXIS = 12
SLOT_COUNT = SLOTS_PER_AXIS ** 3
ATLAS_DIM = SLOTS_PER_AXIS * BRICK_SIZE

# Chunks further than this from the focus are never candidates. 40 chunks is
# 2560 voxels, ~24mm, already past what SLOT_COUNT bricks can cover even when
# the meshes are a single flat sheet.
FOCUS_RADIUS_CHUNKS = 40

# Bricks uploaded per redraw. Each one is a staging texture plus a compute
# dispatch, so a handful per frame keeps the viewport responsive while the
# rest of the working set streams in over the following frames.
UPLOADS_PER_DRAW = 8
LOADER_THREADS = 6


def grid_dims(shape_xyz):
	"""Number of chunks along each axis needed to cover the volume."""
	return tuple(int(-(-int(s) // BRICK_CORE)) for s in shape_xyz)


def read_brick(volume, shape_xyz, chunk_xyz):
	"""Read one padded brick out of the volume, as float32 in [0, 1].

	Returned in numpy (Z, Y, X) order, ready for a GPUTexture staging buffer.
	Padding that falls outside the volume is left at zero.
	"""
	brick = np.zeros((BRICK_SIZE, BRICK_SIZE, BRICK_SIZE), dtype=np.float32)
	lo = np.asarray(chunk_xyz, dtype=np.int64) * BRICK_CORE - BRICK_PAD
	hi = lo + BRICK_SIZE
	clipped_lo = np.clip(lo, 0, shape_xyz)
	clipped_hi = np.clip(hi, 0, shape_xyz)
	if np.any(clipped_hi <= clipped_lo):
		return brick

	# The volume is indexed (Z, Y, X, level); level 0 is full resolution.
	data = volume[
		clipped_lo[2]:clipped_hi[2],
		clipped_lo[1]:clipped_hi[1],
		clipped_lo[0]:clipped_hi[0],
		0,
	]
	off = clipped_lo - lo
	brick[
		off[2]:off[2] + data.shape[0],
		off[1]:off[1] + data.shape[1],
		off[0]:off[0] + data.shape[2],
	] = data
	brick *= np.float32(1.0 / 255.0)
	return brick


class BrickLoader:
	"""Reads bricks off disk on worker threads, never blocking the UI thread."""

	def __init__(self, shape_xyz):
		self.shape_xyz = np.asarray(shape_xyz, dtype=np.int64)
		self.done = queue.Queue()
		self.executor = ThreadPoolExecutor(
			max_workers=LOADER_THREADS, thread_name_prefix="vlend-brick"
		)
		self.lock = threading.Lock()
		self.inflight = set()
		# `vesuvius.Volume` makes no threadsafety promise, so each worker gets
		# its own handle rather than sharing the one in `state`.
		self.local = threading.local()

	def volume(self):
		volume = getattr(self.local, "volume", None)
		if volume is None:
			volume = state.new_volume()
			self.local.volume = volume
		return volume

	def request(self, key, chunk_xyz):
		with self.lock:
			if key in self.inflight:
				return
			self.inflight.add(key)
		self.executor.submit(self._load, key, chunk_xyz)

	def _load(self, key, chunk_xyz):
		try:
			brick = read_brick(self.volume(), self.shape_xyz, chunk_xyz)
		except Exception as error:
			print("vlend: brick load failed at", chunk_xyz, error)
			brick = None
		self.done.put((key, brick))

	def drain(self, limit):
		"""Pop up to `limit` finished bricks. Returns [(key, brick_or_None)]."""
		loaded = []
		while len(loaded) < limit:
			try:
				item = self.done.get_nowait()
			except queue.Empty:
				break
			with self.lock:
				self.inflight.discard(item[0])
			loaded.append(item)
		return loaded

	def idle(self):
		with self.lock:
			return not self.inflight and self.done.empty()

	def shutdown(self):
		self.executor.shutdown(wait=False, cancel_futures=True)


class Residency:
	"""Which chunks own which atlas slots, and the page table that says so.

	`page` is the CPU mirror of the page table texture: one entry per chunk of
	the whole volume, holding `slot + 1`, or 0 when the chunk is not resident
	and the fragment shader should fall back to the low resolution volume. It
	is float32 because that is the only buffer format `GPUTexture` accepts;
	slot indices are small enough to be exact.
	"""

	def __init__(self, dims_xyz):
		self.dims = tuple(int(d) for d in dims_xyz)
		nx, ny, nz = self.dims
		self.page = np.zeros((nz, ny, nx), dtype=np.float32)
		self.flat = self.page.reshape(-1)
		self.slot_of = {}
		self.chunk_of = [None] * SLOT_COUNT
		self.free = list(reversed(range(SLOT_COUNT)))
		self.dirty = True

	def chunk_xyz(self, key):
		nx, ny, _ = self.dims
		return (key % nx, (key // nx) % ny, key // (nx * ny))

	def keys_of(self, chunks_xyz):
		"""Vectorised (N, 3) chunk coords -> flat page table keys."""
		nx, ny, _ = self.dims
		return (chunks_xyz[:, 2] * ny + chunks_xyz[:, 1]) * nx + chunks_xyz[:, 0]

	def evict_outside(self, wanted):
		"""Release every resident slot whose chunk is no longer wanted."""
		for key in [k for k in self.slot_of if k not in wanted]:
			slot = self.slot_of.pop(key)
			self.chunk_of[slot] = None
			self.free.append(slot)
			self.flat[key] = 0
			self.dirty = True

	def place(self, key):
		"""Claim a slot for a chunk. Returns the slot, or None if the atlas is full."""
		if key in self.slot_of:
			return None
		if not self.free:
			return None
		slot = self.free.pop()
		self.slot_of[key] = slot
		self.chunk_of[slot] = key
		self.flat[key] = slot + 1
		self.dirty = True
		return slot


def chunks_near(voxels, indices, lo, hi):
	"""Chunk coords touched by the triangles, restricted to the [lo, hi] voxel box.

	Each triangle contributes the chunks its bounding box overlaps, which is a
	superset of the chunks it actually crosses. Over-inclusion only costs a few
	slots; missing a chunk would show up as an untextured patch.
	"""
	if len(indices) == 0:
		return np.zeros((0, 3), dtype=np.int64)

	tris = voxels[indices]
	tri_lo = tris.min(axis=1)
	tri_hi = tris.max(axis=1)
	inside = np.all(tri_hi >= lo, axis=1) & np.all(tri_lo <= hi, axis=1)
	if not inside.any():
		return np.zeros((0, 3), dtype=np.int64)

	tri_lo = np.clip(tri_lo[inside], lo, hi)
	tri_hi = np.clip(tri_hi[inside], lo, hi)
	cmin = np.floor(tri_lo / BRICK_CORE).astype(np.int64)
	cmax = np.floor(tri_hi / BRICK_CORE).astype(np.int64)

	spans = cmax - cmin + 1
	counts = spans.prod(axis=1)
	total = int(counts.sum())
	if total == 0:
		return np.zeros((0, 3), dtype=np.int64)

	# Expand every per-triangle chunk box into the individual chunks it covers.
	owner = np.repeat(np.arange(len(counts)), counts)
	starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
	offset = np.arange(total) - np.repeat(starts, counts)
	span_x = spans[owner, 0]
	span_y = spans[owner, 1]
	return cmin[owner] + np.stack(
		[
			offset % span_x,
			(offset // span_x) % span_y,
			offset // (span_x * span_y),
		],
		axis=1,
	)
