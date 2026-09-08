"""CPU side of the brick streaming system.

Nothing in here touches `gpu`, so it can run from operators and worker threads,
where no GPU context exists. `atlas.py` owns the matching GPU resources.

Each pyramid level is cut into a grid of cubic chunks of `BRICK_CORE` voxels.
A chunk that is resident on the GPU occupies one slot of its level's atlas, and
is stored there padded by `BRICK_PAD` voxels of its neighbours on every side so
that hardware trilinear filtering stays seamless right up to the core boundary.
"""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import state


# Brick geometry, in voxels of the level being loaded.
BRICK_CORE = 64
BRICK_PAD = 1
BRICK_SIZE = BRICK_CORE + 2 * BRICK_PAD

# Atlas capacity is independently tunable for each streamed pyramid level.
# L0 is 792^3 R8 (~497MB); L1 is 660^3 R8 (~287MB).
L0_SLOTS_PER_AXIS = 12
L1_SLOTS_PER_AXIS = 10
SLOTS_PER_AXIS = (L0_SLOTS_PER_AXIS, L1_SLOTS_PER_AXIS)
SLOT_COUNTS = tuple(n ** 3 for n in SLOTS_PER_AXIS)
ATLAS_DIMS = tuple(n * BRICK_SIZE for n in SLOTS_PER_AXIS)

# The combined working set reaches this many L1 chunks from the focus, or twice
# as many L0 chunks because L1 is downsampled by two on every axis.
L1_FOCUS_RADIUS_CHUNKS = 40

# Bricks uploaded per redraw. Each one is a staging texture plus a compute
# dispatch, so a handful per frame keeps the viewport responsive while the
# rest of the working set streams in over the following frames.
UPLOADS_PER_DRAW = 8
LOADER_THREADS = 6


def grid_dims(shape_xyz):
	"""Number of chunks along each axis needed to cover the volume."""
	return tuple(int(-(-int(s) // BRICK_CORE)) for s in shape_xyz)


def select_lod_chunks(chunks, l0_dims, l1_dims, focus_voxel):
	"""Split mesh-intersecting L0 chunks into nearest L0 and L1 overflow.

	Returns ``((l0_keys, l0_coords), (l1_keys, l1_coords))`` with each level
	ordered nearest-first and capped to that atlas's capacity.
	"""
	l0_dims = np.asarray(l0_dims, dtype=np.int64)
	l1_dims = np.asarray(l1_dims, dtype=np.int64)
	focus_voxel = np.asarray(focus_voxel, dtype=np.float64)
	chunks = np.clip(np.asarray(chunks, dtype=np.int64), 0, l0_dims - 1)
	if not len(chunks):
		empty_keys = np.zeros(0, dtype=np.int64)
		empty_coords = np.zeros((0, 3), dtype=np.int64)
		return (empty_keys, empty_coords), (empty_keys.copy(), empty_coords.copy())

	# Use flat keys to deduplicate efficiently, then recover XYZ coordinates.
	nx0, ny0, _ = (int(d) for d in l0_dims)
	keys = np.unique((chunks[:, 2] * ny0 + chunks[:, 1]) * nx0 + chunks[:, 0])
	coords = np.stack(
		[keys % nx0, (keys // nx0) % ny0, keys // (nx0 * ny0)], axis=1
	)
	centers = (coords + 0.5) * BRICK_CORE
	nearest = np.argsort(
		np.linalg.norm(centers - focus_voxel, axis=1), kind='stable'
	)
	ordered_keys = keys[nearest]
	ordered_coords = coords[nearest]
	l0_count = min(len(ordered_keys), SLOT_COUNTS[0])
	l0 = (ordered_keys[:l0_count], ordered_coords[:l0_count])

	overflow = ordered_coords[l0_count:]
	if not len(overflow):
		empty_keys = np.zeros(0, dtype=np.int64)
		empty_coords = np.zeros((0, 3), dtype=np.int64)
		return l0, (empty_keys, empty_coords)

	l1_chunks = np.clip(overflow // 2, 0, l1_dims - 1)
	nx1, ny1, _ = (int(d) for d in l1_dims)
	l1_keys = np.unique(
		(l1_chunks[:, 2] * ny1 + l1_chunks[:, 1]) * nx1 + l1_chunks[:, 0]
	)
	l1_coords = np.stack(
		[l1_keys % nx1, (l1_keys // nx1) % ny1, l1_keys // (nx1 * ny1)],
		axis=1,
	)
	l1_centers_l0 = (l1_coords + 0.5) * BRICK_CORE * 2
	l1_nearest = np.argsort(
		np.linalg.norm(l1_centers_l0 - focus_voxel, axis=1), kind='stable'
	)[:SLOT_COUNTS[1]]
	return l0, (l1_keys[l1_nearest], l1_coords[l1_nearest])


def read_brick(volume, shape_xyz, chunk_xyz, level=0):
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

	# The volume is indexed (Z, Y, X, level).
	data = volume[
		clipped_lo[2]:clipped_hi[2],
		clipped_lo[1]:clipped_hi[1],
		clipped_lo[0]:clipped_hi[0],
		level,
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

	def __init__(self, shapes_xyz):
		self.shapes_xyz = {
			int(level): np.asarray(shape, dtype=np.int64)
			for level, shape in shapes_xyz.items()
		}
		self.done = queue.Queue()
		self.executor = ThreadPoolExecutor(
			max_workers=LOADER_THREADS, thread_name_prefix="vlend-brick"
		)
		self.lock = threading.Lock()
		# key -> Future, so a stale request can still be cancelled while it's
		# only sitting in the executor's queue rather than actually reading.
		self.inflight = {}
		# `vesuvius.Volume` makes no threadsafety promise, so each worker gets
		# its own handle rather than sharing the one in `state`.
		self.local = threading.local()

	def volume(self):
		volume = getattr(self.local, "volume", None)
		if volume is None:
			volume = state.new_volume()
			self.local.volume = volume
		return volume

	def request(self, level, key, chunk_xyz):
		request_key = (int(level), int(key))
		with self.lock:
			if request_key in self.inflight:
				return
			self.inflight[request_key] = self.executor.submit(
				self._load, request_key, chunk_xyz
			)

	def cancel_unwanted(self, wanted):
		"""Drop queued reads for chunks that fell out of `wanted`.

		A future can only be cancelled while it's still sitting in the executor's
		queue; one already reading off disk finishes normally, and its result is
		silently discarded by the `wanted` check in `renderer.pump_uploads`. With
		a handful of worker threads and dozens of requests queued during a drag,
		most of the backlog is still cancellable, which is what actually frees
		the workers up for the chunks currently wanted.
		"""
		with self.lock:
			stale = [key for key in self.inflight if key not in wanted]
			for key in stale:
				if self.inflight[key].cancel():
					del self.inflight[key]

	def _load(self, request_key, chunk_xyz):
		level, _ = request_key
		try:
			brick = read_brick(
				self.volume(), self.shapes_xyz[level], chunk_xyz, level=level
			)
		except Exception as error:
			print("vlend: L%d brick load failed at" % level, chunk_xyz, error)
			brick = None
		self.done.put((request_key, brick))

	def drain(self, limit):
		"""Pop up to `limit` finished bricks. Returns [(key, brick_or_None)]."""
		loaded = []
		while len(loaded) < limit:
			try:
				item = self.done.get_nowait()
			except queue.Empty:
				break
			with self.lock:
				self.inflight.pop(item[0], None)
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
	and the fragment shader should fall back to the next coarser source. It is
	float32 because that is the only buffer format `GPUTexture` accepts;
	slot indices are small enough to be exact.
	"""

	def __init__(self, dims_xyz, slot_count):
		self.dims = tuple(int(d) for d in dims_xyz)
		nx, ny, nz = self.dims
		self.page = np.zeros((nz, ny, nx), dtype=np.float32)
		self.flat = self.page.reshape(-1)
		self.slot_of = {}
		self.chunk_of = [None] * int(slot_count)
		self.free = list(reversed(range(int(slot_count))))
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

	def release(self, key):
		"""Undo a placement, for example when its GPU upload failed."""
		slot = self.slot_of.pop(key, None)
		if slot is None:
			return
		self.chunk_of[slot] = None
		self.free.append(slot)
		self.flat[key] = 0
		self.dirty = True


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
