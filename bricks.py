"""CPU side of the brick streaming system.

Nothing in here touches `gpu`, so it can run from operators and worker threads,
where no GPU context exists. `atlas.py` owns the matching GPU resources.

Each pyramid level is cut into a grid of cubic chunks of `BRICK_CORE` voxels.
A chunk that is resident on the GPU occupies one slot of its level's atlas, and
is stored there padded by `BRICK_PAD` voxels of its neighbours on every side so
that hardware trilinear filtering stays seamless right up to the core boundary.
"""

import math
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np


# Brick geometry, in voxels of the level being loaded.
BRICK_CORE = 64
BRICK_PAD = 1
BRICK_SIZE = BRICK_CORE + 2 * BRICK_PAD

# Stream levels zero through this level. Level 5 remains the whole-volume
# fallback. Lowering this constant also removes the disabled levels from the
# compiled fragment shader and avoids allocating any of their GPU resources.
FALLBACK_LEVEL = 5
LEVEL_CAP = 4

# Decimal megabytes available to each L0..L4 R8 atlas. Page tables and the one
# short-lived staging brick are not part of these budgets. The defaults derive
# 12^3, 10^3, 8^3, 6^3, and 4^3 slots respectively (~1.01GB in total).
ATLAS_MEMORY_MB = (500, 300, 150, 64, 20)


def atlas_slots_per_axis(memory_mb):
	"""Largest cubic padded-brick atlas whose R8 payload fits `memory_mb`."""
	budget_bytes = int(float(memory_mb) * 1_000_000)
	brick_bytes = BRICK_SIZE ** 3
	if budget_bytes < brick_bytes:
		raise ValueError("atlas budget must fit at least one padded brick")

	# Correct around floating-point cube-root boundaries so the result never
	# exceeds the configured byte budget.
	slots = int(math.floor((budget_bytes / brick_bytes) ** (1.0 / 3.0)))
	while (slots + 1) ** 3 * brick_bytes <= budget_bytes:
		slots += 1
	while slots ** 3 * brick_bytes > budget_bytes:
		slots -= 1
	return slots


if not 0 <= LEVEL_CAP < FALLBACK_LEVEL:
	raise ValueError("LEVEL_CAP must be between 0 and 4")
if len(ATLAS_MEMORY_MB) < FALLBACK_LEVEL:
	raise ValueError("ATLAS_MEMORY_MB must provide budgets for L0 through L4")

ACTIVE_LEVELS = tuple(range(LEVEL_CAP + 1))
SLOTS_PER_AXIS = tuple(atlas_slots_per_axis(ATLAS_MEMORY_MB[level]) for level in ACTIVE_LEVELS)
SLOT_COUNTS = tuple(n ** 3 for n in SLOTS_PER_AXIS)
ATLAS_DIMS = tuple(n * BRICK_SIZE for n in SLOTS_PER_AXIS)

# The combined working set reaches this many chunks at the coarsest streamed
# level. Each increase in LEVEL_CAP therefore doubles its physical reach.
FOCUS_RADIUS_CHUNKS = 40

# Bricks uploaded per redraw. Each one is a staging texture plus a compute
# dispatch, so a handful per frame keeps the viewport responsive while the
# rest of the working set streams in over the following frames.
UPLOADS_PER_DRAW = 8
LOADER_THREADS = 6

# Distinct from None, which means the read failed. The singleton only crosses
# worker-thread queues within this Python process.
EMPTY_BRICK = object()


def grid_dims(shape_xyz):
	"""Number of chunks along each axis needed to cover the volume."""
	return tuple(int(-(-int(s) // BRICK_CORE)) for s in shape_xyz)


def select_lod_chunks(chunks, level_dims, focus_voxel, slot_counts=None):
	"""Cascade mesh-intersecting L0 chunks through the streamed levels.

	Each level takes its nearest candidates up to capacity. Its overflow is
	collapsed by two on every axis and becomes the next level's candidates.
	Returns ``{level: (keys, coords_xyz)}`` for every supplied level.
	"""
	levels = tuple(sorted(level_dims))
	if not levels or levels != tuple(range(len(levels))):
		raise ValueError("level_dims must contain consecutive levels starting at 0")
	if slot_counts is None:
		slot_counts = SLOT_COUNTS[:len(levels)]
	if len(slot_counts) != len(levels):
		raise ValueError("slot_counts must match level_dims")

	dims = {level: np.asarray(level_dims[level], dtype=np.int64) for level in levels}
	focus_voxel = np.asarray(focus_voxel, dtype=np.float64)
	chunks = np.clip(np.asarray(chunks, dtype=np.int64), 0, dims[0] - 1)
	selected = {}
	if not len(chunks):
		for level in levels:
			selected[level] = (
				np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=np.int64)
			)
		return selected

	candidates = chunks
	for level, capacity in zip(levels, slot_counts):
		candidates = np.clip(candidates, 0, dims[level] - 1)
		nx, ny, _ = (int(d) for d in dims[level])
		keys = np.unique(
			(candidates[:, 2] * ny + candidates[:, 1]) * nx + candidates[:, 0]
		)
		coords = np.stack(
			[keys % nx, (keys // nx) % ny, keys // (nx * ny)], axis=1
		)
		centers_l0 = (coords + 0.5) * BRICK_CORE * (1 << level)
		nearest = np.argsort(
			np.linalg.norm(centers_l0 - focus_voxel, axis=1), kind='stable'
		)
		ordered_keys = keys[nearest]
		ordered_coords = coords[nearest]
		count = min(len(ordered_keys), int(capacity))
		selected[level] = (ordered_keys[:count], ordered_coords[:count])
		overflow = ordered_coords[count:]
		if level == levels[-1]:
			break
		if len(overflow):
			candidates = overflow // 2
		else:
			for coarser in levels[level + 1:]:
				selected[coarser] = (
					np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=np.int64)
				)
			break
	return selected


def read_brick(array, chunk_xyz):
	"""Read one padded brick, or return EMPTY_BRICK when every voxel is zero.

	`array` is the zarr array of the level being loaded. The brick is returned
	in numpy (Z, Y, X) order, ready for a GPUTexture staging buffer. Padding
	that falls outside the volume is left at zero.
	"""
	shape_xyz = np.asarray(array.shape[::-1], dtype=np.int64)
	lo = np.asarray(chunk_xyz, dtype=np.int64) * BRICK_CORE - BRICK_PAD
	hi = lo + BRICK_SIZE
	clipped_lo = np.clip(lo, 0, shape_xyz)
	clipped_hi = np.clip(hi, 0, shape_xyz)
	if np.any(clipped_hi <= clipped_lo):
		return EMPTY_BRICK

	# The zarr arrays are indexed (Z, Y, X).
	data = array[
		clipped_lo[2]:clipped_hi[2],
		clipped_lo[1]:clipped_hi[1],
		clipped_lo[0]:clipped_hi[0],
	]
	if not np.any(data):
		return EMPTY_BRICK
	brick = np.zeros((BRICK_SIZE, BRICK_SIZE, BRICK_SIZE), dtype=np.float32)
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

	def __init__(self, volume):
		# The pyramid's zarr arrays, indexed by level. Reads are thread safe, so
		# every worker shares them.
		self.volume = volume
		self.done = queue.Queue()
		self.executor = ThreadPoolExecutor(
			max_workers=LOADER_THREADS, thread_name_prefix="velend-brick"
		)
		self.lock = threading.Lock()
		# key -> Future, so a stale request can still be cancelled while it's
		# only sitting in the executor's queue rather than actually reading.
		self.inflight = {}

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
			brick = read_brick(self.volume[level], chunk_xyz)
		except Exception as error:
			print("velend: L%d brick load failed at" % level, chunk_xyz, error)
			brick = None
		self.done.put((request_key, brick))

	def drain(self, limit):
		"""Pop results containing a brick, EMPTY_BRICK, or None on failure."""
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
