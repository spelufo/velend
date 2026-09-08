import os
import weakref
import bpy
import gpu
import numpy as np

from . import atlas as atlas_module
from . import bricks
from . import state


_SHADERS_DIR = os.path.join(os.path.dirname(__file__), "shaders")
_VERT_SHADER_PATH = os.path.join(_SHADERS_DIR, "volume.vert")
_FRAG_SHADER_PATH = os.path.join(_SHADERS_DIR, "volume.frag")
_SHADER_WATCH_INTERVAL = 0.25
# `area.tag_redraw()` only marks the area dirty; it does not wake Blender's
# event loop. Without a timer forcing a wake-up, uploads queued from
# `pump_uploads` would only drain while the user is otherwise generating
# events (e.g. orbiting the viewport).
_STREAM_WATCH_INTERVAL = 0.05
# The 3D cursor is tool state, not depsgraph data, so moving it never reaches
# `view_update`; polling is the same trick used above for the other two.
_CURSOR_WATCH_INTERVAL = 0.1

class VolumeSamplerRenderEngine(bpy.types.RenderEngine):
	bl_idname = "VOLUME_SAMPLER"
	bl_label = "Volume Sampler"

	# Created on the first draw, where a GPU context is guaranteed to be active,
	# and shared by all engine instances (Blender creates one per viewport).
	volume = None
	shader = None
	shader_mtimes = None
	uniform_buffer = None

	# Level 0 voxels per Blender unit, and the extent in level 0 voxels that the
	# low resolution texture spans. Both are set up by `ensure_grid`.
	voxels_per_unit = 1.0
	lores_extent = (1.0, 1.0, 1.0)
	shape_xyz = None

	# Brick streaming. Residency and loader state are CPU only so the operator
	# can retarget without a GPU context; the atlases own the textures.
	atlases = {}
	residencies = {}
	loader = None
	shapes_xyz = {}
	# Chunks the focus point currently asks for, per pyramid level. Completed
	# reads that fall out of these sets are dropped rather than uploaded.
	wanted = {level: frozenset() for level in bricks.ACTIVE_LEVELS}
	# Coarser levels start one at a time after all finer work has reached its
	# atlas. Values are tuples of (page_key, chunk_xyz) requests.
	pending_levels = {}
	# Occupancy learned from completed reads. Empty chunks are never requested
	# again, and their parents are promoted as higher-level fallbacks.
	empty_chunks = {level: set() for level in bricks.ACTIVE_LEVELS}
	fallback_keys = {level: set() for level in bricks.ACTIVE_LEVELS}
	last_focus_voxel = np.zeros(3, dtype=np.float64)
	# The focus last passed to `retarget`, compared against the live 3D cursor
	# by `_watch_cursor` -- cursor moves aren't depsgraph updates, so nothing
	# else notices them.
	last_cursor = None

	# Rebuilt only when the geometry of the scene changes.
	batches = {}
	meshes = {}

	# One instance per viewport that has drawn at least once. `Area.tag_redraw()`
	# only marks the region dirty, which a RENDERED-shading viewport treats as
	# "repaint the cached image" rather than "call view_draw again" -- only the
	# engine's own `tag_redraw()` sets the `RE_ENGINE_DO_DRAW` flag Blender
	# actually checks before re-invoking it. A WeakSet avoids outliving Blender's
	# own ownership of these instances.
	live_instances = weakref.WeakSet()

	@classmethod
	def shader_file_mtimes(cls):
		return (
			os.stat(_VERT_SHADER_PATH).st_mtime_ns,
			os.stat(_FRAG_SHADER_PATH).st_mtime_ns,
			os.stat(atlas_module.COPY_SHADER_PATH).st_mtime_ns,
		)

	@classmethod
	def shaders_changed(cls):
		try:
			return cls.shader_file_mtimes() != cls.shader_mtimes
		except OSError:
			return False

	@classmethod
	def request_redraw(cls):
		for engine in list(cls.live_instances):
			try:
				engine.tag_redraw()
			except ReferenceError:
				cls.live_instances.discard(engine)

	@classmethod
	def ensure_grid(cls):
		"""Set up the chunk grid and the brick loader. No GPU context needed."""
		if cls.residencies:
			return

		volume = state.get_volume()
		cls.shapes_xyz = {
			level: tuple(int(s) for s in reversed(volume[level].shape))
			for level in bricks.ACTIVE_LEVELS
		}
		cls.shape_xyz = cls.shapes_xyz[0]
		unit_scale = bpy.context.scene.unit_settings.scale_length
		# Full-res voxels are `state.resolution` µm across.
		cls.voxels_per_unit = (1000000.0 * unit_scale) / state.resolution
		cls.residencies = {
			level: bricks.Residency(
				bricks.grid_dims(cls.shapes_xyz[level]), bricks.SLOT_COUNTS[level]
			)
			for level in bricks.ACTIVE_LEVELS
		}
		cls.loader = bricks.BrickLoader(volume)
		cls.empty_chunks = {level: set() for level in bricks.ACTIVE_LEVELS}
		cls.fallback_keys = {level: set() for level in bricks.ACTIVE_LEVELS}
		for level in bricks.ACTIVE_LEVELS:
			print(
				"velend: L%d chunk grid" % level,
				cls.residencies[level].dims,
				"over",
				cls.shapes_xyz[level],
				"voxels",
			)

	@classmethod
	def ensure_volume(cls):
		if cls.volume is not None:
			return

		volume = state.get_volume()
		lores = np.ascontiguousarray(
			volume[bricks.FALLBACK_LEVEL][:], dtype=np.float32
		)
		lores *= np.float32(1.0 / 255.0)

		# Numpy is C-order (Z, Y, X); GPUTexture is (width, height, depth). The
		# level 5 array covers the level 0 extent rounded up to a multiple of
		# 2^5, which is the extent to normalise against.
		dims = tuple(reversed(lores.shape))
		cls.lores_extent = tuple(d * (1 << bricks.FALLBACK_LEVEL) for d in dims)

		cls.volume = gpu.types.GPUTexture(
			dims,
			format='R8',
			data=atlas_module.texture_data(lores),
		)
		# Trilinear interpolation between the random texels.
		cls.volume.filter_mode(True)

	@classmethod
	def ensure_atlases(cls):
		for level in bricks.ACTIVE_LEVELS:
			if level in cls.atlases:
				continue
			residency = cls.residencies[level]
			atlas = atlas_module.BrickAtlas(
				residency.dims, bricks.SLOTS_PER_AXIS[level]
			)
			atlas.sync_page(residency.page)
			residency.dirty = False
			cls.atlases[level] = atlas

	@classmethod
	def ensure_shader(cls):
		if cls.shader is not None and not cls.shaders_changed():
			return

		try:
			mtimes = cls.shader_file_mtimes()
			with open(_VERT_SHADER_PATH, encoding="utf-8") as vert_file:
				vert_source = vert_file.read()
			with open(_FRAG_SHADER_PATH, encoding="utf-8") as frag_file:
				frag_source = frag_file.read()
		except OSError as error:
			print("Failed to read shaders:", error)
			return

		vert_out = gpu.types.GPUStageInterfaceInfo("volume_interface")
		vert_out.smooth('VEC3', "voxelCoord")

		shader_info = gpu.types.GPUShaderCreateInfo()
		# The brick geometry never changes at runtime, so it costs nothing to
		# bake it into the shader rather than pay for it in the uniform block.
		shader_info.define("BRICK_CORE", str(bricks.BRICK_CORE))
		shader_info.define("BRICK_PAD", str(bricks.BRICK_PAD))
		shader_info.define("BRICK_SIZE", str(bricks.BRICK_SIZE))
		shader_info.define("LEVEL_CAP", str(bricks.LEVEL_CAP))
		for level in bricks.ACTIVE_LEVELS:
			shader_info.define(
				"L%d_SLOTS_PER_AXIS" % level, str(bricks.SLOTS_PER_AXIS[level])
			)
			shader_info.define(
				"L%d_ATLAS_DIM" % level, str(bricks.ATLAS_DIMS[level])
			)
			shader_info.define(
				"L%d_PAGE_DIMS" % level,
				"ivec3(%d, %d, %d)" % cls.residencies[level].dims,
			)
		shader_info.typedef_source("""
			struct VolumeUniforms {
				mat4 viewProjectionMatrix;
				mat4 modelMatrix;
				vec3 voxelsPerUnit;
				vec3 loresExtent;
			};
		""")
		shader_info.uniform_buf(0, "VolumeUniforms", "volumeUniforms")
		shader_info.sampler(0, 'FLOAT_3D', "volume")
		for level in bricks.ACTIVE_LEVELS:
			shader_info.sampler(
				1 + level * 2, 'FLOAT_3D', "l%dAtlas" % level
			)
			shader_info.sampler(
				2 + level * 2, 'FLOAT_3D', "l%dPageTable" % level
			)
		shader_info.vertex_in(0, 'VEC3', "position")
		shader_info.vertex_out(vert_out)
		shader_info.fragment_out(0, 'VEC4', "FragColor")
		shader_info.vertex_source(vert_source)
		shader_info.fragment_source(frag_source)

		try:
			shader = gpu.shader.create_from_info(shader_info)
		except Exception as error:
			print("Shader compile failed:", error)
			cls.shader_mtimes = mtimes
			return

		cls.shader = shader
		cls.shader_mtimes = mtimes
		cls.batches.clear()
		print("Loaded shaders")

	@classmethod
	def ensure_gpu_resources(cls):
		cls.ensure_grid()
		cls.ensure_volume()
		cls.ensure_atlases()
		cls.ensure_shader()

	@classmethod
	def update_uniform_buffer(cls, view_projection_matrix, model_matrix):
		# std140 lays out each mat4 as four vec4 columns. `vec3` has a 16-byte stride.
		data = np.empty(40, dtype=np.float32)
		data[:16] = np.asarray(view_projection_matrix, dtype=np.float32).T.ravel()
		data[16:32] = np.asarray(model_matrix, dtype=np.float32).T.ravel()
		data[32:35] = cls.voxels_per_unit
		data[35] = 0.0
		data[36:39] = cls.lores_extent
		data[39] = 0.0
		buffer = gpu.types.Buffer('FLOAT', len(data), data)
		if cls.uniform_buffer is None:
			cls.uniform_buffer = gpu.types.GPUUniformBuf(buffer)
		else:
			cls.uniform_buffer.update(buffer)

	@classmethod
	def mesh_arrays(cls, obj_eval):
		"""Local-space triangle soup for an object, cached until its geometry changes."""
		name = obj_eval.name
		if name in cls.meshes:
			return cls.meshes[name]

		mesh = obj_eval.to_mesh()
		mesh.calc_loop_triangles()
		arrays = None
		if mesh.loop_triangles:
			# `foreach_get` copies whole attributes at once, and buffers supporting the Python
			# buffer protocol are uploaded to the vertex buffer without a per element conversion.
			positions = np.empty((len(mesh.vertices), 3), 'f')
			indices = np.empty((len(mesh.loop_triangles), 3), 'i')
			mesh.vertices.foreach_get("co", np.reshape(positions, len(mesh.vertices) * 3))
			mesh.loop_triangles.foreach_get("vertices", np.reshape(indices, len(mesh.loop_triangles) * 3))
			arrays = (positions, indices)
		obj_eval.to_mesh_clear()
		cls.meshes[name] = arrays
		return arrays

	@staticmethod
	def instance_visible(instance, depsgraph, viewport=None):
		"""Effective viewport visibility, including the owning collection."""
		if not instance.show_self:
			return False
		if instance.is_instance:
			# Dupli visibility is computed per generated instance by the
			# depsgraph; its source object need not belong to the view layer.
			return True
		return instance.object.original.visible_get(
			view_layer=depsgraph.view_layer,
			viewport=viewport,
		)

	@classmethod
	def batch_from_object(cls, obj_eval):
		from gpu_extras.batch import batch_for_shader

		arrays = cls.mesh_arrays(obj_eval)
		if arrays is None:
			return None
		positions, indices = arrays
		return batch_for_shader(cls.shader, 'TRIS', {"position": positions}, indices=indices)

	@classmethod
	def retarget(cls, depsgraph, focus):
		"""Pick the chunks nearest the focus point and start streaming them in.

		Runs without a GPU context, from the operator or from `view_update`: it
		only decides what the working set should be and queues the reads.
		`pump_uploads` moves the results onto the GPU on the following draws.
		"""
		cls.last_cursor = tuple(focus)
		cls.ensure_grid()
		l0_residency = cls.residencies[0]
		dims = np.asarray(l0_residency.dims, dtype=np.int64)
		focus_voxel = np.asarray(focus, dtype=np.float64) * cls.voxels_per_unit
		cls.last_focus_voxel = focus_voxel

		# Search the full multilevel reach in L0 space. Each level takes its
		# nearest candidates; only the remainder is collapsed into the next level.
		focus_chunk = focus_voxel / bricks.BRICK_CORE
		radius = bricks.FOCUS_RADIUS_CHUNKS * (1 << bricks.LEVEL_CAP)
		lo_chunk = np.clip(np.floor(focus_chunk - radius), 0, dims - 1)
		hi_chunk = np.clip(np.ceil(focus_chunk + radius), 0, dims - 1)
		lo = lo_chunk * bricks.BRICK_CORE
		hi = (hi_chunk + 1) * bricks.BRICK_CORE - 1.0

		found = []
		for instance in depsgraph.object_instances:
			# The iterator can retain hidden instancers and objects whose layer
			# collection is disabled, so apply effective visibility explicitly.
			if not cls.instance_visible(instance, depsgraph):
				continue
			obj = instance.object
			if obj.type != 'MESH':
				continue
			arrays = cls.mesh_arrays(obj)
			if arrays is None:
				continue
			positions, indices = arrays
			matrix = np.asarray(instance.matrix_world, dtype=np.float64)
			world = positions @ matrix[:3, :3].T + matrix[:3, 3]
			chunks = bricks.chunks_near(world * cls.voxels_per_unit, indices, lo, hi)
			if len(chunks):
				found.append(chunks)

		if not found:
			for residency in cls.residencies.values():
				residency.evict_outside(set())
			cls.wanted = {level: frozenset() for level in bricks.ACTIVE_LEVELS}
			cls.pending_levels = {}
			cls.fallback_keys = {level: set() for level in bricks.ACTIVE_LEVELS}
			cls.loader.cancel_unwanted(set())
			cls.request_redraw()
			return 0

		chunks = np.clip(np.concatenate(found), 0, dims - 1)
		selected = bricks.select_lod_chunks(
			chunks,
			{level: cls.residencies[level].dims for level in bricks.ACTIVE_LEVELS},
			focus_voxel,
		)

		cls.wanted = {
			level: frozenset(int(k) for k in selected[level][0])
			for level in bricks.ACTIVE_LEVELS
		}
		cls.pending_levels = {}
		cls.fallback_keys = {level: set() for level in bricks.ACTIVE_LEVELS}
		# Reapply occupancy learned by earlier targets before queuing any reads.
		for level in bricks.ACTIVE_LEVELS:
			for key in tuple(cls.wanted[level] & cls.empty_chunks[level]):
				cls.mark_empty(level, key, reschedule=False)
		for level in bricks.ACTIVE_LEVELS:
			cls.residencies[level].evict_outside(cls.wanted[level])
		# Frees the worker pool from stale reads queued by an earlier retarget
		# (e.g. mid-drag) before dispatching this round's requests.
		wanted_requests = {
			(level, key) for level in bricks.ACTIVE_LEVELS for key in cls.wanted[level]
		}
		cls.loader.cancel_unwanted(wanted_requests)
		for level in bricks.ACTIVE_LEVELS:
			pending = cls.missing_requests(level)
			if level == 0:
				for key, coord in pending:
					cls.loader.request(0, key, coord)
				continue
			if pending:
				cls.pending_levels[level] = pending
		cls.queue_next_level_if_ready()

		cls.request_redraw()
		return sum(len(cls.wanted[level]) for level in bricks.ACTIVE_LEVELS)

	@classmethod
	def key_distance(cls, level, key):
		coord = np.asarray(cls.residencies[level].chunk_xyz(key), dtype=np.float64)
		center_l0 = (coord + 0.5) * bricks.BRICK_CORE * (1 << level)
		return float(np.linalg.norm(center_l0 - cls.last_focus_voxel))

	@classmethod
	def missing_requests(cls, level):
		"""Wanted, nonresident chunks in nearest-first order."""
		residency = cls.residencies[level]
		keys = [
			key for key in cls.wanted[level]
			if key not in residency.slot_of and key not in cls.empty_chunks[level]
		]
		keys.sort(key=lambda key: (cls.key_distance(level, key), key))
		return tuple((key, residency.chunk_xyz(key)) for key in keys)

	@classmethod
	def mark_empty(cls, level, key, reschedule=True):
		"""Skip an empty chunk and prioritize its parent in the next atlas."""
		cls.empty_chunks[level].add(key)
		wanted = set(cls.wanted[level])
		wanted.discard(key)
		cls.wanted[level] = frozenset(wanted)
		cls.fallback_keys[level].discard(key)
		cls.residencies[level].evict_outside(cls.wanted[level])

		next_level = level + 1
		if next_level not in cls.residencies:
			return
		coord = np.asarray(cls.residencies[level].chunk_xyz(key), dtype=np.int64) // 2
		next_residency = cls.residencies[next_level]
		coord = np.clip(coord, 0, np.asarray(next_residency.dims) - 1)
		parent = int(next_residency.keys_of(coord.reshape(1, 3))[0])

		next_wanted = set(cls.wanted[next_level])
		next_wanted.add(parent)
		cls.fallback_keys[next_level].add(parent)
		capacity = bricks.SLOT_COUNTS[next_level]
		if len(next_wanted) > capacity:
			ordinary = next_wanted - cls.fallback_keys[next_level]
			victims = ordinary if ordinary else next_wanted
			victim = max(victims, key=lambda item: (cls.key_distance(next_level, item), item))
			next_wanted.remove(victim)
			cls.fallback_keys[next_level].discard(victim)
		cls.wanted[next_level] = frozenset(next_wanted)
		next_residency.evict_outside(cls.wanted[next_level])

		if parent in cls.wanted[next_level] and parent in cls.empty_chunks[next_level]:
			cls.mark_empty(next_level, parent, reschedule=reschedule)
		elif reschedule:
			# Rebuild this pending level so an evicted ordinary request is removed
			# and the newly discovered fallback is inserted in distance order.
			cls.pending_levels[next_level] = cls.missing_requests(next_level)

		if reschedule:
			wanted_requests = {
				(active_level, wanted_key)
				for active_level in bricks.ACTIVE_LEVELS
				for wanted_key in cls.wanted[active_level]
			}
			cls.loader.cancel_unwanted(wanted_requests)

	@classmethod
	def queue_next_level_if_ready(cls):
		"""Start the next coarser level once all earlier work is drained."""
		if not cls.pending_levels or not cls.loader.idle():
			return
		level = min(cls.pending_levels)
		pending = cls.pending_levels.pop(level)
		for key, coord in pending:
			if key in cls.wanted[level] and key not in cls.residencies[level].slot_of:
				cls.loader.request(level, key, coord)

	@classmethod
	def pump_uploads(cls):
		"""Move finished bricks into their atlases. Only valid inside `view_draw`."""
		loaded = cls.loader.drain(bricks.UPLOADS_PER_DRAW)
		for request_key, brick in loaded:
			level, key = request_key
			if key not in cls.wanted[level]:
				continue
			if brick is bricks.EMPTY_BRICK:
				cls.mark_empty(level, key)
				continue
			if brick is None:
				continue
			residency = cls.residencies[level]
			slot = residency.place(key)
			if slot is None:
				continue
			if not cls.atlases[level].upload(slot, brick):
				residency.release(key)

		# After the uploads, so the page table never points at a slot whose
		# brick has not been written yet.
		for level in bricks.ACTIVE_LEVELS:
			residency = cls.residencies[level]
			if residency.dirty:
				cls.atlases[level].sync_page(residency.page)
				residency.dirty = False

		cls.queue_next_level_if_ready()

		if loaded or cls.pending_levels or not cls.loader.idle():
			cls.request_redraw()

	def view_update(self, context, depsgraph):
		# This runs outside of the drawing code, without an active GPU context,
		# so it only invalidates the caches used by `view_draw` and queues reads;
		# it never touches the GPU itself.
		self.live_instances.add(self)
		geometry_changed = False
		transform_changed = False
		visibility_changed = False
		for update in depsgraph.updates:
			geometry = update.is_updated_geometry
			transform = update.is_updated_transform
			shading = update.is_updated_shading
			geometry_changed |= geometry
			transform_changed |= transform
			# Blender exposes no dedicated visibility flag. Hide/show arrives as
			# an otherwise-unclassified depsgraph update.
			visibility_changed |= not (geometry or transform or shading)

		if geometry_changed:
			# Batches hold local-space positions only, so a transform-only change
			# doesn't need them rebuilt -- `instance.matrix_world` is reapplied
			# every draw regardless.
			self.batches.clear()
			self.meshes.clear()

		if (geometry_changed or transform_changed or visibility_changed) and self.residencies:
			# Hide/show changes have neither the geometry nor transform flag, but
			# they change both what is drawn and which bricks are wanted.
			# Retargeting depends on world-space triangle positions, which a
			# transform change moves just as much as an edit to the mesh itself.
			self.retarget(depsgraph, context.scene.cursor.location)

	def view_draw(self, context, depsgraph):
		self.live_instances.add(self)
		first_init = not self.residencies
		self.ensure_gpu_resources()
		if first_init and self.residencies:
			# Otherwise nothing streams in until geometry changes or the operator
			# runs, even though the grid (and so a focus point) already exists.
			self.retarget(depsgraph, context.scene.cursor.location)
		if self.shader is None:
			return
		self.pump_uploads()

		# Blender clears the color and depth buffers before the engine draws. Writing depth
		# is what lets it occlude the overlays that are drawn on top of the result.
		gpu.state.depth_test_set('LESS_EQUAL')
		gpu.state.depth_mask_set(True)

		self.shader.uniform_sampler("volume", self.volume)
		for level in bricks.ACTIVE_LEVELS:
			self.shader.uniform_sampler(
				"l%dAtlas" % level, self.atlases[level].texture
			)
			self.shader.uniform_sampler(
				"l%dPageTable" % level, self.atlases[level].page_texture
			)

		# Iterating the instances also draws the duplis and the geometry nodes instances,
		# which share the batch of the object they instance.
		for instance in depsgraph.object_instances:
			if not self.instance_visible(instance, depsgraph, context.space_data):
				continue
			obj = instance.object
			if obj.type != 'MESH':
				continue
			if obj.name not in self.batches:
				self.batches[obj.name] = self.batch_from_object(obj)
			batch = self.batches[obj.name]
			if batch is None:
				continue
			self.update_uniform_buffer(
				context.region_data.perspective_matrix,
				instance.matrix_world,
			)
			self.shader.uniform_block("volumeUniforms", self.uniform_buffer)
			batch.draw(self.shader)

		gpu.state.depth_mask_set(False)
		gpu.state.depth_test_set('NONE')


def _watch_shader_files():
	if VolumeSamplerRenderEngine.shaders_changed():
		VolumeSamplerRenderEngine.request_redraw()
	return _SHADER_WATCH_INTERVAL


def _watch_streaming():
	loader = VolumeSamplerRenderEngine.loader
	if loader is not None and not loader.idle():
		VolumeSamplerRenderEngine.request_redraw()
	return _STREAM_WATCH_INTERVAL


def _watch_cursor():
	cls = VolumeSamplerRenderEngine
	scene = bpy.context.scene
	if not cls.residencies or scene is None:
		return _CURSOR_WATCH_INTERVAL
	cursor = tuple(scene.cursor.location)
	if cursor != cls.last_cursor:
		cls.retarget(bpy.context.evaluated_depsgraph_get(), cursor)
	return _CURSOR_WATCH_INTERVAL


def register():
	bpy.utils.register_class(VolumeSamplerRenderEngine)
	if not bpy.app.timers.is_registered(_watch_shader_files):
		bpy.app.timers.register(_watch_shader_files, persistent=True)
	if not bpy.app.timers.is_registered(_watch_streaming):
		bpy.app.timers.register(_watch_streaming, persistent=True)
	if not bpy.app.timers.is_registered(_watch_cursor):
		bpy.app.timers.register(_watch_cursor, persistent=True)


def unregister():
	if bpy.app.timers.is_registered(_watch_cursor):
		bpy.app.timers.unregister(_watch_cursor)
	if bpy.app.timers.is_registered(_watch_streaming):
		bpy.app.timers.unregister(_watch_streaming)
	if bpy.app.timers.is_registered(_watch_shader_files):
		bpy.app.timers.unregister(_watch_shader_files)
	if VolumeSamplerRenderEngine.loader is not None:
		VolumeSamplerRenderEngine.loader.shutdown()
		VolumeSamplerRenderEngine.loader = None
	bpy.utils.unregister_class(VolumeSamplerRenderEngine)


if __name__ == "__main__":
	register()
