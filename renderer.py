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

# Level 5 is the whole volume downsampled by 2^5, and is what fragments fall
# back to wherever no brick is resident.
_LORES_LEVEL = 5


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

	# Brick streaming. `residency` and `loader` are CPU only so the operator can
	# retarget without a GPU context; `atlas` owns the textures.
	atlas = None
	residency = None
	loader = None
	# Chunks the focus point currently asks for. Bricks that finish loading after
	# falling out of this set are dropped rather than uploaded.
	wanted = frozenset()

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
		if cls.residency is not None:
			return

		volume = state.get_volume()
		cls.shape_xyz = tuple(int(s) for s in reversed(volume.shape(0)))
		unit_scale = bpy.context.scene.unit_settings.scale_length
		# Full-res voxels are `volume.resolution` µm across.
		cls.voxels_per_unit = (1000000.0 * unit_scale) / volume.resolution
		cls.residency = bricks.Residency(bricks.grid_dims(cls.shape_xyz))
		cls.loader = bricks.BrickLoader(cls.shape_xyz)
		print("vlend: chunk grid", cls.residency.dims, "over", cls.shape_xyz, "voxels")

	@classmethod
	def ensure_volume(cls):
		if cls.volume is not None:
			return

		volume = state.get_volume()
		lores = np.ascontiguousarray(volume[:, :, :, _LORES_LEVEL], dtype=np.float32)
		lores *= np.float32(1.0 / 255.0)

		# Numpy is C-order (Z, Y, X); GPUTexture is (width, height, depth). The
		# level 5 array covers the level 0 extent rounded up to a multiple of
		# 2^5, which is the extent to normalise against.
		dims = tuple(reversed(lores.shape))
		cls.lores_extent = tuple(d * (1 << _LORES_LEVEL) for d in dims)

		cls.volume = gpu.types.GPUTexture(
			dims,
			format='R8',
			data=atlas_module.texture_data(lores),
		)
		# Trilinear interpolation between the random texels.
		cls.volume.filter_mode(True)

	@classmethod
	def ensure_atlas(cls):
		if cls.atlas is not None:
			return
		cls.atlas = atlas_module.BrickAtlas(cls.residency.dims)
		cls.atlas.sync_page(cls.residency.page)
		cls.residency.dirty = False

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
		shader_info.define("SLOTS_PER_AXIS", str(bricks.SLOTS_PER_AXIS))
		shader_info.define("ATLAS_DIM", str(bricks.ATLAS_DIM))
		shader_info.define("PAGE_DIMS", "ivec3(%d, %d, %d)" % cls.residency.dims)
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
		shader_info.sampler(1, 'FLOAT_3D', "atlas")
		shader_info.sampler(2, 'FLOAT_3D', "pageTable")
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
		cls.ensure_atlas()
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

	@classmethod
	def batch_from_object(cls, obj_eval):
		from gpu_extras.batch import batch_for_shader

		arrays = cls.mesh_arrays(obj_eval)
		if arrays is None:
			return None
		positions, indices = arrays
		return batch_for_shader(cls.shader, 'TRIS', {"position": positions}, indices=indices)

	@classmethod
	def retarget(cls, context, focus):
		"""Pick the chunks nearest the focus point and start streaming them in.

		Runs from the operator, without a GPU context: it only decides what the
		working set should be and queues the reads. `pump_uploads` moves the
		results onto the GPU on the following draws.
		"""
		cls.ensure_grid()
		residency = cls.residency
		dims = np.asarray(residency.dims, dtype=np.int64)
		focus_voxel = np.asarray(focus, dtype=np.float64) * cls.voxels_per_unit

		# Only triangles within reach of the focus can contribute, which keeps
		# this bounded no matter how large the meshes are.
		focus_chunk = focus_voxel / bricks.BRICK_CORE
		radius = bricks.FOCUS_RADIUS_CHUNKS
		lo_chunk = np.clip(np.floor(focus_chunk - radius), 0, dims - 1)
		hi_chunk = np.clip(np.ceil(focus_chunk + radius), 0, dims - 1)
		lo = lo_chunk * bricks.BRICK_CORE
		hi = (hi_chunk + 1) * bricks.BRICK_CORE - 1.0

		found = []
		for instance in context.evaluated_depsgraph_get().object_instances:
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
			residency.evict_outside(set())
			cls.wanted = frozenset()
			cls.request_redraw()
			return 0

		chunks = np.clip(np.concatenate(found), 0, dims - 1)
		# Deduplicate on the flat key rather than on rows; it is the same answer
		# and vastly cheaper for a dense triangle soup.
		keys = np.unique(residency.keys_of(chunks))
		nx, ny, _ = residency.dims
		coords = np.stack([keys % nx, (keys // nx) % ny, keys // (nx * ny)], axis=1)
		centers = (coords + 0.5) * bricks.BRICK_CORE
		nearest = np.argsort(np.linalg.norm(centers - focus_voxel, axis=1))[:bricks.SLOT_COUNT]
		keys = keys[nearest]
		coords = coords[nearest]

		cls.wanted = frozenset(int(k) for k in keys)
		residency.evict_outside(cls.wanted)
		for key, coord in zip(keys, coords):
			key = int(key)
			if key in residency.slot_of:
				continue
			cls.loader.request(key, (int(coord[0]), int(coord[1]), int(coord[2])))

		cls.request_redraw()
		return len(keys)

	@classmethod
	def pump_uploads(cls):
		"""Move finished bricks into the atlas. Only valid inside `view_draw`."""
		loaded = cls.loader.drain(bricks.UPLOADS_PER_DRAW)
		for key, brick in loaded:
			if brick is None or key not in cls.wanted:
				continue
			slot = cls.residency.place(key)
			if slot is None:
				continue
			cls.atlas.upload(slot, brick)

		# After the uploads, so the page table never points at a slot whose
		# brick has not been written yet.
		if cls.residency.dirty:
			cls.atlas.sync_page(cls.residency.page)
			cls.residency.dirty = False

		if loaded or not cls.loader.idle():
			cls.request_redraw()

	def view_update(self, context, depsgraph):
		# This runs outside of the drawing code, without an active GPU context,
		# so it only invalidates the caches used by `view_draw`.
		self.live_instances.add(self)
		if any(update.is_updated_geometry for update in depsgraph.updates):
			self.batches.clear()
			self.meshes.clear()

	def view_draw(self, context, depsgraph):
		self.live_instances.add(self)
		self.ensure_gpu_resources()
		if self.shader is None:
			return
		self.pump_uploads()

		# Blender clears the color and depth buffers before the engine draws. Writing depth
		# is what lets it occlude the overlays that are drawn on top of the result.
		gpu.state.depth_test_set('LESS_EQUAL')
		gpu.state.depth_mask_set(True)

		self.shader.uniform_sampler("volume", self.volume)
		self.shader.uniform_sampler("atlas", self.atlas.texture)
		self.shader.uniform_sampler("pageTable", self.atlas.page_texture)

		# Iterating the instances also draws the duplis and the geometry nodes instances,
		# which share the batch of the object they instance.
		for instance in depsgraph.object_instances:
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


def register():
	bpy.utils.register_class(VolumeSamplerRenderEngine)
	if not bpy.app.timers.is_registered(_watch_shader_files):
		bpy.app.timers.register(_watch_shader_files, persistent=True)
	if not bpy.app.timers.is_registered(_watch_streaming):
		bpy.app.timers.register(_watch_streaming, persistent=True)


def unregister():
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
