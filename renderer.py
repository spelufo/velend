import os
import weakref
from concurrent.futures import ThreadPoolExecutor
import bpy
import gpu
import numpy as np

from . import atlas as atlas_module
from . import bricks
from . import metadata
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
# Orbiting the viewport changes which bricks are wanted but only ever reaches
# `view_draw`, and retargeting from there would walk every mesh on each frame of
# the drag. The view matrices are recorded as they are drawn with and this timer
# retargets once they have stopped changing.
_VIEW_WATCH_INTERVAL = 0.1

class VolumeSamplerRenderEngine(bpy.types.RenderEngine):
	bl_idname = "VOLUME_SAMPLER"
	bl_label = "Volume Sampler"

	# Created on the first draw, where a GPU context is guaranteed to be active,
	# and shared by all engine instances (Blender creates one per viewport).
	load_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="velend-volume")
	load_future = None
	load_key = None
	pyramid = None
	shader = None
	shader_mtimes = None
	uniform_buffer = None

	# Why the scene's volume could not be opened, shown by the scene panel, or
	# None when it is open. `failed_path` keeps a bad path from being retried on
	# every single draw.
	volume_error = "No volume set"
	failed_path = None
	# Set when the scene's volume changes. The teardown frees GPU textures, so
	# it waits for `view_draw`, where a GPU context is active.
	pending_reset = False

	shape_xyz = None
	# The affine taking a Blender world point into the loaded volume's level 0
	# voxels, set up by `ensure_grid`. The scene's coordinates are metric -- a
	# Blender unit is `scale_length` metres -- and sit in the frame of the volume
	# the scene was set up against, which `ui` names and states the voxel size
	# of. `compute_world_to_voxels` is the three steps between the two: the
	# scene's units into micrometers, the registration onto the volume being
	# rendered, and that volume's own voxel size.
	world_to_voxels = np.eye(4)

	# Brick streaming. Residency and loader state are CPU only so the operator
	# can retarget without a GPU context; the atlases own the textures.
	atlases = {}
	residencies = {}
	loader = None
	shapes_xyz = {}
	# Chunks the focus point currently asks for, per pyramid level. Completed
	# reads that fall out of these sets are dropped rather than uploaded.
	wanted = {level: frozenset() for level in bricks.LEVELS}
	# Coarser levels start one at a time after all finer work has reached its
	# atlas. Values are tuples of (page_key, chunk_xyz) requests.
	pending_levels = {}
	# Occupancy learned from completed reads. Empty chunks are never requested
	# again, and their parents are promoted as higher-level fallbacks.
	empty_chunks = {level: set() for level in bricks.LEVELS}
	fallback_keys = {level: set() for level in bricks.LEVELS}
	last_focus_voxel = np.zeros(3, dtype=np.float64)
	# The focus last passed to `retarget`, compared against the live 3D cursor
	# by `_watch_cursor` -- cursor moves aren't depsgraph updates, so nothing
	# else notices them.
	last_cursor = None

	# Rebuilt only when the geometry of the scene changes.
	batches = {}
	meshes = {}

	# The matrix the viewport this instance draws was last drawn with, as a plain
	# tuple: hashable, and it keeps no reference to Blender's own data. Set per
	# instance by `view_draw`; the class level None covers the instances that
	# have not drawn yet and the UV editor, which has no 3D view to cull against.
	frustum_matrix = None
	# The view matrices `_watch_view` last saw, and whether it has already
	# retargeted for them.
	last_view_key = None
	view_settled = False

	# One instance per viewport that has drawn at least once. `Area.tag_redraw()`
	# only marks the region dirty, which a RENDERED-shading viewport treats as
	# "repaint the cached image" rather than "call view_draw again" -- only the
	# engine's own `tag_redraw()` sets the `RE_ENGINE_DO_DRAW` flag Blender
	# actually checks before re-invoking it. A WeakSet avoids outliving Blender's
	# own ownership of these instances.
	live_instances = weakref.WeakSet()

	@classmethod
	def settings(cls):
		return bpy.context.scene.velend

	@classmethod
	def volume_path(cls):
		"""The configured volume directory, resolved against the .blend."""
		path = cls.settings().volume_path.strip()
		if not path:
			return ""
		# The file browser always hands back an absolute path; expanding covers
		# the paths typed into the field by hand.
		if "://" in path:
			return path
		return os.path.expanduser(bpy.path.abspath(path))

	@classmethod
	def get_volume(cls):
		"""The scene's pyramid, or None with the reason left in `volume_error`."""
		path = cls.volume_path()
		key = (path, cls.settings().source_url.strip())
		state.online_access = bpy.app.online_access
		if not path:
			cls.volume_error = "No volume set"
			return None
		if key == cls.failed_path:
			return None
		if cls.load_key != key:
			cls.pyramid = None
			cls.load_key = key
			cls.load_future = cls.load_executor.submit(cls.load_initial, *key)
		if cls.pyramid is not None:
			return cls.pyramid
		if not cls.load_future.done():
			cls.volume_error = "Loading volume..."
			return None
		try:
			cls.pyramid = cls.load_future.result()
		except Exception as error:
			cls.failed_path = key
			cls.volume_error = "Cannot open volume: %s" % error
			print("velend:", cls.volume_error)
			cls.load_future = None
			return None
		cls.load_future = None
		cls.volume_error = None
		return cls.pyramid

	@staticmethod
	def load_initial(path, source_url):
		if not source_url and "://" in path and not state.online_access:
			raise OSError("Network access is disabled in Blender")
		volume = state.open_volume(path, source_url)
		if len(volume) <= bricks.LEVELS[-1]:
			raise ValueError("Volume has %d pyramid levels, level %d is needed" % (
				len(volume), bricks.LEVELS[-1]))
		return volume

	@staticmethod
	def um_per_unit():
		"""Micrometers to a Blender unit, from the scene's own unit settings."""
		return 1000000.0 * bpy.context.scene.unit_settings.scale_length

	@classmethod
	def physical_transform(cls):
		"""The affine taking a point in the frame the scene's coordinates are in
		onto the volume being rendered, both frames in micrometers.

		The metadata registers a sample's volumes pairwise in each other's
		voxels, so the matrix for the pair is conjugated by the two voxel sizes
		to have it say the same thing about micrometers instead.

		Nothing registering the pair leaves the identity, which takes the two
		volumes to share an origin and their axes: as much as can be assumed of
		two scans of one object, and a guess that never resizes what the scene
		already holds. Identity as well while the volume being rendered is the
		one the scene is in, the pair being the same volume.
		"""
		settings = cls.settings()
		matrix = metadata.volume_transform(
			settings.scene_volume_id,
			# The URL first: while both name the volume, only it is untouched
			# by the cache directory naming VC3D puts around it.
			metadata.volume_id_for(settings.source_url, settings.volume_path),
		)
		if matrix is None:
			return np.eye(4)
		scene_um = settings.scene_resolution or settings.resolution
		loaded_um = settings.resolution or scene_um
		physical = np.eye(4)
		# Voxels of the scene's volume in, so a column is scaled by what one of
		# them is worth in micrometers; voxels of the loaded volume out, so a row
		# is scaled by what one of those is worth.
		physical[:3, :3] = matrix[:3, :3] * (loaded_um / scene_um)
		physical[:3, 3] = matrix[:3, 3] * loaded_um
		return physical

	@classmethod
	def compute_world_to_voxels(cls):
		"""The affine a world point reaches the loaded volume's level 0 voxels
		through: the scene's units into micrometers, the registration above, and
		then the voxel size of the volume being rendered."""
		resolution = cls.settings().resolution or 1.0
		matrix = cls.physical_transform()
		matrix[:3, :3] *= cls.um_per_unit() / resolution
		matrix[:3, 3] /= resolution
		return matrix

	@classmethod
	def compute_world_from_voxels(cls):
		"""The way back, off the scene's fields rather than off what is loaded:
		an importer places its points before anything has been rendered."""
		try:
			return np.linalg.inv(cls.compute_world_to_voxels())
		except np.linalg.LinAlgError:
			return np.eye(4)

	@classmethod
	def to_voxels(cls, world, matrix=None):
		"""Blender world coordinates into the loaded volume's level 0 voxels, the
		way the vertex shader takes them. One point or an (n, 3) array of them."""
		matrix = cls.world_to_voxels if matrix is None else matrix
		return world @ matrix[:3, :3].T + matrix[:3, 3]

	@classmethod
	def view_frusta(cls):
		"""The frustum of every viewport currently drawing, in volume voxels.

		One entry per live instance that has drawn, dilated by
		`FRUSTUM_MARGIN_CHUNKS`. A brick is kept when it falls inside any of
		them, so every open viewport stays textured.

		Empty -- meaning nothing is culled -- while the setting is off, before
		any viewport has drawn, and for a matrix the planes cannot come out of.
		`retarget` also runs from operators and from the very first draw, and
		culling everything there would leave the viewport blank.
		"""
		if not cls.settings().frustum_culling:
			return []
		margin = bricks.FRUSTUM_MARGIN_CHUNKS * bricks.BRICK_CORE
		frusta = []
		for instance in cls.live_instances:
			matrix = instance.frustum_matrix
			if matrix is None:
				continue
			planes = bricks.frustum_planes(matrix, cls.world_to_voxels, margin)
			if planes is not None:
				frusta.append(planes)
		return frusta

	@classmethod
	def world_bounds(cls):
		"""The loaded volume's extent as a `(low, high)` pair of corners in
		Blender world coordinates: the box its own corners span, brought back
		through `world_to_voxels`.

		The volume need not sit square to the scene once transformed, so this
		is the axis aligned box around it rather than the volume itself.
		"""
		shape = np.asarray(cls.shape_xyz, dtype=np.float64)
		corners = np.array([
			[x, y, z] for x in (0.0, shape[0]) for y in (0.0, shape[1])
			for z in (0.0, shape[2])
		])
		try:
			inverse = np.linalg.inv(cls.world_to_voxels)
		except np.linalg.LinAlgError:
			return np.zeros(3), shape
		corners = corners @ inverse[:3, :3].T + inverse[:3, 3]
		return corners.min(axis=0), corners.max(axis=0)

	@classmethod
	def status(cls):
		"""A (message, icon) pair describing what is loaded, for the UI."""
		if not cls.settings().volume_path.strip():
			return "No volume set", 'INFO'
		if cls.volume_error:
			return cls.volume_error, 'INFO' if cls.load_future is not None else 'ERROR'
		if cls.loader is not None and cls.loader.error:
			return cls.loader.error, 'ERROR'
		if cls.shape_xyz is None or cls.pending_reset:
			return "Not loaded yet", 'INFO'
		return "%d x %d x %d voxels" % cls.shape_xyz, 'CHECKMARK'

	@classmethod
	def reset(cls):
		"""Ask for everything derived from the volume to be rebuilt."""
		if cls.load_future is not None:
			cls.load_future.cancel()
		cls.load_future = None
		cls.load_key = None
		cls.pyramid = None
		cls.pending_reset = True
		# Cleared here rather than in the deferred teardown so that the panel
		# stops reporting the previous volume's trouble straight away.
		cls.failed_path = None
		cls.volume_error = None
		cls.request_redraw()

	@classmethod
	def apply_reset(cls):
		"""Drop it all, so the rest of this draw rebuilds it. Needs a GPU context."""
		cls.pending_reset = False
		if cls.loader is not None:
			cls.loader.shutdown()
			cls.loader = None
		cls.atlases = {}
		cls.residencies = {}
		cls.shapes_xyz = {}
		cls.shape_xyz = None
		cls.world_to_voxels = np.eye(4)
		cls.wanted = {level: frozenset() for level in bricks.LEVELS}
		cls.pending_levels = {}
		cls.empty_chunks = {level: set() for level in bricks.LEVELS}
		cls.fallback_keys = {level: set() for level in bricks.LEVELS}
		cls.last_focus_voxel = np.zeros(3, dtype=np.float64)
		cls.last_cursor = None
		cls.last_view_key = None
		cls.view_settled = False
		# The page table dimensions are baked into the fragment shader, so a
		# volume of a different size needs the shader compiled again.
		cls.shader = None
		cls.shader_mtimes = None
		cls.batches.clear()

	@classmethod
	def retarget_now(cls):
		"""Recompute the working set against the scene as it stands."""
		if not cls.residencies or cls.pending_reset:
			return
		cls.retarget(bpy.context.evaluated_depsgraph_get(), bpy.context.scene.cursor.location)

	@classmethod
	def rescale(cls):
		"""Reapply the voxel size to the grid already built, without rebuilding it."""
		if not cls.residencies or cls.pending_reset:
			return
		cls.world_to_voxels = cls.compute_world_to_voxels()
		cls.retarget_now()

	@classmethod
	def shader_file_mtimes(cls):
		return (
			os.stat(_VERT_SHADER_PATH).st_mtime_ns,
			os.stat(_FRAG_SHADER_PATH).st_mtime_ns,
			os.stat(atlas_module.COPY_SHADER_PATH).st_mtime_ns,
		)

	@classmethod
	def reload_shaders(cls):
		"""Drop the compiled shaders so the next draw builds them again."""
		cls.shader = None
		cls.shader_mtimes = None
		cls.request_redraw()

	@classmethod
	def shaders_changed(cls):
		try:
			return cls.shader_file_mtimes() != cls.shader_mtimes
		except OSError:
			return False

	@classmethod
	def request_redraw(cls):
		for window in bpy.context.window_manager.windows:
			for area in window.screen.areas:
				if area.type == 'PROPERTIES' or (area.type == 'IMAGE_EDITOR' and area.spaces.active.mode == 'UV'):
					area.tag_redraw()
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

		volume = cls.get_volume()
		if volume is None:
			return
		cls.shapes_xyz = {
			level: tuple(int(s) for s in reversed(volume[level].shape))
			for level in bricks.LEVELS
		}
		# Level 0 is the coordinate system everything else is expressed in, so
		# it is read straight off the pyramid rather than out of `shapes_xyz`:
		# `LEVELS` need not name it.
		cls.shape_xyz = tuple(int(s) for s in reversed(volume[0].shape))
		cls.world_to_voxels = cls.compute_world_to_voxels()
		cls.residencies = {
			level: bricks.Residency(
				bricks.grid_dims(cls.shapes_xyz[level]), bricks.SLOT_COUNTS[level]
			)
			for level in bricks.LEVELS
		}
		cls.loader = bricks.BrickLoader(volume)
		cls.empty_chunks = {level: set() for level in bricks.LEVELS}
		cls.fallback_keys = {level: set() for level in bricks.LEVELS}
		for level in bricks.LEVELS:
			print(
				"velend: L%d chunk grid" % level,
				cls.residencies[level].dims,
				"over",
				cls.shapes_xyz[level],
				"voxels",
			)

	@classmethod
	def ensure_atlases(cls):
		if not cls.residencies:
			return
		for level in bricks.LEVELS:
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
		if not cls.residencies:
			return
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

		try:
			shader = cls.create_shader(vert_source, frag_source)
		except Exception as error:
			print("Shader compile failed:", error)
			cls.shader_mtimes = mtimes
			return
		cls.shader = shader
		cls.shader_mtimes = mtimes
		cls.batches.clear()
		print("Loaded shaders")

	@classmethod
	def create_shader(cls, vert_source, frag_source, uv=False):
		"""Share volume sampling between the 3D and UV projections."""
		vert_out = gpu.types.GPUStageInterfaceInfo("volume_interface")
		vert_out.smooth('VEC3', "voxelCoord")

		shader_info = gpu.types.GPUShaderCreateInfo()
		# The brick geometry never changes at runtime, so it costs nothing to
		# bake it into the shader rather than pay for it in the uniform block.
		shader_info.define("BRICK_CORE", str(bricks.BRICK_CORE))
		shader_info.define("BRICK_PAD", str(bricks.BRICK_PAD))
		shader_info.define("BRICK_SIZE", str(bricks.BRICK_SIZE))
		settings = cls.settings()
		resolution = settings.resolution or 1.0
		shader_info.define("NORMAL_SAMPLES", str(max(settings.num_samples, 1)))
		shader_info.define(
			"SAMPLE_DELTA",
			repr(settings.render_depth / max(settings.num_samples, 1) / resolution),
		)
		shader_info.define(
			"SAMPLE_OFFSET", repr(settings.render_depth_offset / resolution)
		)
		shader_info.define("SKIP_VOID", "1" if settings.skip_void else "0")
		shader_info.define(
			"INVERT_SAMPLING_DIRECTION",
			"1" if settings.invert_sampling_direction else "0",
		)
		# The debug view is a whole other branch of the fragment shader rather
		# than a uniform, so toggling it recompiles: `reload_shaders` is what
		# the setting calls to make that happen.
		shader_info.define(
			"DEBUG_LEVEL_COLORS", "1" if settings.debug_level_colors else "0"
		)
		for level in bricks.LEVELS:
			shader_info.define("L%d_ACTIVE" % level, "1")
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
			# What a level 0 coordinate is multiplied by to reach this level's
			# voxels. A negative power of two, so the literal is exact.
			shader_info.define(
				"L%d_SCALE" % level, repr(1.0 / (1 << level))
			)
		shader_info.typedef_source("""
			struct VolumeUniforms {
				mat4 viewProjectionMatrix;
				mat4 modelMatrix;
				mat4 worldToVoxels;
			};
		""")
		shader_info.uniform_buf(0, "VolumeUniforms", "volumeUniforms")
		for level in bricks.LEVELS:
			shader_info.sampler(
				level * 2, 'FLOAT_3D', "l%dAtlas" % level
			)
			shader_info.sampler(
				level * 2 + 1, 'FLOAT_3D', "l%dPageTable" % level
			)
		shader_info.vertex_in(0, 'VEC3', "position")
		if uv:
			shader_info.vertex_in(1, 'VEC2', "uv")
		shader_info.vertex_out(vert_out)
		shader_info.fragment_out(0, 'VEC4', "FragColor")
		shader_info.vertex_source(vert_source)
		shader_info.fragment_source(frag_source)

		return gpu.shader.create_from_info(shader_info)

	@classmethod
	def ensure_gpu_resources(cls):
		cls.ensure_grid()
		if not cls.residencies:
			return
		cls.ensure_atlases()
		cls.ensure_shader()

	@classmethod
	def update_uniform_buffer(cls, view_projection_matrix, model_matrix):
		# std140 lays out each mat4 as four vec4 columns. The matrices are
		# transposed on the way in: Blender writes them a row at a time, GLSL
		# reads them a column at a time.
		data = np.empty(48, dtype=np.float32)
		data[:16] = np.asarray(view_projection_matrix, dtype=np.float32).T.ravel()
		data[16:32] = np.asarray(model_matrix, dtype=np.float32).T.ravel()
		data[32:48] = np.asarray(cls.world_to_voxels, dtype=np.float32).T.ravel()
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
		if cls.pending_reset:
			return 0
		cls.last_cursor = tuple(focus)
		cls.ensure_grid()
		if not cls.residencies:
			return 0
		cls.loader.error = None
		# The chunk search runs in level 0 chunks, which `LEVELS` need not
		# stream, so the grid comes from the volume's own shape.
		dims = np.asarray(bricks.grid_dims(cls.shape_xyz), dtype=np.int64)
		focus_voxel = cls.to_voxels(np.asarray(focus, dtype=np.float64))
		cls.last_focus_voxel = focus_voxel

		# Search the full multilevel reach in L0 space. Each level takes its
		# nearest candidates; only the remainder is collapsed into the next level.
		focus_chunk = focus_voxel / bricks.BRICK_CORE
		radius = bricks.FOCUS_RADIUS_CHUNKS * (1 << bricks.LEVELS[-1])
		lo_chunk = np.clip(np.floor(focus_chunk - radius), 0, dims - 1)
		hi_chunk = np.clip(np.ceil(focus_chunk + radius), 0, dims - 1)
		lo = lo_chunk * bricks.BRICK_CORE
		hi = (hi_chunk + 1) * bricks.BRICK_CORE - 1.0

		# One set of planes for the whole pass: the frusta do not change while it
		# runs, and extracting them per mesh would just repeat the inversion.
		planes_list = cls.view_frusta()

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
			# The chunks a triangle covers are the ones it covers in the loaded
			# volume, which an affine leaves a triangle in all the same.
			voxels = cls.to_voxels(world)
			chunks = bricks.chunks_near(voxels, indices, lo, hi, planes_list)
			if len(chunks):
				found.append(chunks)

		if not found:
			for residency in cls.residencies.values():
				residency.evict_outside(set())
			cls.wanted = {level: frozenset() for level in bricks.LEVELS}
			cls.pending_levels = {}
			cls.fallback_keys = {level: set() for level in bricks.LEVELS}
			cls.loader.cancel_unwanted(set())
			cls.request_redraw()
			return 0

		chunks = np.clip(np.concatenate(found), 0, dims - 1)
		selected = bricks.select_lod_chunks(
			chunks,
			{level: cls.residencies[level].dims for level in bricks.LEVELS},
			focus_voxel,
		)

		cls.wanted = {
			level: frozenset(int(k) for k in selected[level][0])
			for level in bricks.LEVELS
		}
		cls.pending_levels = {}
		cls.fallback_keys = {level: set() for level in bricks.LEVELS}
		# Reapply occupancy learned by earlier targets before queuing any reads.
		for level in bricks.LEVELS:
			for key in tuple(cls.wanted[level] & cls.empty_chunks[level]):
				cls.mark_empty(level, key, reschedule=False)
		for level in bricks.LEVELS:
			cls.residencies[level].evict_outside(cls.wanted[level])
		# Frees the worker pool from stale reads queued by an earlier retarget
		# (e.g. mid-drag) before dispatching this round's requests.
		wanted_requests = {
			(level, key) for level in bricks.LEVELS for key in cls.wanted[level]
		}
		cls.loader.cancel_unwanted(wanted_requests)
		for level in bricks.LEVELS:
			pending = cls.missing_requests(level)
			# The coarsest level goes out at once, so the whole mesh has
			# something on it as soon as possible; the finer ones wait their
			# turn and sharpen it from the cursor outwards.
			if level == bricks.LOAD_ORDER[0]:
				for key, coord in pending:
					cls.loader.request(level, key, coord)
				continue
			if pending:
				cls.pending_levels[level] = pending
		cls.queue_next_level_if_ready()

		cls.request_redraw()
		return sum(len(cls.wanted[level]) for level in bricks.LEVELS)

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

		next_level = bricks.next_level(level)
		if next_level is None:
			return
		# Collapsing onto the next level's grid takes one halving per level
		# skipped between the two.
		coord = np.asarray(
			cls.residencies[level].chunk_xyz(key), dtype=np.int64
		) >> (next_level - level)
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
				for active_level in bricks.LEVELS
				for wanted_key in cls.wanted[active_level]
			}
			cls.loader.cancel_unwanted(wanted_requests)

	@classmethod
	def queue_next_level_if_ready(cls):
		"""Start the next level in `LOAD_ORDER` once all earlier work is drained."""
		if not cls.pending_levels or not cls.loader.idle():
			return
		level = next(
			candidate for candidate in bricks.LOAD_ORDER
			if candidate in cls.pending_levels
		)
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
		for level in bricks.LEVELS:
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
		# Recorded rather than acted on: `_watch_view` retargets once the view
		# has settled, so orbiting does not walk every mesh on each frame.
		region_data = context.region_data
		if region_data is not None:
			self.frustum_matrix = tuple(
				tuple(row) for row in region_data.perspective_matrix
			)
		if self.pending_reset:
			self.apply_reset()
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

		for level in bricks.LEVELS:
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
	state.online_access = bpy.app.online_access
	if VolumeSamplerRenderEngine.load_future is not None:
		VolumeSamplerRenderEngine.request_redraw()
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


def _watch_view():
	"""Retarget once the viewports have stopped moving.

	Orbiting reaches `view_draw` alone, and only records the matrix there, so
	this is what notices. A tick that sees the same matrices as the last one had
	no navigation between them, which is when the new working set is worth the
	walk over the scene's meshes.
	"""
	cls = VolumeSamplerRenderEngine
	scene = bpy.context.scene
	if not cls.residencies or scene is None or not scene.velend.frustum_culling:
		return _VIEW_WATCH_INTERVAL
	# `live_instances` is a WeakSet, whose iteration order says nothing, so the
	# key is sorted rather than taken in the order the instances come out in.
	key = tuple(sorted(
		instance.frustum_matrix for instance in cls.live_instances
		if instance.frustum_matrix is not None
	))
	if key != cls.last_view_key:
		cls.last_view_key = key
		cls.view_settled = False
	elif not cls.view_settled:
		cls.view_settled = True
		cls.retarget_now()
	return _VIEW_WATCH_INTERVAL


@bpy.app.handlers.persistent
def _load_post(_file_path):
	# The engine's state is class level, so it outlives the file it was built
	# for; the new file may well name a different volume.
	VolumeSamplerRenderEngine.reset()


def register():
	bpy.utils.register_class(VolumeSamplerRenderEngine)
	if _load_post not in bpy.app.handlers.load_post:
		bpy.app.handlers.load_post.append(_load_post)
	if not bpy.app.timers.is_registered(_watch_shader_files):
		bpy.app.timers.register(_watch_shader_files, persistent=True)
	if not bpy.app.timers.is_registered(_watch_streaming):
		bpy.app.timers.register(_watch_streaming, persistent=True)
	if not bpy.app.timers.is_registered(_watch_cursor):
		bpy.app.timers.register(_watch_cursor, persistent=True)
	if not bpy.app.timers.is_registered(_watch_view):
		bpy.app.timers.register(_watch_view, persistent=True)


def unregister():
	VolumeSamplerRenderEngine.reset()
	state.close_volume()
	if _load_post in bpy.app.handlers.load_post:
		bpy.app.handlers.load_post.remove(_load_post)
	if bpy.app.timers.is_registered(_watch_view):
		bpy.app.timers.unregister(_watch_view)
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
