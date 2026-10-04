"""Render the active tifxyz surface through its UV map to a metadata PNG."""

import datetime
import os
import tempfile

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from . import bricks
from . import commands
from .surface_output import PNGWriter, linear_rgba_to_srgb8, uv_output_size
from . import uv_renderer
from .renderer import VolumeSamplerRenderEngine as Engine


TILE_WIDTH = 1024
STRIP_HEIGHT = 256


def _surface_info(context, obj, pixel_size_um, positions, uvs, object_matrix):
	shape = tuple(obj.get("velend_tifxyz_source_shape", ()))
	if len(shape) != 2:
		shape = tuple(obj.get("velend_tifxyz_shape", ()))
		if len(shape) != 2:
			shape = commands._surface_shape_from_uvs(obj)
		step = max(1, int(obj.get("velend_tifxyz_step", 1)))
		shape = tuple((int(size) - 1) * step + 1 for size in shape)
	scale = tuple(obj.get("velend_tifxyz_scale", (1.0, 1.0)))
	if len(scale) != 2:
		scale = (1.0, 1.0)
	settings = context.scene.velend
	source_voxel_um = float(obj.get(
		"velend_tifxyz_voxel_size", settings.scene_resolution or settings.resolution
	))
	(width, height), physical_size = uv_output_size(
		positions, uvs, object_matrix, Engine.um_per_unit(), pixel_size_um
	)
	return (
		tuple(int(value) for value in shape), scale, source_voxel_um,
		width, height, physical_size,
	)


def _metadata(
	context, obj, shape, scale, source_voxel_um, width, height, pixel_size_um,
	physical_size,
):
	settings = context.scene.velend
	return {
		"schema": "velend.surface_render.v1",
		"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		"generator": {
			"name": "velend",
			"version": _velend_version(),
			"blender_version": bpy.app.version_string,
		},
		"image": {
			"width_px": width,
			"height_px": height,
			"pixel_size_um": pixel_size_um,
			"physical_size_um": [float(value) for value in physical_size],
			"sizing": "uv_physical_rms_stretch",
			"channels": "RGBA8",
			"uv_bounds": [0.0, 0.0, 1.0, 1.0],
			"volume_level": 0,
		},
		"surface": {
			"object": obj.name,
			"uuid": str(obj.get("velend_tifxyz_uuid", obj.name)),
			"path": str(obj.get("velend_tifxyz_path", "")),
			"grid_shape_yx": list(shape),
			"grid_scale_xy": [float(value) for value in scale],
			"source_voxel_size_um": source_voxel_um,
		},
		"volume": {
			"id": Engine.current_volume_id(),
			"path": Engine.volume_path(),
			"source_url": settings.source_url,
			"voxel_size_um": settings.resolution,
			"scene_volume_id": settings.scene_volume_id,
		},
		"renderer": {
			"render_depth_um": settings.render_depth,
			"samples": settings.num_samples,
			"depth_offset_um": settings.render_depth_offset,
			"invert_sampling_direction": settings.invert_sampling_direction,
			"volumetric_rendering": settings.volumetric_rendering,
			"transmittance_factor": settings.tfactor,
			"gamma": settings.gamma,
			"depth_colors": settings.depth_colors,
			"level_colors": settings.debug_level_colors,
		},
	}


def _velend_version():
	try:
		with open(os.path.join(os.path.dirname(__file__), "blender_manifest.toml")) as file:
			for line in file:
				if line.startswith("version = "):
					return line.split("=", 1)[1].strip().strip('"')
	except OSError:
		pass
	return "unknown"


def _expanded_triangle_chunks(voxels, triangle_mask, padding):
	"""Conservative L0 chunks for selected triangles and their normal samples."""
	triangles = voxels.reshape(-1, 3, 3)[triangle_mask]
	if not len(triangles):
		return np.zeros((0, 3), dtype=np.int64)
	dims = np.asarray(bricks.grid_dims(Engine.shape_xyz), dtype=np.int64)
	lo = np.floor((triangles.min(axis=1) - padding) / bricks.BRICK_CORE).astype(np.int64)
	hi = np.floor((triangles.max(axis=1) + padding) / bricks.BRICK_CORE).astype(np.int64)
	lo = np.clip(lo, 0, dims - 1)
	hi = np.clip(hi, 0, dims - 1)
	spans = hi - lo + 1
	counts = spans.prod(axis=1)
	owner = np.repeat(np.arange(len(counts)), counts)
	starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
	offset = np.arange(int(counts.sum())) - np.repeat(starts, counts)
	span_x = spans[owner, 0]
	span_y = spans[owner, 1]
	chunks = lo[owner] + np.stack((
		offset % span_x,
		(offset // span_x) % span_y,
		offset // (span_x * span_y),
	), axis=1)
	return np.unique(chunks, axis=0)


def _target_chunks(chunks):
	"""Give the shared atlases an exact, capacity-checked export working set."""
	selected = {}
	for level in bricks.LEVELS:
		residency = Engine.residencies[level]
		dims = np.asarray(residency.dims, dtype=np.int64)
		coords = np.unique(np.clip(chunks >> level, 0, dims - 1), axis=0)
		keys = frozenset(int(key) for key in residency.keys_of(coords))
		if len(keys) > bricks.SLOT_COUNTS[level]:
			raise OverflowError("tile needs too many level-%d volume bricks" % level)
		selected[level] = keys

	Engine.wanted = selected
	Engine.pending_levels = {}
	Engine.fallback_keys = {level: set() for level in bricks.LEVELS}
	for level in bricks.LEVELS:
		for key in tuple(Engine.wanted[level] & Engine.empty_chunks[level]):
			Engine.mark_empty(level, key, reschedule=False)
		Engine.residencies[level].evict_outside(Engine.wanted[level])
	wanted = {
		(level, key) for level in bricks.LEVELS for key in Engine.wanted[level]
	}
	Engine.loader.cancel_unwanted(wanted)
	for level in bricks.LEVELS:
		for key, coord in Engine.missing_requests(level):
			Engine.loader.request(level, key, coord)


def _projection(u0, u1, v0, v1):
	du, dv = u1 - u0, v1 - v0
	return np.array([
		[2.0 / du, 0.0, 0.0, -1.0 - 2.0 * u0 / du],
		[0.0, 2.0 / dv, 0.0, -1.0 - 2.0 * v0 / dv],
		[0.0, 0.0, 1.0, 0.0],
		[0.0, 0.0, 0.0, 1.0],
	], dtype=np.float32)


class velend_OT_render_tifxyz(bpy.types.Operator):
	bl_idname = "velend.render_tifxyz"
	bl_label = "Rendered tifxyz Surface"
	bl_description = "Render the active tifxyz mesh's volume texture through its UV map to PNG"
	bl_options = {'REGISTER'}

	filepath: bpy.props.StringProperty(subtype='FILE_PATH', options={'SKIP_SAVE'})
	filter_glob: bpy.props.StringProperty(default="*.png", options={'HIDDEN', 'SKIP_SAVE'})
	pixel_size_um: bpy.props.FloatProperty(
		name="Pixel Size",
		description="Physical width of an output pixel, in micrometers",
		default=9.362,
		min=1e-6,
		soft_max=100.0,
		precision=3,
		options={'SKIP_SAVE'},
	)

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.type == 'MESH'

	def invoke(self, context, event):
		obj = context.active_object
		self.pixel_size_um = context.scene.velend.resolution
		self.filepath = os.path.join(bpy.path.abspath("//"), obj.name + ".png")
		context.window_manager.fileselect_add(self)
		return {'RUNNING_MODAL'}

	def execute(self, context):
		if Engine.exporting:
			self.report({'ERROR'}, "Another tifxyz surface render is already running")
			return {'CANCELLED'}
		path = os.path.expanduser(bpy.path.abspath(self.filepath.strip()))
		if not path:
			self.report({'ERROR'}, "No PNG path selected")
			return {'CANCELLED'}
		if not path.lower().endswith(".png"):
			path += ".png"
		self._path = path
		self._object_name = context.active_object.name
		self._phase = 'SETUP'
		self._timer = context.window_manager.event_timer_add(0.05, window=context.window)
		context.window_manager.modal_handler_add(self)
		context.window_manager.progress_begin(0, 100)
		Engine.exporting = True
		return {'RUNNING_MODAL'}

	def modal(self, context, event):
		if event.type == 'ESC':
			return self._finish(context, error="Surface render cancelled")
		if event.type != 'TIMER':
			return {'PASS_THROUGH'}
		try:
			if self._phase == 'SETUP':
				return self._setup(context)
			if self._phase == 'PREPARE_STRIP':
				self._prepare_strip()
				if self._phase == 'DONE':
					return self._finish(context)
				return {'RUNNING_MODAL'}
			if self._phase == 'START_TILE':
				self._start_tile()
				return {'RUNNING_MODAL'}
			if self._phase == 'WAIT_TILE':
				return self._wait_tile(context)
			if self._phase == 'DONE':
				return self._finish(context)
		except Exception as error:
			return self._finish(context, error="Could not render surface: %s" % error)
		return {'RUNNING_MODAL'}

	def cancel(self, context):
		self._finish(context, error="Surface render cancelled")

	def _setup(self, context):
		obj = bpy.data.objects.get(self._object_name)
		if obj is None or obj.type != 'MESH':
			raise ValueError("the active mesh no longer exists")
		if Engine.pending_reset:
			Engine.apply_reset()
		if Engine.get_volume() is None:
			if Engine.load_future is not None:
				return {'RUNNING_MODAL'}
			raise ValueError(Engine.status()[0])
		Engine.ensure_gpu_resources()
		if Engine.shader is None or not Engine.residencies:
			raise ValueError(Engine.status()[0])
		Engine.world_to_voxels = Engine.compute_world_to_voxels()

		arrays = uv_renderer.mesh_arrays(obj)
		if arrays is None:
			raise ValueError("the active mesh has no UV triangles")
		self._positions, self._normals, self._uvs = arrays
		self._batch = batch_for_shader(
			self._uv_shader(), 'TRIS',
			{'position': self._positions, 'normal': self._normals, 'uv': self._uvs},
		)
		object_matrix = np.asarray(obj.matrix_world, dtype=np.float64)
		world = self._positions @ object_matrix[:3, :3].T
		world += object_matrix[:3, 3]
		self._voxels = Engine.to_voxels(world)
		self._uv_triangles = self._uvs.reshape(-1, 3, 2)
		self._uv_low = self._uv_triangles.min(axis=1)
		self._uv_high = self._uv_triangles.max(axis=1)
		depth = max(
			abs(context.scene.velend.render_depth_offset),
			abs(context.scene.velend.render_depth_offset + context.scene.velend.render_depth),
		)
		self._chunk_padding = depth / (context.scene.velend.resolution or 1.0) + 1.0
		shape, scale, source_um, self._width, self._height, physical_size = _surface_info(
			context, obj, self.pixel_size_um, self._positions, self._uvs, object_matrix
		)
		metadata = _metadata(
			context, obj, shape, scale, source_um, self._width, self._height,
			self.pixel_size_um, physical_size,
		)
		parent = os.path.dirname(self._path) or "."
		if not os.path.isdir(parent):
			raise ValueError("the output directory does not exist")
		temp = tempfile.NamedTemporaryFile(
			prefix=".%s." % os.path.basename(self._path), suffix=".tmp", dir=parent,
			delete=False,
		)
		self._temp_path = temp.name
		temp.close()
		self._writer = PNGWriter(self._temp_path, self._width, self._height, metadata)
		self._next_y = 0
		self._strip_height = min(STRIP_HEIGHT, self._height)
		self._phase = 'PREPARE_STRIP'
		return {'RUNNING_MODAL'}

	def _uv_shader(self):
		path = os.path.join(os.path.dirname(__file__), 'shaders', 'volume.frag')
		with open(path, encoding='utf-8') as source:
			self._shader = Engine.create_shader(uv_renderer._VERTEX_SOURCE, source.read(), uv=True)
		return self._shader

	def _tile_chunks(self, x0, x1, y0, y1):
		u0, u1 = x0 / self._width, x1 / self._width
		v0, v1 = 1.0 - y1 / self._height, 1.0 - y0 / self._height
		mask = (
			(self._uv_high[:, 0] >= u0) & (self._uv_low[:, 0] <= u1)
			& (self._uv_high[:, 1] >= v0) & (self._uv_low[:, 1] <= v1)
		)
		chunks = _expanded_triangle_chunks(self._voxels, mask, self._chunk_padding)
		return (x0, x1, y0, y1, u0, u1, v0, v1, chunks)

	def _prepare_strip(self):
		if self._next_y >= self._height:
			self._writer.close()
			os.replace(self._temp_path, self._path)
			self._temp_path = None
			self._phase = 'DONE'
			return
		y0 = self._next_y
		height = min(self._strip_height, self._height - y0)
		while True:
			tiles = []
			x0 = 0
			failed = False
			while x0 < self._width:
				width = min(TILE_WIDTH, self._width - x0)
				while True:
					tile = self._tile_chunks(x0, x0 + width, y0, y0 + height)
					if len(tile[-1]) <= bricks.SLOT_COUNTS[0]:
						break
					if width == 1:
						failed = True
						break
					width = max(1, width // 2)
				if failed:
					break
				tiles.append(tile)
				x0 += width
			if not failed:
				break
			if height == 1:
				raise ValueError("one output pixel crosses more bricks than the atlas can hold")
			height = max(1, height // 2)
		self._tiles = tiles
		self._tile_index = 0
		self._strip = np.zeros((height, self._width, 4), dtype=np.uint8)
		self._phase = 'START_TILE'

	def _start_tile(self):
		if self._tile_index >= len(self._tiles):
			self._writer.write_rows(self._strip)
			self._next_y += len(self._strip)
			self._phase = 'PREPARE_STRIP'
			return
		self._tile = self._tiles[self._tile_index]
		_target_chunks(self._tile[-1])
		self._phase = 'WAIT_TILE'

	def _wait_tile(self, context):
		x0, x1, y0, y1, u0, u1, v0, v1, chunks = self._tile
		width, height = x1 - x0, y1 - y0
		offscreen = gpu.types.GPUOffScreen(width, height, format='RGBA16F')
		try:
			with offscreen.bind():
				Engine.pump_uploads()
				if Engine.loader.error:
					raise OSError(Engine.loader.error)
				missing = any(Engine.missing_requests(level) for level in bricks.LEVELS)
				if missing or Engine.pending_levels or not Engine.loader.idle():
					return {'RUNNING_MODAL'}
				self._draw_tile(context, u0, u1, v0, v1, width, height)
				framebuffer = gpu.state.active_framebuffer_get()
				buffer = framebuffer.read_color(0, 0, width, height, 4, 0, 'FLOAT')
				buffer.dimensions = width * height * 4
				linear = np.frombuffer(
					bytearray(buffer), dtype=np.float32
				).reshape(height, width, 4)[::-1]
				pixels = linear_rgba_to_srgb8(linear)
		finally:
			offscreen.free()
		self._strip[:, x0:x1] = pixels
		self._tile_index += 1
		progress = 100.0 * (y0 * self._width + x1) / (self._width * self._height)
		context.window_manager.progress_update(progress)
		self._phase = 'START_TILE'
		return {'RUNNING_MODAL'}

	def _draw_tile(self, context, u0, u1, v0, v1, width, height):
		framebuffer = gpu.state.active_framebuffer_get()
		framebuffer.clear(color=(0.0, 0.0, 0.0, 0.0))
		gpu.state.viewport_set(0, 0, width, height)
		gpu.state.blend_set('NONE')
		gpu.state.depth_test_set('NONE')
		gpu.state.depth_mask_set(False)
		self._shader.bind()
		settings = context.scene.velend
		self._shader.uniform_float('gamma', settings.gamma)
		if settings.volumetric_rendering:
			self._shader.uniform_float('tfactor', settings.tfactor)
		for level in bricks.LEVELS:
			self._shader.uniform_sampler('l%dAtlas' % level, Engine.atlases[level].texture)
			self._shader.uniform_sampler(
				'l%dPageTable' % level, Engine.atlases[level].page_texture
			)
		obj = bpy.data.objects[self._object_name]
		Engine.update_uniform_buffer(_projection(u0, u1, v0, v1), obj.matrix_world)
		self._shader.uniform_block('volumeUniforms', Engine.uniform_buffer)
		self._shader.uniform_float('uvOpacity', 1.0)
		self._batch.draw(self._shader)

	def _finish(self, context, error=None):
		if getattr(self, "_timer", None) is not None:
			context.window_manager.event_timer_remove(self._timer)
			self._timer = None
		context.window_manager.progress_end()
		writer = getattr(self, "_writer", None)
		if error and writer is not None:
			writer.abort()
		temp_path = getattr(self, "_temp_path", None)
		if temp_path and os.path.exists(temp_path):
			try:
				os.unlink(temp_path)
			except OSError:
				pass
		Engine.exporting = False
		Engine.retarget_now()
		if error:
			print("velend:", error)
			self.report({'ERROR'}, error)
			return {'CANCELLED'}
		self.report({'INFO'}, "Rendered tifxyz surface to %s" % self._path)
		return {'FINISHED'}


def _export_menu(self, context):
	self.layout.operator(
		velend_OT_render_tifxyz.bl_idname, text="Rendered tifxyz Surface (.png)"
	)


def register():
	bpy.utils.register_class(velend_OT_render_tifxyz)
	bpy.types.TOPBAR_MT_file_export.append(_export_menu)


def unregister():
	bpy.types.TOPBAR_MT_file_export.remove(_export_menu)
	bpy.utils.unregister_class(velend_OT_render_tifxyz)
