import os
import bpy
import gpu
import numpy as np
from random import random

from . import state


_SHADERS_DIR = os.path.join(os.path.dirname(__file__), "shaders")
_VERT_SHADER_PATH = os.path.join(_SHADERS_DIR, "volume.vert")
_FRAG_SHADER_PATH = os.path.join(_SHADERS_DIR, "volume.frag")
_SHADER_WATCH_INTERVAL = 0.25


class VolumeSamplerRenderEngine(bpy.types.RenderEngine):
	bl_idname = "VOLUME_SAMPLER"
	bl_label = "Volume Sampler"

	# Created on the first draw, where a GPU context is guaranteed to be active,
	# and shared by all engine instances (Blender creates one per viewport).
	volume = None
	volume_hires = None
	shader = None
	shader_mtimes = None
	uniform_buffer = None
	volume_dims = (1.0, 1.0, 1.0)
	hires_origin = (0.0, 0.0, 0.0)
	hires_scale = (1.0, 1.0, 1.0)
	hires_window = None
	# Rebuilt only when the geometry of the scene changes.
	batches = {}

	@classmethod
	def shader_file_mtimes(cls):
		return (
			os.stat(_VERT_SHADER_PATH).st_mtime_ns,
			os.stat(_FRAG_SHADER_PATH).st_mtime_ns,
		)

	@classmethod
	def shaders_changed(cls):
		try:
			return cls.shader_file_mtimes() != cls.shader_mtimes
		except OSError:
			return False

	@classmethod
	def tag_viewports(cls):
		wm = bpy.context.window_manager
		if wm is None:
			return
		for window in wm.windows:
			if window.scene.render.engine != cls.bl_idname:
				continue
			for area in window.screen.areas:
				if area.type != 'VIEW_3D':
					continue
				if area.spaces.active.shading.type != 'RENDERED':
					continue
				area.tag_redraw()

	@classmethod
	def ensure_volume(cls):
		if cls.volume is not None:
			return

		volume = state.get_volume()
		downsample = 5
		# Full-res voxels are `volume.resolution` µm; each lores voxel covers 2^downsample of them.
		lores_resolution = volume.resolution * (1 << downsample)
		lores = np.ascontiguousarray(volume[:, :, :, downsample], dtype=np.float32)
		lores *= np.float32(1.0 / 255.0)

		# Numpy is C-order (Z, Y, X); GPUTexture is (width, height, depth) with width fastest.
		dims = tuple(reversed(lores.shape))
		unit_scale = bpy.context.scene.unit_settings.scale_length
		voxel_size = lores_resolution / (1000000.0 * unit_scale)
		cls.volume_dims = tuple(d * voxel_size for d in dims)
		print("Volume scale: ", cls.volume_dims)

		cls.volume = gpu.types.GPUTexture(
			dims,
			format='R8',
			data=gpu.types.Buffer('FLOAT', lores.size, lores),
		)
		# Trilinear interpolation between the random texels.
		cls.volume.filter_mode(True)

	@classmethod
	def ensure_hires(cls, cursor_location):
		volume = state.get_volume()
		hires_size = 1024
		unit_scale = bpy.context.scene.unit_settings.scale_length
		voxel_size_hires = volume.resolution / (1000000.0 * unit_scale)
		shape_zyx = np.asarray(volume.shape(0), dtype=np.int64)
		shape_xyz = shape_zyx[::-1]
		center_xyz = np.asarray(cursor_location, dtype=np.float64) / voxel_size_hires
		imin = np.rint(center_xyz).astype(np.int64) - (hires_size // 2)
		imin = np.clip(imin, 0, np.maximum(shape_xyz - hires_size, 0))
		imax = np.minimum(imin + hires_size, shape_xyz)
		window = tuple(int(v) for v in (*imin, *imax))
		# print(window)
		if cls.volume_hires is not None and window == cls.hires_window:
			# print('im out')
			return

		xmin, ymin, zmin, xmax, ymax, zmax = window
		# Numpy is (Z, Y, X); level 0 is full resolution.
		hires = np.ascontiguousarray(volume[zmin:zmax, ymin:ymax, xmin:xmax, 0], dtype=np.float32)
		hires *= np.float32(1.0 / 255.0)
		hires_dims = tuple(reversed(hires.shape))
		cls.hires_origin = (xmin * voxel_size_hires, ymin * voxel_size_hires, zmin * voxel_size_hires)
		cls.hires_scale = tuple(d * voxel_size_hires for d in hires_dims)
		cls.hires_window = window
		print("Hires origin, scale: ", cls.hires_origin, cls.hires_scale)

		cls.volume_hires = gpu.types.GPUTexture(
			hires_dims,
			format='R8',
			data=gpu.types.Buffer('FLOAT', hires.size, hires),
		)
		cls.volume_hires.filter_mode(True)

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
		vert_out.smooth('VEC3', "volumeCoord")
		vert_out.smooth('VEC3', "hiresCoord")

		shader_info = gpu.types.GPUShaderCreateInfo()
		shader_info.typedef_source("""
			struct VolumeUniforms {
				mat4 viewProjectionMatrix;
				mat4 modelMatrix;
				vec3 volumeScale;
				vec3 hiresOrigin;
				vec3 hiresScale;
			};
		""")
		shader_info.uniform_buf(0, "VolumeUniforms", "volumeUniforms")
		shader_info.sampler(0, 'FLOAT_3D', "volume")
		shader_info.sampler(1, 'FLOAT_3D', "volumeHires")
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
		print("Loaded shaders")

	@classmethod
	def ensure_gpu_resources(cls):
		cls.ensure_volume()
		cls.ensure_shader()

	@classmethod
	def update_uniform_buffer(cls, view_projection_matrix, model_matrix):
		# std140 lays out each mat4 as four vec4 columns. `vec3` has a 16-byte stride.
		data = np.empty(44, dtype=np.float32)
		data[:16] = np.asarray(view_projection_matrix, dtype=np.float32).T.ravel()
		data[16:32] = np.asarray(model_matrix, dtype=np.float32).T.ravel()
		data[32:35] = cls.volume_dims
		data[35] = 0.0
		data[36:39] = cls.hires_origin
		data[39] = 0.0
		data[40:43] = cls.hires_scale
		data[43] = 0.0
		buffer = gpu.types.Buffer('FLOAT', len(data), data)
		if cls.uniform_buffer is None:
			cls.uniform_buffer = gpu.types.GPUUniformBuf(buffer)
		else:
			cls.uniform_buffer.update(buffer)

	@classmethod
	def batch_from_object(cls, obj_eval):
		from gpu_extras.batch import batch_for_shader

		mesh = obj_eval.to_mesh()
		mesh.calc_loop_triangles()
		batch = None
		if mesh.loop_triangles:
			# `foreach_get` copies whole attributes at once, and buffers supporting the Python
			# buffer protocol are uploaded to the vertex buffer without a per element conversion.
			positions = np.empty((len(mesh.vertices), 3), 'f')
			indices = np.empty((len(mesh.loop_triangles), 3), 'i')
			mesh.vertices.foreach_get("co", np.reshape(positions, len(mesh.vertices) * 3))
			mesh.loop_triangles.foreach_get("vertices", np.reshape(indices, len(mesh.loop_triangles) * 3))
			batch = batch_for_shader(cls.shader, 'TRIS', {"position": positions}, indices=indices)
		obj_eval.to_mesh_clear()
		return batch

	def view_update(self, context, depsgraph):
		# This runs outside of the drawing code, without an active GPU context,
		# so it only invalidates the caches used by `view_draw`.
		if any(update.is_updated_geometry for update in depsgraph.updates):
			self.batches.clear()

	def view_draw(self, context, depsgraph):
		self.ensure_gpu_resources()
		self.ensure_hires(context.scene.cursor.location)
		if self.shader is None:
			return

		# Blender clears the color and depth buffers before the engine draws. Writing depth
		# is what lets it occlude the overlays that are drawn on top of the result.
		gpu.state.depth_test_set('LESS_EQUAL')
		gpu.state.depth_mask_set(True)

		self.shader.uniform_sampler("volume", self.volume)
		self.shader.uniform_sampler("volumeHires", self.volume_hires)

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
		VolumeSamplerRenderEngine.tag_viewports()
	return _SHADER_WATCH_INTERVAL


def register():
	bpy.utils.register_class(VolumeSamplerRenderEngine)
	if not bpy.app.timers.is_registered(_watch_shader_files):
		bpy.app.timers.register(_watch_shader_files, persistent=True)


def unregister():
	if bpy.app.timers.is_registered(_watch_shader_files):
		bpy.app.timers.unregister(_watch_shader_files)
	bpy.utils.unregister_class(VolumeSamplerRenderEngine)


if __name__ == "__main__":
	register()
