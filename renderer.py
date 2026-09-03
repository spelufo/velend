import os
import bpy
import gpu
import numpy as np
from random import random

from . import once


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
	shader = None
	shader_mtimes = None
	volume_dims = (1.0, 1.0, 1.0)
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

		volume = once.get_volume()
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

		shader_info = gpu.types.GPUShaderCreateInfo()
		shader_info.push_constant('MAT4', "viewProjectionMatrix")
		shader_info.push_constant('MAT4', "modelMatrix")
		shader_info.push_constant('VEC3', "volumeScale")
		shader_info.sampler(0, 'FLOAT_3D', "volume")
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
		if self.shader is None:
			return

		# Blender clears the color and depth buffers before the engine draws. Writing depth
		# is what lets it occlude the overlays that are drawn on top of the result.
		gpu.state.depth_test_set('LESS_EQUAL')
		gpu.state.depth_mask_set(True)

		self.shader.uniform_sampler("volume", self.volume)
		self.shader.uniform_float("viewProjectionMatrix", context.region_data.perspective_matrix)
		self.shader.uniform_float("volumeScale", self.volume_dims)

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
			self.shader.uniform_float("modelMatrix", instance.matrix_world)
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
