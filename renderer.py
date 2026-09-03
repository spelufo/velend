"""
Render Engine sampling a 3D Texture
+++++++++++++++++++++++++++++++++++

A viewport render engine that shades the meshes of the scene by sampling a 3D texture
(a ``FLOAT_3D`` sampler) filled with random values, with the world position of the surface.
The texture covers the ``[0, 1]`` region of the scene on each axis and geometry outside of
that region samples the clamped edge values.

Unlike a :class:`bpy.types.SpaceView3D` draw handler, the engine owns the color pass, so the
depth values it writes are not in conflict with another engine drawing the same geometry.
Blender draws the background and the overlays around the result.

Both ``view_update`` and ``view_draw`` are required for an engine to draw the viewport.
Only viewport drawing is implemented here, a final render needs a ``render`` method.

To try it out, select "Volume Sampler" in the Render properties
and set the viewport shading to Rendered.
"""

import bpy
import numpy as np
from random import random

SIZE = 16

# The GPU resources are created on the first draw, where a GPU context is guaranteed to be
# active, and shared by all the engine instances (Blender creates one per viewport).
volume = None
shader = None

# Batches are rebuilt only when the geometry of the scene changes.
batches = {}


def ensure_gpu_resources():
	global volume, shader

	if shader is not None:
		return

	import gpu

	volume = gpu.types.GPUTexture(
		(SIZE, SIZE, SIZE),
		format='RGBA32F',
		data=gpu.types.Buffer('FLOAT', SIZE ** 3 * 4, [random() for _ in range(SIZE ** 3 * 4)]),
	)
	# Trilinear interpolation between the random texels.
	volume.filter_mode(True)

	vert_out = gpu.types.GPUStageInterfaceInfo("volume_interface")
	vert_out.smooth('VEC3', "volumeCoord")

	shader_info = gpu.types.GPUShaderCreateInfo()
	shader_info.push_constant('MAT4', "viewProjectionMatrix")
	shader_info.push_constant('MAT4', "modelMatrix")
	shader_info.sampler(0, 'FLOAT_3D', "volume")
	shader_info.vertex_in(0, 'VEC3', "position")
	shader_info.vertex_out(vert_out)
	shader_info.fragment_out(0, 'VEC4', "FragColor")

	shader_info.vertex_source("""
void main()
{
  vec4 worldPosition = modelMatrix * vec4(position, 1.0f);
  volumeCoord = worldPosition.xyz;
  gl_Position = viewProjectionMatrix * worldPosition;
}
""")

	shader_info.fragment_source("""
void main()
{
  FragColor = texture(volume, volumeCoord);
}
""")

	shader = gpu.shader.create_from_info(shader_info)


def batch_from_object(obj_eval):
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
		batch = batch_for_shader(shader, 'TRIS', {"position": positions}, indices=indices)
	obj_eval.to_mesh_clear()
	return batch


class VolumeSamplerRenderEngine(bpy.types.RenderEngine):
	bl_idname = "VOLUME_SAMPLER"
	bl_label = "Volume Sampler"

	def view_update(self, context, depsgraph):
		# This runs outside of the drawing code, without an active GPU context,
		# so it only invalidates the caches used by `view_draw`.
		if any(update.is_updated_geometry for update in depsgraph.updates):
			batches.clear()

	def view_draw(self, context, depsgraph):
		import gpu

		ensure_gpu_resources()

		# Blender clears the color and depth buffers before the engine draws. Writing depth
		# is what lets it occlude the overlays that are drawn on top of the result.
		gpu.state.depth_test_set('LESS_EQUAL')
		gpu.state.depth_mask_set(True)

		shader.uniform_sampler("volume", volume)
		shader.uniform_float("viewProjectionMatrix", context.region_data.perspective_matrix)

		# Iterating the instances also draws the duplis and the geometry nodes instances,
		# which share the batch of the object they instance.
		for instance in depsgraph.object_instances:
			obj = instance.object
			if obj.type != 'MESH':
				continue
			if obj.name not in batches:
				batches[obj.name] = batch_from_object(obj)
			batch = batches[obj.name]
			if batch is None:
				continue
			shader.uniform_float("modelMatrix", instance.matrix_world)
			batch.draw(shader)

		gpu.state.depth_mask_set(False)
		gpu.state.depth_test_set('NONE')


def register():
	bpy.utils.register_class(VolumeSamplerRenderEngine)


def unregister():
	bpy.utils.unregister_class(VolumeSamplerRenderEngine)


if __name__ == "__main__":
	register()
