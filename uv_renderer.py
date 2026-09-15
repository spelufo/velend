"""Sample shared volume atlases on the active mesh's UV triangles."""

import os
import bmesh
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from . import bricks
from .renderer import VolumeSamplerRenderEngine as Engine

_handler = None
_shader = None
_source_shader = None
_batch = None
_mesh_key = None
_dirty = True
_keymaps = []

_VERTEX_SOURCE = """
void main() {
    vec4 world = volumeUniforms.modelMatrix * vec4(position, 1.0);
    mat3 localToVoxels = mat3(volumeUniforms.worldToVoxels * volumeUniforms.modelMatrix);
    voxelCoord = (volumeUniforms.worldToVoxels * vec4(world.xyz, 1.0)).xyz;
    voxelNormal = normalize(transpose(inverse(localToVoxels)) * normal);
    gl_Position = volumeUniforms.viewProjectionMatrix * vec4(uv, 0.0, 1.0);
    // Behind UV edges/vertices, in front of the image.
    gl_Position.z = 0.5 * gl_Position.w;
}
"""


def point_on_uv_mesh(obj, uv):
	"""The world-space point under a UV coordinate, or None outside the mesh."""
	arrays = mesh_arrays(obj)
	if arrays is None:
		return None
	positions, _normals, uvs = arrays
	for index in range(0, len(uvs), 3):
		triangle = uvs[index:index + 3]
		a, b, c = triangle
		denominator = (b[1] - c[1]) * (a[0] - c[0]) + (
			c[0] - b[0]) * (a[1] - c[1]
		)
		if abs(denominator) < 1e-12:
			continue
		wa = ((b[1] - c[1]) * (uv[0] - c[0]) +
		(c[0] - b[0]) * (uv[1] - c[1])) / denominator
		wb = ((c[1] - a[1]) * (uv[0] - c[0]) +
		(a[0] - c[0]) * (uv[1] - c[1])) / denominator
		wc = 1.0 - wa - wb
		if min(wa, wb, wc) >= -1e-6:
			local = (
				wa * positions[index]
				+ wb * positions[index + 1]
				+ wc * positions[index + 2]
			)
			return obj.matrix_world @ Vector(local)
	return None


class velend_OT_cursor_from_uv(bpy.types.Operator):
	bl_idname = "velend.cursor_from_uv"
	bl_label = "Set 3D Cursor from UV"
	bl_description = "Place the 3D cursor on the mesh below this UV coordinate"
	bl_options = {'INTERNAL'}

	location: bpy.props.FloatVectorProperty(size=2, options={'SKIP_SAVE', 'HIDDEN'})

	@classmethod
	def poll(cls, context):
		space = context.space_data
		obj = context.active_object
		return (
			space is not None and space.type == 'IMAGE_EDITOR' and space.mode == 'UV'
			and obj is not None and obj.type == 'MESH'
		)

	def invoke(self, context, event):
		self.location = context.region.view2d.region_to_view(
			event.mouse_region_x, event.mouse_region_y
		)
		return self.execute(context)

	def execute(self, context):
		point = point_on_uv_mesh(context.active_object, self.location)
		if point is None:
			self.report({'WARNING'}, "No UV face under the cursor")
			return {'CANCELLED'}
		context.space_data.cursor_location = self.location
		context.scene.cursor.location = point
		Engine.retarget_now()
		return {'FINISHED'}


class IMAGE_PT_velend(bpy.types.Panel):
	bl_label = "Velend"
	bl_space_type = 'IMAGE_EDITOR'
	bl_region_type = 'UI'
	bl_category = "Velend"

	@classmethod
	def poll(cls, context):
		return context.space_data.mode == 'UV'

	def draw(self, context):
		self.layout.prop(context.scene.velend, "uv_volume_rendering")
		self.layout.prop(context.scene.velend, "uv_transparency")


def mesh_arrays(obj):
	"""Read corner attributes for seams, including live edits from BMesh."""
	if obj.mode == 'EDIT':
		bm = bmesh.from_edit_mesh(obj.data)
		layer = bm.loops.layers.uv.active
		if layer is None:
			return None
		triangles = [tri for tri in bm.calc_loop_triangles() if not tri[0].face.hide]
		if not triangles:
			return None
		return (
			np.asarray([loop.vert.co[:] for tri in triangles for loop in tri], dtype='f'),
			np.asarray([loop.vert.normal[:] for tri in triangles for loop in tri], dtype='f'),
			np.asarray([loop[layer].uv[:] for tri in triangles for loop in tri], dtype='f'),
		)
	mesh = obj.data
	layer = mesh.uv_layers.active
	if layer is None:
		return None
	mesh.calc_loop_triangles()
	if not mesh.loop_triangles:
		return None
	positions = np.empty((len(mesh.vertices), 3), dtype='f')
	normals = np.empty_like(positions)
	uvs = np.empty((len(mesh.loops), 2), dtype='f')
	vertices = np.empty((len(mesh.loop_triangles), 3), dtype='i')
	loops = np.empty_like(vertices)
	mesh.vertices.foreach_get('co', positions.ravel())
	mesh.vertices.foreach_get('normal', normals.ravel())
	layer.data.foreach_get('uv', uvs.ravel())
	mesh.loop_triangles.foreach_get('vertices', vertices.ravel())
	mesh.loop_triangles.foreach_get('loops', loops.ravel())
	return positions[vertices.ravel()], normals[vertices.ravel()], uvs[loops.ravel()]


def _draw():
	global _shader, _source_shader, _batch, _mesh_key, _dirty
	context = bpy.context
	space = context.space_data
	obj = context.active_object
	if (space is None or space.type != 'IMAGE_EDITOR' or space.mode != 'UV'
			or obj is None or obj.type != 'MESH' or not obj.data.uv_layers.active
			or not context.scene.velend.volume_path.strip()
			or not context.scene.velend.uv_volume_rendering):
		return
	if Engine.pending_reset:
		Engine.apply_reset()
	first_init = not Engine.residencies
	Engine.ensure_gpu_resources()
	if Engine.shader is None or not Engine.residencies:
		return
	if first_init or _dirty:
		Engine.meshes.clear()
		Engine.retarget(context.evaluated_depsgraph_get(), context.scene.cursor.location)
	Engine.pump_uploads()
	if _source_shader is not Engine.shader:
		path = os.path.join(os.path.dirname(__file__), 'shaders', 'volume.frag')
		with open(path, encoding='utf-8') as source:
			_shader = Engine.create_shader(_VERTEX_SOURCE, source.read(), uv=True)
		_source_shader = Engine.shader
		_mesh_key = None
	key = (obj.as_pointer(), obj.data.as_pointer(), obj.mode, obj.data.uv_layers.active.name)
	if _dirty or key != _mesh_key:
		arrays = mesh_arrays(obj)
		_batch = None if arrays is None else batch_for_shader(
			_shader, 'TRIS', {'position': arrays[0], 'normal': arrays[1], 'uv': arrays[2]},
		)
		_mesh_key = key
		_dirty = False
	if _batch is None:
		return

	blend = gpu.state.blend_get()
	depth_test = gpu.state.depth_test_get()
	depth_mask = gpu.state.depth_mask_get()
	try:
		gpu.state.blend_set('ALPHA')
		gpu.state.depth_test_set('LESS_EQUAL')
		gpu.state.depth_mask_set(False)
		_shader.bind()
		_shader.uniform_float('gamma', context.scene.velend.gamma)
		for level in bricks.LEVELS:
			_shader.uniform_sampler('l%dAtlas' % level, Engine.atlases[level].texture)
			_shader.uniform_sampler('l%dPageTable' % level, Engine.atlases[level].page_texture)
		Engine.update_uniform_buffer(gpu.matrix.get_projection_matrix(), obj.matrix_world)
		_shader.uniform_block('volumeUniforms', Engine.uniform_buffer)
		_shader.uniform_float('uvOpacity', 1.0 - context.scene.velend.uv_transparency)
		_batch.draw(_shader)
	finally:
		gpu.state.depth_mask_set(depth_mask)
		gpu.state.depth_test_set(depth_test)
		gpu.state.blend_set(blend)


@bpy.app.handlers.persistent
def _invalidate(*_args):
	global _dirty
	_dirty = True
	_redraw()


def _redraw():
	for window in bpy.context.window_manager.windows:
		for area in window.screen.areas:
			if area.type == 'IMAGE_EDITOR':
				area.tag_redraw()


def register():
	global _handler
	bpy.utils.register_class(velend_OT_cursor_from_uv)
	bpy.utils.register_class(IMAGE_PT_velend)
	if _handler is None:
		_handler = bpy.types.SpaceImageEditor.draw_handler_add(_draw, (), 'WINDOW', 'POST_VIEW')
	keyconfig = bpy.context.window_manager.keyconfigs.addon
	if keyconfig is not None:
		# Blender's UV Editor keymap is disabled outside Edit Mode. Image Generic
		# still has the operator poll below to keep this binding specific to UVs.
		keymap = keyconfig.keymaps.new(name='Image Generic', space_type='IMAGE_EDITOR')
		item = keymap.keymap_items.new(
			velend_OT_cursor_from_uv.bl_idname, 'RIGHTMOUSE', 'PRESS', shift=True
		)
		_keymaps.append((keymap, item))
	for handlers in (bpy.app.handlers.depsgraph_update_post, bpy.app.handlers.load_post):
		if _invalidate not in handlers:
			handlers.append(_invalidate)
	_invalidate()


def unregister():
	global _handler, _shader, _source_shader, _batch, _mesh_key
	for keymap, item in _keymaps:
		keymap.keymap_items.remove(item)
	_keymaps.clear()
	for handlers in (bpy.app.handlers.depsgraph_update_post, bpy.app.handlers.load_post):
		if _invalidate in handlers:
			handlers.remove(_invalidate)
	if _handler is not None:
		bpy.types.SpaceImageEditor.draw_handler_remove(_handler, 'WINDOW')
		_handler = None
	_batch = _shader = _source_shader = _mesh_key = None
	_redraw()
	bpy.utils.unregister_class(IMAGE_PT_velend)
	bpy.utils.unregister_class(velend_OT_cursor_from_uv)
