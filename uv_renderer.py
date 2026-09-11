"""Sample shared volume atlases on the active mesh's UV triangles."""

import os
import bmesh
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from . import bricks
from .renderer import VolumeSamplerRenderEngine as Engine

_handler = None
_shader = None
_source_shader = None
_batch = None
_mesh_key = None
_dirty = True

_VERTEX_SOURCE = """
void main() {
    vec4 world = volumeUniforms.modelMatrix * vec4(position, 1.0);
    voxelCoord = (volumeUniforms.volumeTransform *
        vec4(world.xyz * volumeUniforms.voxelsPerUnit, 1.0)).xyz;
    gl_Position = volumeUniforms.viewProjectionMatrix * vec4(uv, 0.0, 1.0);
    // Behind UV edges/vertices, in front of the image.
    gl_Position.z = 0.5 * gl_Position.w;
}
"""


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
	uvs = np.empty((len(mesh.loops), 2), dtype='f')
	vertices = np.empty((len(mesh.loop_triangles), 3), dtype='i')
	loops = np.empty_like(vertices)
	mesh.vertices.foreach_get('co', positions.ravel())
	layer.data.foreach_get('uv', uvs.ravel())
	mesh.loop_triangles.foreach_get('vertices', vertices.ravel())
	mesh.loop_triangles.foreach_get('loops', loops.ravel())
	return positions[vertices.ravel()], uvs[loops.ravel()]


def _draw():
	global _shader, _source_shader, _batch, _mesh_key, _dirty
	context = bpy.context
	space = context.space_data
	obj = context.active_object
	if (space is None or space.type != 'IMAGE_EDITOR' or space.mode != 'UV'
			or obj is None or obj.type != 'MESH' or not obj.data.uv_layers.active
			or not context.scene.velend.volume_path.strip()):
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
			_shader, 'TRIS', {'position': arrays[0], 'uv': arrays[1]},
		)
		_mesh_key = key
		_dirty = False
	if _batch is None:
		return

	blend = gpu.state.blend_get()
	depth_test = gpu.state.depth_test_get()
	depth_mask = gpu.state.depth_mask_get()
	try:
		gpu.state.blend_set('NONE')
		gpu.state.depth_test_set('LESS_EQUAL')
		gpu.state.depth_mask_set(False)
		_shader.bind()
		for level in bricks.LEVELS:
			_shader.uniform_sampler('l%dAtlas' % level, Engine.atlases[level].texture)
			_shader.uniform_sampler('l%dPageTable' % level, Engine.atlases[level].page_texture)
		Engine.update_uniform_buffer(gpu.matrix.get_projection_matrix(), obj.matrix_world)
		_shader.uniform_block('volumeUniforms', Engine.uniform_buffer)
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
	if _handler is None:
		_handler = bpy.types.SpaceImageEditor.draw_handler_add(_draw, (), 'WINDOW', 'POST_VIEW')
	for handlers in (bpy.app.handlers.depsgraph_update_post, bpy.app.handlers.load_post):
		if _invalidate not in handlers:
			handlers.append(_invalidate)
	_invalidate()


def unregister():
	global _handler, _shader, _source_shader, _batch, _mesh_key
	for handlers in (bpy.app.handlers.depsgraph_update_post, bpy.app.handlers.load_post):
		if _invalidate in handlers:
			handlers.remove(_invalidate)
	if _handler is not None:
		bpy.types.SpaceImageEditor.draw_handler_remove(_handler, 'WINDOW')
		_handler = None
	_batch = _shader = _source_shader = _mesh_key = None
	_redraw()
