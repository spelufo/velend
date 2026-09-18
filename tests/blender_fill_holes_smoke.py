"""Run with Blender --background --factory-startup --python-exit-code 1 --python."""

import os
import sys
import tempfile
from pathlib import Path

if os.environ.get('VELEND_TEST_DEPS'):
	sys.path.insert(0, os.environ['VELEND_TEST_DEPS'])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bmesh
import bpy
import numpy as np

import velend
from velend import commands, tifxyz

for module in velend._modules:
	if hasattr(module, 'register'):
		module.register()


def make_mesh():
	mesh = bpy.data.meshes.new('hole grid')
	vertices = [(float(row), float(col), float(row + col))
		for row in range(5) for col in range(5)]
	faces = [(row * 5 + col, (row + 1) * 5 + col,
		(row + 1) * 5 + col + 1, row * 5 + col + 1)
		for row in range(4) for col in range(4)]
	mesh.from_pydata(vertices, [], faces)
	mesh.update()
	layer = mesh.uv_layers.new(name='UVMap')
	for polygon in mesh.polygons:
		for loop_index in polygon.loop_indices:
			index = mesh.loops[loop_index].vertex_index
			row, col = divmod(index, 5)
			layer.data[loop_index].uv = (1 - col / 4, row / 4)
	attribute = mesh.attributes.new('score', 'FLOAT', 'POINT')
	for index, point in enumerate(attribute.data):
		row, col = divmod(index, 5)
		point.value = 2 * row - col
	obj = bpy.data.objects.new('hole grid', mesh)
	bpy.context.scene.collection.objects.link(obj)
	bpy.context.view_layer.objects.active = obj
	obj.select_set(True)
	return obj


obj = make_mesh()
bm = bmesh.new()
bm.from_mesh(obj.data)
for face in list(bm.faces):
	center = sum(vert.co.x for vert in face.verts) / 4
	center_col = sum(vert.co.y for vert in face.verts) / 4
	if 1 < center < 3 and 1 < center_col < 3:
		bm.faces.remove(face)
for vert in list(bm.verts):
	if tuple(vert.co) == (2.0, 2.0, 4.0):
		bm.verts.remove(vert)
bm.to_mesh(obj.data)
bm.free()

assert bpy.ops.velend.fill_tifxyz_holes() == {'FINISHED'}
uvs, faces = commands._mesh_uvs_faces(obj.data)
grid = tifxyz.grid_from_uvs(uvs, faces)
assert grid.shape == (5, 5)
center_index = int(grid[2, 2])
assert np.allclose(obj.data.vertices[center_index].co, (2, 2, 4))
assert abs(obj.data.attributes['score'].data[center_index].value - 2) < 1e-5
assert np.allclose(uvs[center_index], (0.5, 0.5))
assert len(obj.data.polygons) == 16
assert all(polygon.normal.z > 0 for polygon in obj.data.polygons)
_, mask, _ = commands._completed_mesh_arrays(obj.data, obj)
assert np.all(mask == 1)
with tempfile.TemporaryDirectory() as directory:
	obj['velend_tifxyz_placement'] = np.eye(4).ravel().tolist()
	assert bpy.ops.velend.export_tifxyz(
		filepath=str(Path(directory) / 'surface'), uuid='filled',
		scale_x=1, scale_y=1, voxel_size=1,
	) == {'FINISHED'}
	assert np.all(tifxyz.read_page(Path(directory) / 'surface' / 'mask.tif') == 255)

bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)
face = next(face for face in bm.faces if all(
	vert.co.x in (1, 2) and vert.co.y in (1, 2) for vert in face.verts))
bm.faces.remove(face)
bmesh.update_edit_mesh(obj.data)
assert bpy.ops.velend.fill_tifxyz_holes() == {'FINISHED'}
bpy.ops.object.mode_set(mode='OBJECT')
uvs, faces = commands._mesh_uvs_faces(obj.data)
assert tifxyz.grid_from_uvs(uvs, faces).shape == (5, 5)
print('VELEND TIFXYZ HOLE FILL PASSED', flush=True)
