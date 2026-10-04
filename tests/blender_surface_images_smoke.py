"""Blender smoke test for surface image ownership and graph resolution."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bpy

import velend
from velend import surface_images
from velend.renderer import VolumeSamplerRenderEngine


for module in velend._modules:
	if hasattr(module, 'register'):
		module.register()

assert VolumeSamplerRenderEngine.bl_use_shading_nodes_custom is False
bpy.context.scene.render.engine = VolumeSamplerRenderEngine.bl_idname

bpy.ops.mesh.primitive_plane_add()
obj = bpy.context.active_object
obj.name = "Surface Image Test"
obj["velend_tifxyz_shape"] = (4, 8)
obj["velend_tifxyz_source_shape"] = (12, 24)
obj.data.uv_layers.new(name="UVMap")

shared = bpy.data.materials.new("Shared Surface")
shared.use_nodes = True
obj.data.materials.append(shared)
bpy.ops.mesh.primitive_plane_add()
other = bpy.context.active_object
other.data.materials.append(shared)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
other.select_set(False)

image = bpy.data.images.new("Prediction", width=8, height=4, alpha=True)
node = surface_images.attach_image(obj, image, 'PREDICTION')
assert obj.active_material != shared
assert node.image == image
assert node.parent is None
assert node.inputs["Vector"].is_linked
assert node.inputs["Vector"].links[0].from_node.bl_idname == 'ShaderNodeUVMap'
bpy.context.view_layer.update()
bpy.ops.object.mode_set(mode='TEXTURE_PAINT')
assert image in obj.active_material.texture_paint_images[:]
bpy.ops.object.mode_set(mode='OBJECT')

surface_images.connect_for_render(obj.active_material, node)
resolved, uv_name, resolved_node = surface_images.resolved_image(obj.active_material)
assert resolved == image and uv_name == "UVMap" and resolved_node == node
assert surface_images.render_source(obj.active_material)[0] == 'TEXTURE'
obj.active_material.velend_render_mode = 'OVERLAY'
assert surface_images.render_source(obj.active_material)[0] == 'OVERLAY'
principled = node.outputs["Color"].links[0].to_node
obj.active_material.node_tree.links.remove(node.outputs["Color"].links[0])
mode, _image, _uv, _node, warning = surface_images.render_source(obj.active_material)
assert mode == 'VOLUME' and warning
surface_images.connect_for_render(obj.active_material, node)
assert principled == node.outputs["Color"].links[0].to_node

with tempfile.TemporaryDirectory() as directory:
	path = os.path.join(directory, "annotation.png")
	assert bpy.ops.velend.new_surface_annotation(
		'EXEC_DEFAULT', filepath=path
	) == {'FINISHED'}
	annotation = surface_images.selected_node(obj.active_material).image
	assert tuple(annotation.size) == (8, 4)
	assert os.path.isfile(path)

assert len(surface_images.managed_nodes(obj.active_material)) == 2
nodes = surface_images.managed_nodes(obj.active_material)
legacy_frame = obj.active_material.node_tree.nodes.new('NodeFrame')
legacy_frame.name = "Velend Surface Images"
nodes[0].parent = legacy_frame
surface_images._migrate_image_layout()
assert all(node.parent is None for node in nodes)
assert len({tuple(node.location) for node in nodes}) == len(nodes)
assert obj.active_material.node_tree.nodes.get("Velend Surface Images") is None

# An existing file already has managed nodes when the extension registers.
for module in reversed(velend._modules):
	if hasattr(module, 'unregister'):
		module.unregister()
assert not hasattr(bpy.context.scene, "velend")
for module in velend._modules:
	if hasattr(module, 'register'):
		module.register()
assert hasattr(bpy.context.scene, "velend")
assert surface_images._migrate_when_registered() is None
print("VELEND SURFACE IMAGES PASSED", flush=True)
