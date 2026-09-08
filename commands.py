import math

import bpy

from . import state
from .renderer import VolumeSamplerRenderEngine


class velend_OT_load_hires(bpy.types.Operator):
	bl_idname = "velend.load_hires"
	bl_label = "Load High-Res Volume"
	bl_description = (
		"Stream multiresolution volume bricks around meshes nearest the 3D cursor"
	)
	bl_options = {'REGISTER'}

	def execute(self, context):
		depsgraph = context.evaluated_depsgraph_get()
		count = VolumeSamplerRenderEngine.retarget(depsgraph, context.scene.cursor.location)
		if not VolumeSamplerRenderEngine.residencies:
			# No chunk grid means no volume: either none is set, or opening the
			# one that is failed.
			self.report({'ERROR'}, VolumeSamplerRenderEngine.status()[0])
			return {'CANCELLED'}
		if count == 0:
			self.report({'WARNING'}, "No mesh geometry near the 3D cursor")
		else:
			self.report({'INFO'}, "Streaming %d bricks" % count)
		return {'FINISHED'}


class velend_OT_reload_volume(bpy.types.Operator):
	bl_idname = "velend.reload_volume"
	bl_label = "Reload Volume"
	bl_description = (
		"Reopen the scene's volume and rebuild everything streamed from it"
	)
	bl_options = {'REGISTER'}

	def execute(self, context):
		state.close_volume()
		VolumeSamplerRenderEngine.reset()
		return {'FINISHED'}


# The plane objects "Setup Scene for Volume" puts through the volume: their
# name, the rotation that turns a plane's local +Z normal onto a world axis,
# and which of the volume's extents the plane spans in its local X and Y.
_PLANE_SPECS = (
	("Cut X", (0.0, math.pi / 2, 0.0), (2, 1)),
	("Cut Y", (-math.pi / 2, 0.0, 0.0), (0, 2)),
	("Cut Z", (0.0, 0.0, 0.0), (0, 1)),
)


def _ensure_plane(scene, name):
	"""The named plane object, created and linked to the scene when missing."""
	plane = bpy.data.objects.get(name)
	if plane is None or plane.type != 'MESH':
		mesh = bpy.data.meshes.new(name)
		# A unit square in the local XY plane. Its size lives in the object's
		# scale, so running this again only has to move the objects about.
		mesh.from_pydata(
			[(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)],
			[],
			[(0, 1, 2, 3)],
		)
		mesh.update()
		plane = bpy.data.objects.new(name, mesh)
	if plane.name not in scene.collection.all_objects:
		scene.collection.objects.link(plane)
	return plane


def _setup_viewports(context):
	"""Rendered shading, since that is the only one the engine draws in, and a
	grid matching the scene's unit scale."""
	if context.screen is None:
		return
	for area in context.screen.areas:
		if area.type != 'VIEW_3D':
			continue
		for space in area.spaces:
			if space.type == 'VIEW_3D':
				space.shading.type = 'RENDERED'
				space.overlay.grid_scale = 0.001


class velend_OT_setup_scene(bpy.types.Operator):
	bl_idname = "velend.setup_scene"
	bl_label = "Setup Scene for Volume"
	bl_description = (
		"Switch the scene to millimeters and to the volume renderer, and put "
		"three orthogonal planes through the middle of the volume"
	)
	bl_options = {'REGISTER', 'UNDO'}

	def execute(self, context):
		scene = context.scene
		engine = VolumeSamplerRenderEngine

		# Voxels are a few µm across, so a millimeter scene keeps a whole scan
		# down to the hundreds of Blender units the viewport is happiest with.
		scene.unit_settings.system = 'METRIC'
		scene.unit_settings.length_unit = 'MILLIMETERS'
		scene.unit_settings.scale_length = 0.001
		scene.render.engine = engine.bl_idname
		_setup_viewports(context)

		engine.ensure_grid()
		if not engine.residencies:
			self.report({'ERROR'}, engine.status()[0])
			return {'CANCELLED'}
		# A grid built before this ran measured itself against the old unit scale.
		engine.voxels_per_unit = engine.compute_voxels_per_unit()

		extents = [size / engine.voxels_per_unit for size in engine.shape_xyz]
		center = [extent / 2.0 for extent in extents]
		for name, rotation, (local_x, local_y) in _PLANE_SPECS:
			plane = _ensure_plane(scene, name)
			plane.location = center
			plane.rotation_euler = rotation
			plane.scale = (extents[local_x], extents[local_y], 1.0)

		# Bricks stream in around the 3D cursor, so leaving it at the origin
		# would fill the atlases from one corner of the volume.
		scene.cursor.location = center
		engine.rescale()
		self.report(
			{'INFO'}, "Scene set up for a %.1f x %.1f x %.1f mm volume" % tuple(extents)
		)
		return {'FINISHED'}


def _view_menu(self, context):
	self.layout.operator(velend_OT_load_hires.bl_idname)


def register():
	bpy.utils.register_class(velend_OT_load_hires)
	bpy.utils.register_class(velend_OT_reload_volume)
	bpy.utils.register_class(velend_OT_setup_scene)
	bpy.types.VIEW3D_MT_view.append(_view_menu)


def unregister():
	bpy.types.VIEW3D_MT_view.remove(_view_menu)
	bpy.utils.unregister_class(velend_OT_setup_scene)
	bpy.utils.unregister_class(velend_OT_reload_volume)
	bpy.utils.unregister_class(velend_OT_load_hires)
