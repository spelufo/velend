import math
import os

import bpy
import numpy as np

from . import state
from . import tifxyz
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

	def modal(self, context, event):
		engine = VolumeSamplerRenderEngine
		key = (engine.volume_path(), engine.settings().source_url.strip())
		if event.type == 'ESC' or key != self._key or context.scene != self._scene:
			context.window_manager.event_timer_remove(self._timer)
			return {'CANCELLED'}
		if event.type != 'TIMER':
			return {'PASS_THROUGH'}
		volume = engine.get_volume()
		if volume is None and engine.load_future is not None:
			return {'PASS_THROUGH'}
		context.window_manager.event_timer_remove(self._timer)
		if volume is None:
			self.report({'ERROR'}, engine.status()[0])
			return {'CANCELLED'}
		return self.execute(context)

	def execute(self, context):
		engine = VolumeSamplerRenderEngine
		if engine.get_volume() is None:
			if engine.load_future is None:
				self.report({'ERROR'}, engine.status()[0])
				return {'CANCELLED'}
			self._key = (engine.volume_path(), engine.settings().source_url.strip())
			self._scene = context.scene
			self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
			context.window_manager.modal_handler_add(self)
			return {'RUNNING_MODAL'}
		scene = context.scene
		engine = VolumeSamplerRenderEngine

		# Voxels are a few µm across, so a millimeter scene keeps a whole scan
		# down to the hundreds of Blender units the viewport is happiest with.
		scene.unit_settings.system = 'METRIC'
		scene.unit_settings.length_unit = 'MILLIMETERS'
		scene.unit_settings.scale_length = 0.001
		scene.render.engine = engine.bl_idname
		_setup_viewports(context)

		engine.voxels_per_unit = engine.compute_voxels_per_unit()
		shape_xyz = reversed(engine.get_volume()[0].shape)
		extents = [size / engine.voxels_per_unit for size in shape_xyz]
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


def _surface_mesh(name, surface, voxels_per_unit):
	"""A quad mesh of the surface, in Blender units."""
	mesh = bpy.data.meshes.new(name)
	positions = (surface.positions / voxels_per_unit).astype(np.float32)
	quads = surface.quads
	loops = quads.size
	# The grid is far too big to go through `from_pydata`: the mesh is sized
	# first and then filled in one `foreach_set` per array, the same way the
	# renderer reads meshes back out.
	mesh.vertices.add(len(positions))
	mesh.loops.add(loops)
	mesh.polygons.add(len(quads))
	mesh.vertices.foreach_set("co", positions.ravel())
	mesh.loops.foreach_set("vertex_index", quads.ravel())
	# Every face is a quad, so the faces start every four loops.
	mesh.polygons.foreach_set("loop_start", np.arange(0, loops, 4, dtype=np.int32))

	uvs = mesh.uv_layers.new(name="UVMap")
	uvs.data.foreach_set("uv", surface.uvs[quads.ravel()].ravel())
	for channel, values in surface.channels.items():
		attribute = mesh.attributes.new(channel, 'FLOAT', 'POINT')
		attribute.data.foreach_set("value", np.ascontiguousarray(values, dtype=np.float32))

	mesh.update()
	mesh.validate()
	return mesh


def _link_surface(context, surface, voxels_per_unit, step, voxel_size):
	"""Build the surface's object and put it in the scene, selected."""
	name = surface.uuid
	obj = bpy.data.objects.new(name, _surface_mesh(name, surface, voxels_per_unit))
	# Where it came from and what it was read with, so that a later reload or
	# export does not have to be told again.
	obj["velend_tifxyz_path"] = surface.path
	obj["velend_tifxyz_uuid"] = surface.uuid
	obj["velend_tifxyz_scale"] = list(surface.scale)
	obj["velend_tifxyz_step"] = step
	obj["velend_tifxyz_voxel_size"] = voxel_size
	context.scene.collection.objects.link(obj)
	obj.select_set(True)
	context.view_layer.objects.active = obj
	return obj


class velend_OT_import_tifxyz(bpy.types.Operator):
	bl_idname = "velend.import_tifxyz"
	bl_label = "Import tifxyz Surface"
	bl_description = (
		"Import a volume-cartographer surface directory, or a folder of them, "
		"as meshes placed in the volume"
	)
	bl_options = {'REGISTER', 'UNDO'}

	# A surface is a directory, so the file browser is pointed at its meta.json;
	# picking the directory itself works too.
	filepath: bpy.props.StringProperty(subtype='FILE_PATH', options={'SKIP_SAVE'})
	directory: bpy.props.StringProperty(subtype='DIR_PATH', options={'SKIP_SAVE'})
	filter_glob: bpy.props.StringProperty(default="meta.json", options={'HIDDEN'})

	step: bpy.props.IntProperty(
		name="Step",
		description=(
			"Keep every nth row and column of the grid. Raise it when a segment "
			"brings in more quads than the viewport is happy with"
		),
		default=1,
		min=1,
		soft_max=32,
	)
	voxel_size: bpy.props.FloatProperty(
		name="Voxel Size",
		description=(
			"Width of a full resolution voxel, in micrometers. Surface "
			"coordinates are in voxels of the volume they were segmented from, "
			"so this places them in the scene. Filled in from the scene's own "
			"voxel size"
		),
		default=9.362,
		min=1e-6,
		soft_max=100.0,
		precision=3,
	)
	load_mask: bpy.props.BoolProperty(
		name="Apply Mask",
		description="Drop the grid points mask.tif takes out of the surface",
		default=True,
	)
	load_channels: bpy.props.BoolProperty(
		name="Extra Channels",
		description=(
			"Read the surface's other TIFFs, such as generations.tif, as mesh "
			"attributes"
		),
		default=True,
	)

	def invoke(self, context, event):
		# The scene's volume is the one a surface most likely belongs to.
		self.voxel_size = context.scene.velend.resolution
		context.window_manager.fileselect_add(self)
		return {'RUNNING_MODAL'}

	def chosen_dir(self):
		"""The directory the file browser was left in, whatever was picked."""
		path = self.filepath.strip() or self.directory.strip()
		if not path:
			return ""
		path = os.path.expanduser(bpy.path.abspath(path))
		return path if os.path.isdir(path) else os.path.dirname(path)

	def execute(self, context):
		directory = self.chosen_dir()
		if not os.path.isdir(directory):
			self.report({'ERROR'}, "No directory to import from")
			return {'CANCELLED'}
		paths = tifxyz.surface_dirs(directory)
		if not paths:
			self.report({'ERROR'}, "No tifxyz surface in %s" % directory)
			return {'CANCELLED'}

		voxels_per_unit = (
			1000000.0 * context.scene.unit_settings.scale_length
		) / self.voxel_size
		# Only what was just imported ends up selected, when there is a mode
		# where that means anything.
		if bpy.ops.object.select_all.poll():
			bpy.ops.object.select_all(action='DESELECT')
		vertices = 0
		faces = 0
		failures = []
		for path in paths:
			try:
				surface = tifxyz.read_surface(
					path,
					step=self.step,
					load_mask=self.load_mask,
					load_channels=self.load_channels,
				)
			except Exception as error:
				# One unreadable segment should not lose the rest of a folder.
				print("velend: reading %s failed: %s" % (path, error))
				failures.append("%s (%s)" % (os.path.basename(path), error))
				continue
			_link_surface(context, surface, voxels_per_unit, self.step, self.voxel_size)
			vertices += len(surface.positions)
			faces += len(surface.quads)

		if failures:
			self.report({'ERROR' if not vertices else 'WARNING'},
				"Could not read %d of %d surfaces: %s"
				% (len(failures), len(paths), ", ".join(failures)))
			if not vertices:
				return {'CANCELLED'}
		else:
			self.report(
				{'INFO'},
				"Imported %d surface%s, %d vertices and %d faces"
				% (len(paths), "" if len(paths) == 1 else "s", vertices, faces),
			)
		return {'FINISHED'}


def _view_menu(self, context):
	self.layout.operator(velend_OT_load_hires.bl_idname)


def _import_menu(self, context):
	self.layout.operator(
		velend_OT_import_tifxyz.bl_idname,
		text="Volume Cartographer Surface (tifxyz)",
	)


def register():
	bpy.utils.register_class(velend_OT_load_hires)
	bpy.utils.register_class(velend_OT_reload_volume)
	bpy.utils.register_class(velend_OT_setup_scene)
	bpy.utils.register_class(velend_OT_import_tifxyz)
	bpy.types.VIEW3D_MT_view.append(_view_menu)
	bpy.types.TOPBAR_MT_file_import.append(_import_menu)


def unregister():
	bpy.types.TOPBAR_MT_file_import.remove(_import_menu)
	bpy.types.VIEW3D_MT_view.remove(_view_menu)
	bpy.utils.unregister_class(velend_OT_import_tifxyz)
	bpy.utils.unregister_class(velend_OT_setup_scene)
	bpy.utils.unregister_class(velend_OT_reload_volume)
	bpy.utils.unregister_class(velend_OT_load_hires)
