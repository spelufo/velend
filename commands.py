import json
import math
import os

import bmesh
import bpy
import numpy as np

from . import metadata as volume_metadata
from . import state
from . import tifxyz
from . import umbilicus
from . import ui
from . import volpkg
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


# The volumes of every project file picked this session, keyed by the file and
# the time it was written. The menu below asks for them on every redraw, which
# is no reason to reread and reparse the project each time.
_volpkg_volumes = {}

# Blender does not copy the strings an enum callback returns, so the last list
# has to outlive the call that handed it out. See `ui._items`.
_enum_items = []


def _volpkg_items(path):
	"""The volumes of the project at `path`, parsed at most once per write."""
	key = (path, os.path.getmtime(path))
	volumes = _volpkg_volumes.get(key)
	if volumes is None:
		volumes = _volpkg_volumes[key] = volpkg.volumes(path)
	return volumes


def _volume_description(volume):
	"""What the project says about one of its volumes, for the menu."""
	parts = []
	if volume.voxel_size_um:
		parts.append("%g um voxels" % volume.voxel_size_um)
	if volume.base_scale:
		# The whole pyramid is read either way, so picking one of these is the
		# same as picking the entry it selects a level of.
		parts.append("entered at level %d" % volume.base_scale)
	if volume.sample_id:
		parts.append(volume.sample_id)
	parts.extend(
		tag for tag in volume.tags if not tag.startswith("vc-") and tag != "open-data"
	)
	if not volume.remote:
		parts.append(volume.path if volume.cached else "missing: %s" % volume.path)
	else:
		parts.append("mirrored to disk" if volume.cached else "not downloaded yet")
	return ", ".join(parts)


def _volpkg_volume_items(self, context):
	"""The picked project's volumes, as menu items keyed by their position in
	the project's list: two entries can name the same volume at different
	levels, so nothing else about them is unique."""
	global _enum_items
	try:
		volumes = _volpkg_items(self.filepath)
	except (OSError, ValueError, LookupError):
		# `execute` reported it already; the dialog is just drawing.
		volumes = []
	_enum_items = [
		(
			str(index),
			volume.name,
			_volume_description(volume),
			'DISK_DRIVE' if volume.cached else 'URL' if volume.remote else 'ERROR',
			index,
		)
		for index, volume in enumerate(volumes)
	]
	return _enum_items


class velend_OT_choose_volpkg(bpy.types.Operator):
	bl_idname = "velend.choose_volpkg"
	bl_label = "Choose from volpkg.json"
	bl_description = (
		"Point the scene at one of the volumes a VC3D project lists, through "
		"the same cache directory VC3D reads it in"
	)
	bl_options = {'REGISTER'}

	filepath: bpy.props.StringProperty(subtype='FILE_PATH', options={'SKIP_SAVE'})
	filter_glob: bpy.props.StringProperty(
		default="*.volpkg.json", options={'HIDDEN', 'SKIP_SAVE'}
	)

	def invoke(self, context, event):
		# Where VC3D puts the open data projects it downloads. A project of
		# one's own is as likely, so this only says where to start looking.
		projects = volpkg.projects_dir()
		if os.path.isdir(projects):
			self.filepath = os.path.join(projects, "")
		context.window_manager.fileselect_add(self)
		return {'RUNNING_MODAL'}

	def execute(self, context):
		path = os.path.expanduser(bpy.path.abspath(self.filepath.strip()))
		name = os.path.basename(path)
		try:
			volumes = _volpkg_items(path)
		except (OSError, ValueError, LookupError) as error:
			self.report({'ERROR'}, "Could not read %s: %s" % (name, error))
			return {'CANCELLED'}
		if not volumes:
			self.report({'ERROR'}, "%s lists no volumes" % name)
			return {'CANCELLED'}
		# Which of them to use is asked in a dialog of its own: the file
		# browser cannot show a menu of what is inside the file it browses for.
		bpy.ops.velend.set_volpkg_volume('INVOKE_DEFAULT', filepath=path)
		return {'FINISHED'}


class velend_OT_set_volpkg_volume(bpy.types.Operator):
	bl_idname = "velend.set_volpkg_volume"
	bl_label = "Choose Volume"
	bl_description = "Set the scene's volume to one of the ones a VC3D project lists"
	bl_options = {'REGISTER', 'UNDO'}

	filepath: bpy.props.StringProperty(options={'SKIP_SAVE', 'HIDDEN'})
	volume: bpy.props.EnumProperty(
		name="Volume",
		description="Which of the project's volumes to render",
		items=_volpkg_volume_items,
		options={'SKIP_SAVE'},
	)

	def invoke(self, context, event):
		try:
			volumes = _volpkg_items(self.filepath)
		except (OSError, ValueError, LookupError) as error:
			self.report({'ERROR'}, "Could not read the project: %s" % error)
			return {'CANCELLED'}
		if not volumes:
			self.report({'ERROR'}, "No volumes to choose from")
			return {'CANCELLED'}
		# The scan's own reconstruction, over the predictions made from it, and
		# one already downloaded over one that is not.
		self.volume = str(max(
			range(len(volumes)),
			key=lambda index: (volumes[index].preferred, volumes[index].cached, -index),
		))
		return context.window_manager.invoke_props_dialog(self, width=500)

	def draw(self, context):
		layout = self.layout
		layout.label(text=os.path.basename(self.filepath), icon='FILE_VOLUME')
		layout.prop(self, "volume")

	def execute(self, context):
		try:
			volume = _volpkg_items(self.filepath)[int(self.volume)]
		except (OSError, ValueError, LookupError) as error:
			self.report({'ERROR'}, "Could not read the project: %s" % error)
			return {'CANCELLED'}

		settings = context.scene.velend
		# The URL first: setting either of the two reopens the volume, and
		# doing it in this order means the reopen that sticks is the one with
		# both of them in hand.
		settings.source_url = volume.url
		settings.volume_path = volume.path
		# The project states the voxel size outright, which beats the guess the
		# assignments above made from the directory's name, and answers the
		# dialog they may have queued for a directory that names none.
		if volume.resolution_um:
			ui.set_resolution(settings, volume.resolution_um)

		if volume.remote and not volume.cached:
			self.report({'INFO'}, "%s streams into %s" % (volume.name, volume.path))
		elif not volume.remote and not volume.cached:
			self.report({'WARNING'}, "%s is not at %s" % (volume.name, volume.path))
		else:
			self.report({'INFO'}, "Volume set to %s" % volume.name)
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
		# A unit square in the local XY plane, resized in place by
		# `_resize_plane` on every run instead of through the object's scale,
		# so the object never ends up with a non-uniform scale.
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


def _resize_plane(plane, width, height):
	"""Set the plane's mesh to `width` x `height` local units, leaving the
	object's scale at 1 so it doesn't trip Blender's non-uniform scale
	warning."""
	half_w, half_h = width / 2.0, height / 2.0
	plane.data.vertices.foreach_set(
		'co',
		(
			-half_w, -half_h, 0.0,
			half_w, -half_h, 0.0,
			half_w, half_h, 0.0,
			-half_w, half_h, 0.0,
		),
	)
	plane.data.update()


def _frame_objects(context, objects):
	"""Point every 3D viewport at `objects`, leaving the selection as it was."""
	# Selecting is how `view3d.view_selected` is aimed, and that only works in
	# object mode; a headless run has no viewport to aim at all.
	if not objects or context.screen is None or context.mode != 'OBJECT':
		return
	view_layer = context.view_layer
	selected = list(context.selected_objects)
	active = view_layer.objects.active
	bpy.ops.object.select_all(action='DESELECT')
	for obj in objects:
		obj.select_set(True)
	view_layer.objects.active = objects[0]
	try:
		for area in context.screen.areas:
			if area.type != 'VIEW_3D':
				continue
			region = next((r for r in area.regions if r.type == 'WINDOW'), None)
			if region is None:
				continue
			with context.temp_override(area=area, region=region):
				bpy.ops.view3d.view_selected()
	finally:
		bpy.ops.object.select_all(action='DESELECT')
		for obj in selected:
			obj.select_set(True)
		view_layer.objects.active = active


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

		engine.world_to_voxels = engine.compute_world_to_voxels()
		engine.ensure_grid()
		# The volume's own box, in the scene's coordinates: those of the volume
		# it was set up against, which is this one unless it is being rendered
		# through the transform registered between the two.
		low, high = engine.world_bounds()
		extents = list(high - low)
		center = list((low + high) / 2.0)
		planes = []
		for name, rotation, (local_x, local_y) in _PLANE_SPECS:
			plane = _ensure_plane(scene, name)
			plane.location = center
			plane.rotation_euler = rotation
			_resize_plane(plane, extents[local_x], extents[local_y])
			planes.append(plane)

		# The planes span the volume, so framing them is framing the scan: the
		# viewports start out looking at all of it rather than at whatever the
		# previous file left them on.
		context.view_layer.update()
		_frame_objects(context, planes)

		# Bricks stream in around the 3D cursor, so leaving it at the origin
		# would fill the atlases from one corner of the volume.
		scene.cursor.location = center
		engine.rescale()
		self.report(
			{'INFO'}, "Scene set up for a %.1f x %.1f x %.1f mm volume" % tuple(extents)
		)
		return {'FINISHED'}


_COORDINATE_SPACE_ITEMS = (
	(
		'SCENE',
		"Scene Volume",
		"Coordinates are in the volume the scene was originally set up against",
	),
	(
		'RENDERED',
		"Rendered Volume",
		"Coordinates are in the volume currently being rendered",
	),
)


def _coordinate_voxel_size(context, coordinate_space):
	"""The default voxel size for an import coordinate-space choice."""
	settings = context.scene.velend
	if coordinate_space == 'SCENE':
		return settings.scene_resolution or settings.resolution
	return settings.resolution or settings.scene_resolution


def _coordinate_space_updated(self, context):
	"""Keep an importer's editable voxel size with its chosen volume."""
	self.voxel_size = _coordinate_voxel_size(context, self.coordinate_space)


def _placement(context, voxel_size, coordinate_space='RENDERED'):
	"""Takes points in voxels `voxel_size` micrometers wide into Blender units.

	Coordinates in the scene volume need only their physical scale. Coordinates
	in the rendered volume are brought back the way the renderer takes the
	scene's coordinates to it, through any transform registered for the pair.
	"""
	if coordinate_space == 'SCENE':
		placement = np.eye(4, dtype=np.float64)
		placement[:3, :3] *= voxel_size / VolumeSamplerRenderEngine.um_per_unit()
	else:
		matrix = VolumeSamplerRenderEngine.compute_world_from_voxels()
		scale = voxel_size / (context.scene.velend.resolution or voxel_size)
		placement = np.array(matrix, dtype=np.float64)
		placement[:3, :3] *= scale

	def place(points):
		return (
			points @ placement[:3, :3].T + placement[:3, 3]
		).astype(np.float32)

	place.matrix = placement
	return place


def _surface_mesh(name, surface, place):
	"""A quad mesh of the surface, in Blender units."""
	mesh = bpy.data.meshes.new(name)
	positions = place(surface.positions)
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


def _surface_uv_material(shape):
	"""An image material that tells Blender the tifxyz UV display aspect."""
	height, width = shape
	grid_size = (max(width - 1, 1), max(height - 1, 1))
	material = next((material for material in bpy.data.materials
		if tuple(material.get("velend_tifxyz_grid_size", ())) == grid_size), None)
	if material is not None:
		return material
	image = next((image for image in bpy.data.images
		if tuple(image.get("velend_tifxyz_grid_size", ())) == grid_size), None)
	if image is None:
		image = bpy.data.images.new(
			"tifxyz UV Aspect %d x %d" % (width, height), width=1, height=1, alpha=True
		)
		image.generated_color = (0.0, 0.0, 0.0, 0.0)
		image.display_aspect = grid_size
		image["velend_tifxyz_grid_size"] = grid_size

	material = bpy.data.materials.new("tifxyz UV Aspect")
	material["velend_tifxyz_grid_size"] = grid_size
	material.use_nodes = True
	node = material.node_tree.nodes.new('ShaderNodeTexImage')
	node.name = "tifxyz UV Aspect"
	node.image = image
	material.node_tree.nodes.active = node
	return material


def _surface_shape_from_uvs(obj):
	"""Infer the tifxyz export grid shape from a mesh's active UV map."""
	if obj.mode == 'EDIT':
		bm = bmesh.from_edit_mesh(obj.data)
		layer = bm.loops.layers.uv.active
		if layer is None:
			raise ValueError("the mesh has no active UV map")
		bm.verts.index_update()
		vertex_uvs = np.full((len(bm.verts), 2), np.nan, dtype=np.float64)
		faces = []
		for face in bm.faces:
			faces.append(tuple(loop.vert.index for loop in face.loops))
			for loop in face.loops:
				uv = np.asarray(loop[layer].uv, dtype=np.float64)
				old = vertex_uvs[loop.vert.index]
				if np.isfinite(old).all() and not np.allclose(old, uv, atol=1e-5, rtol=0):
					raise ValueError("a vertex has different UVs on different faces")
				vertex_uvs[loop.vert.index] = uv
	else:
		vertex_uvs, faces = _mesh_uvs_faces(obj.data, require_all=False)
	return tifxyz.partial_grid_from_uvs(vertex_uvs, faces).shape


class velend_OT_setup_tifxyz_uv_aspect(bpy.types.Operator):
	bl_idname = "velend.setup_tifxyz_uv_aspect"
	bl_label = "Set Up tifxyz UV Aspect"
	bl_description = "Give selected imported tifxyz surfaces their correct UV Editor aspect ratio"
	bl_options = {'REGISTER', 'UNDO'}

	@classmethod
	def poll(cls, context):
		return any(obj.type == 'MESH' for obj in context.selected_objects)

	def execute(self, context):
		updated = 0
		failures = []
		for obj in context.selected_objects:
			if obj.type != 'MESH':
				continue
			try:
				material = _surface_uv_material(_surface_shape_from_uvs(obj))
			except (TypeError, ValueError) as error:
				if isinstance(error, tifxyz.UVGridError):
					_select_uv_grid_error(context, obj, error)
				failures.append("%s (%s)" % (obj.name, error))
				continue
			if all(slot.material != material for slot in obj.material_slots):
				obj.data.materials.append(material)
			updated += 1
		if failures:
			self.report({'WARNING'}, "Could not update: %s" % ", ".join(failures))
		if not updated:
			return {'CANCELLED'}
		self.report({'INFO'}, "Set up UV aspect for %d tifxyz surface%s"
			% (updated, "" if updated == 1 else "s"))
		return {'FINISHED'}


def _link_surface(context, surface, place, step, voxel_size):
	"""Build the surface's object and put it in the scene, selected."""
	name = surface.uuid
	obj = bpy.data.objects.new(name, _surface_mesh(name, surface, place))
	obj.data.materials.append(_surface_uv_material(surface.shape))
	# Where it came from and what it was read with, so that a later reload or
	# export does not have to be told again.
	obj["velend_tifxyz_path"] = surface.path
	obj["velend_tifxyz_uuid"] = surface.uuid
	obj["velend_tifxyz_scale"] = list(surface.scale)
	obj["velend_tifxyz_step"] = step
	obj["velend_tifxyz_shape"] = surface.shape
	obj["velend_tifxyz_voxel_size"] = voxel_size
	obj["velend_tifxyz_meta"] = json.dumps(surface.meta)
	obj["velend_tifxyz_placement"] = place.matrix.ravel().tolist()
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
	coordinate_space: bpy.props.EnumProperty(
		name="Coordinates",
		description="Which volume's voxel coordinate system the surface uses",
		items=_COORDINATE_SPACE_ITEMS,
		default='SCENE',
		update=_coordinate_space_updated,
	)

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
			"so this places them in the scene. Filled in from the chosen "
			"coordinate volume"
		),
		default=9.362,
		min=1e-6,
		soft_max=100.0,
		precision=3,
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
		self.coordinate_space = 'SCENE'
		self.voxel_size = _coordinate_voxel_size(context, self.coordinate_space)
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

		place = _placement(context, self.voxel_size, self.coordinate_space)
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
					load_channels=self.load_channels,
				)
			except Exception as error:
				# One unreadable segment should not lose the rest of a folder.
				print("velend: reading %s failed: %s" % (path, error))
				failures.append("%s (%s)" % (os.path.basename(path), error))
				continue
			_link_surface(context, surface, place, self.step, self.voxel_size)
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



def _attribute_values(attribute, count):
	values = np.empty(count, dtype=np.float32)
	attribute.data.foreach_get("value", values)
	return values


def _mesh_uvs_faces(mesh, require_all=True):
	"""Return one consistent active-UV coordinate per vertex and every face."""
	layer = mesh.uv_layers.active
	if layer is None:
		raise ValueError("the mesh has no active UV map")
	vertex_uvs = np.full((len(mesh.vertices), 2), np.nan, dtype=np.float64)
	for loop in mesh.loops:
		uv = np.asarray(layer.data[loop.index].uv, dtype=np.float64)
		old = vertex_uvs[loop.vertex_index]
		if np.isfinite(old).all() and not np.allclose(old, uv, atol=1e-5, rtol=0):
			raise ValueError("a vertex has different UVs on different faces")
		vertex_uvs[loop.vertex_index] = uv
	if require_all and not np.isfinite(vertex_uvs).all():
		raise ValueError("some vertices are not used by the active UV map")
	return vertex_uvs, [tuple(polygon.vertices) for polygon in mesh.polygons]


def _select_uv_grid_error(context, obj, error):
	"""Select mesh elements attached to a reported UV-grid validation error."""
	was_edit = obj.mode == 'EDIT'
	context.view_layer.objects.active = obj
	obj.select_set(True)
	if was_edit and bpy.ops.object.mode_set.poll():
		bpy.ops.object.mode_set(mode='OBJECT')
	mesh = obj.data
	for vertex in mesh.vertices:
		vertex.select = False
	for edge in mesh.edges:
		edge.select = False
	for polygon in mesh.polygons:
		polygon.select = False
	vertices = set(error.vertices)
	for index in error.faces:
		if 0 <= index < len(mesh.polygons):
			polygon = mesh.polygons[index]
			polygon.select = True
			vertices.update(polygon.vertices)
	for index in vertices:
		if 0 <= index < len(mesh.vertices):
			mesh.vertices[index].select = True
	for edge in mesh.edges:
		if edge.vertices[0] in vertices and edge.vertices[1] in vertices:
			edge.select = True
	mesh.update()
	if was_edit and bpy.ops.object.mode_set.poll():
		bpy.ops.object.mode_set(mode='EDIT')


def _mesh_grid(mesh):
	"""The mesh's rectangular row-major vertex index grid."""
	vertex_uvs, faces = _mesh_uvs_faces(mesh)
	return tifxyz.grid_from_uvs(vertex_uvs, faces)


class velend_OT_fill_tifxyz_holes(bpy.types.Operator):
	bl_idname = "velend.fill_tifxyz_holes"
	bl_label = "Fill tifxyz Holes"
	bl_description = "Restore enclosed missing UV-grid quads with smooth 3D interpolation"
	bl_options = {'REGISTER', 'UNDO'}

	@classmethod
	def poll(cls, context):
		obj = context.active_object
		return obj is not None and obj.type == 'MESH' and obj.mode in {'OBJECT', 'EDIT'}

	def execute(self, context):
		obj = context.active_object
		mesh = obj.data
		editing = obj.mode == 'EDIT'
		bm = bmesh.from_edit_mesh(mesh) if editing else bmesh.new()
		if not editing:
			bm.from_mesh(mesh)
		try:
			bm.verts.ensure_lookup_table()
			bm.verts.index_update()
			uv_layer = bm.loops.layers.uv.active
			if uv_layer is None:
				raise ValueError("the mesh has no active UV map")
			uvs = np.full((len(bm.verts), 2), np.nan, dtype=np.float64)
			faces = []
			for face in bm.faces:
				faces.append(tuple(loop.vert.index for loop in face.loops))
				for loop in face.loops:
					uv = np.asarray(loop[uv_layer].uv, dtype=np.float64)
					old = uvs[loop.vert.index]
					if np.isfinite(old).all() and not np.allclose(old, uv, atol=1e-5, rtol=0):
						raise ValueError("a vertex has different UVs on different faces")
					uvs[loop.vert.index] = uv
			grid, fill_cells = tifxyz.enclosed_uv_holes(uvs, faces)
			if not fill_cells.any():
				self.report({'INFO'}, "No enclosed tifxyz holes found")
				return {'CANCELLED'}

			float_layers = []
			for attribute in mesh.attributes:
				if (attribute.domain == 'POINT' and attribute.data_type == 'FLOAT'
						and not attribute.name.startswith(".")
						and attribute.name not in {"mask", "x", "y", "z"}):
					layer = bm.verts.layers.float.get(attribute.name)
					if layer is not None:
						float_layers.append(layer)
			values = np.array([
			list(vert.co) + [float(vert[layer]) for layer in float_layers]
			for vert in bm.verts
		], dtype=np.float64)
			filled = tifxyz.interpolate_hole_grid(grid, fill_cells, values)
			mapped = {(int(row), int(col)): bm.verts[int(index)]
				for (row, col), index in np.ndenumerate(grid) if index >= 0}
			for cell, value in filled.items():
				vert = bm.verts.new(value[:3])
				for layer, component in zip(float_layers, value[3:]):
					vert[layer] = float(component)
				mapped[cell] = vert
			finite = uvs[np.isfinite(uvs).all(axis=1)]
			u_min, v_min = finite.min(axis=0)
			u_max, v_max = finite.max(axis=0)
			height, width = grid.shape
			for row, col in zip(*np.nonzero(fill_cells)):
				cells = ((row, col), (row + 1, col),
					(row + 1, col + 1), (row, col + 1))
				face = bm.faces.new(tuple(mapped[cell] for cell in cells))
				for loop, (uv_row, uv_col) in zip(face.loops, cells):
					loop[uv_layer].uv = (
						u_max - uv_col * (u_max - u_min) / (width - 1),
						v_max - uv_row * (v_max - v_min) / (height - 1),
					)
			if editing:
				bmesh.update_edit_mesh(mesh, loop_triangles=False)
			else:
				bm.to_mesh(mesh)
				mesh.update()
			self.report({'INFO'}, "Filled %d tifxyz quads" % fill_cells.sum())
			return {'FINISHED'}
		except (TypeError, ValueError) as error:
			self.report({'ERROR'}, "Could not fill tifxyz holes: %s" % error)
			return {'CANCELLED'}
		finally:
			if not editing:
				bm.free()


def _mesh_export_arrays(mesh, obj, grid):
	positions = np.empty((len(mesh.vertices), 3), dtype=np.float64)
	mesh.vertices.foreach_get("co", positions.ravel())
	world = np.asarray(obj.matrix_world, dtype=np.float64)
	positions = positions @ world[:3, :3].T + world[:3, 3]
	mask = np.ones(len(mesh.vertices), dtype=np.float32)
	channels = {}
	for attribute in mesh.attributes:
		if (
			attribute.name not in {"mask", "x", "y", "z"}
			and not attribute.name.startswith(".")
			and attribute.domain == 'POINT'
			and attribute.data_type == 'FLOAT'
		):
			channels[attribute.name] = _attribute_values(attribute, len(mesh.vertices))[grid]
	return positions[grid], mask[grid], channels


def _completed_mesh_arrays(mesh, obj):
	"""Complete the smallest UV crop, masking lattice points absent from the mesh."""
	vertex_uvs, faces = _mesh_uvs_faces(mesh, require_all=False)
	present = tifxyz.partial_grid_from_uvs(vertex_uvs, faces)
	current = np.empty((len(mesh.vertices), 3), dtype=np.float64)
	mesh.vertices.foreach_get("co", current.ravel())
	world = np.asarray(obj.matrix_world, dtype=np.float64)
	current = current @ world[:3, :3].T + world[:3, 3]
	points = tifxyz.scatter_grid(current, present, (-1.0, -1.0, -1.0))

	values = np.ones(len(mesh.vertices), dtype=np.float32)
	mask = tifxyz.scatter_grid(values, present, 0.0)

	channels = {}
	for attribute in mesh.attributes:
		if (
			attribute.name not in {"mask", "x", "y", "z"}
			and not attribute.name.startswith(".")
			and attribute.domain == 'POINT'
			and attribute.data_type == 'FLOAT'
		):
			values = _attribute_values(attribute, len(mesh.vertices))
			channels[attribute.name] = tifxyz.scatter_grid(values, present, 0.0)
	return points, mask, channels


class velend_OT_export_tifxyz(bpy.types.Operator):
	bl_idname = "velend.export_tifxyz"
	bl_label = "Export tifxyz Surface"
	bl_description = "Export the active UV quad grid, cropping and masking deleted vertices"
	bl_options = {'REGISTER'}

	filepath: bpy.props.StringProperty(subtype='DIR_PATH', options={'SKIP_SAVE'})
	uuid: bpy.props.StringProperty(name="UUID")
	scale_x: bpy.props.FloatProperty(name="Grid Scale X", default=1.0, min=1e-9)
	scale_y: bpy.props.FloatProperty(name="Grid Scale Y", default=1.0, min=1e-9)
	voxel_size: bpy.props.FloatProperty(name="Voxel Size", default=9.362, min=1e-9)
	overwrite: bpy.props.BoolProperty(
		name="Replace Existing tifxyz Files",
		description="Replace metadata and TIFF channels already in the chosen directory",
		default=False,
	)

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.type == 'MESH'

	def draw(self, context):
		layout = self.layout
		layout.prop(self, "uuid")
		row = layout.row(align=True)
		row.prop(self, "scale_x")
		row.prop(self, "scale_y")
		layout.prop(self, "voxel_size")
		layout.prop(self, "overwrite")

	def _defaults(self, context):
		obj = context.active_object
		self.uuid = str(obj.get("velend_tifxyz_uuid", obj.name))
		step = max(1, int(obj.get("velend_tifxyz_step", 1)))
		scale = obj.get("velend_tifxyz_scale", (1.0, 1.0))
		self.scale_x = float(scale[0]) / step
		self.scale_y = float(scale[1]) / step
		self.voxel_size = float(
			obj.get("velend_tifxyz_voxel_size", context.scene.velend.resolution)
		)
		source = obj.get("velend_tifxyz_path")
		self.filepath = str(source or os.path.join(os.getcwd(), self.uuid))

	def invoke(self, context, event):
		self._defaults(context)
		if context.active_object.modifiers:
			self.report({'ERROR'}, "Apply or remove modifiers before exporting tifxyz")
			return {'CANCELLED'}
		context.window_manager.fileselect_add(self)
		return {'RUNNING_MODAL'}

	def execute(self, context):
		obj = context.active_object
		if obj.modifiers:
			self.report({'ERROR'}, "Apply or remove modifiers before exporting tifxyz")
			return {'CANCELLED'}
		directory = os.path.expanduser(bpy.path.abspath(self.filepath.strip()))
		if not directory:
			self.report({'ERROR'}, "No export directory selected")
			return {'CANCELLED'}
		existing = os.path.isdir(directory) and any(
			name == "meta.json" or name.endswith(".tif")
			for name in os.listdir(directory)
		)
		if existing and not self.overwrite:
			self.report({'ERROR'}, "Target contains tifxyz files; enable replacement to export")
			return {'CANCELLED'}

		mesh = obj.data
		try:
			missing = None
			stored = obj.get("velend_tifxyz_placement")
			placement = (
				np.asarray(stored, dtype=np.float64).reshape(4, 4)
				if stored is not None else _placement(context, self.voxel_size).matrix
			)
			try:
				grid = _mesh_grid(mesh)
				points, mask, channels = _mesh_export_arrays(mesh, obj, grid)
			except ValueError:
				points, mask, channels = _completed_mesh_arrays(mesh, obj)
				missing = mask < 0.5
				grid = np.empty(points.shape[:2], dtype=np.int32)
			inverse = np.linalg.inv(placement)
			points = points @ inverse[:3, :3].T + inverse[:3, 3]
			if missing is not None:
				points[missing] = -1.0
			try:
				meta = json.loads(obj.get("velend_tifxyz_meta", "{}"))
			except (TypeError, ValueError):
				meta = {}
			tifxyz.write_surface(
				directory, points, mask, channels, meta, self.uuid,
				(self.scale_x, self.scale_y),
			)
		except Exception as error:
			if isinstance(error, tifxyz.UVGridError):
				_select_uv_grid_error(context, obj, error)
			self.report({'ERROR'}, "Could not export tifxyz: %s" % error)
			return {'CANCELLED'}
		self.report(
			{'INFO'}, "Exported %d x %d tifxyz surface" % (grid.shape[1], grid.shape[0])
		)
		return {'FINISHED'}


def _umbilicus_mesh(name, curve, place):
	"""A mesh of the umbilicus' points joined into a polyline, in Blender units."""
	mesh = bpy.data.meshes.new(name)
	positions = place(curve.positions)
	mesh.vertices.add(len(positions))
	mesh.edges.add(len(curve.edges))
	mesh.vertices.foreach_set("co", positions.ravel())
	mesh.edges.foreach_set("vertices", curve.edges.ravel())
	if curve.scores is not None:
		attribute = mesh.attributes.new("score", 'FLOAT', 'POINT')
		attribute.data.foreach_set("value", curve.scores)
	mesh.update()
	mesh.validate()
	return mesh


class velend_OT_import_umbilicus(bpy.types.Operator):
	bl_idname = "velend.import_umbilicus"
	bl_label = "Import Umbilicus"
	bl_description = (
		"Import an umbilicus.json as a polyline running up the scroll's core"
	)
	bl_options = {'REGISTER', 'UNDO'}

	filepath: bpy.props.StringProperty(subtype='FILE_PATH', options={'SKIP_SAVE'})
	filter_glob: bpy.props.StringProperty(
		default="*.json;*.txt;*.csv", options={'HIDDEN', 'SKIP_SAVE'}
	)
	coordinate_space: bpy.props.EnumProperty(
		name="Coordinates",
		description="Which volume's voxel coordinate system the points use",
		items=_COORDINATE_SPACE_ITEMS,
		default='SCENE',
		update=_coordinate_space_updated,
	)

	voxel_size: bpy.props.FloatProperty(
		name="Voxel Size",
		description=(
			"Width of a full resolution voxel, in micrometers, for the volume "
			"the points were placed in. Filled in from the chosen coordinate "
			"volume, and overridden by the file when it states one of its own"
		),
		default=9.362,
		min=1e-6,
		soft_max=100.0,
		precision=3,
	)

	def invoke(self, context, event):
		self.coordinate_space = 'SCENE'
		self.voxel_size = _coordinate_voxel_size(context, self.coordinate_space)
		context.window_manager.fileselect_add(self)
		return {'RUNNING_MODAL'}

	def execute(self, context):
		path = os.path.expanduser(bpy.path.abspath(self.filepath.strip()))
		if not os.path.isfile(path):
			self.report({'ERROR'}, "No umbilicus file to import")
			return {'CANCELLED'}
		try:
			curve = umbilicus.read_umbilicus(path)
		except Exception as error:
			self.report({'ERROR'}, "Could not read %s: %s" % (os.path.basename(path), error))
			return {'CANCELLED'}

		# A file that states its own voxel size pins the scale of its numbers,
		# while the coordinate-space choice says which volume frame they use.
		voxel_size = curve.voxel_size_um or self.voxel_size
		place = _placement(context, voxel_size, self.coordinate_space)

		name = curve.name
		obj = bpy.data.objects.new(name, _umbilicus_mesh(name, curve, place))
		# Where it came from and what it was read with, so that a later reload
		# or export does not have to be told again.
		obj["velend_umbilicus_path"] = curve.path
		obj["velend_umbilicus_voxel_size"] = voxel_size
		obj["velend_umbilicus_coordinate_space"] = self.coordinate_space
		obj["velend_umbilicus_metadata"] = json.dumps(curve.metadata)
		obj["velend_umbilicus_placement"] = place.matrix.ravel().tolist()
		context.scene.collection.objects.link(obj)
		if bpy.ops.object.select_all.poll():
			bpy.ops.object.select_all(action='DESELECT')
		obj.select_set(True)
		context.view_layer.objects.active = obj

		self.report(
			{'INFO'},
			"Imported %d umbilicus points at %g um voxels%s"
			% (
				len(curve.positions),
				voxel_size,
				" (stated by the file)" if curve.voxel_size_um else "",
			),
		)
		return {'FINISHED'}


class velend_OT_export_umbilicus(bpy.types.Operator):
	bl_idname = "velend.export_umbilicus"
	bl_label = "Export Umbilicus"
	bl_description = "Export the active mesh's single connected line as an umbilicus.json"
	bl_options = {'REGISTER'}

	filepath: bpy.props.StringProperty(subtype='FILE_PATH', options={'SKIP_SAVE'})
	filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN', 'SKIP_SAVE'})
	coordinate_space: bpy.props.EnumProperty(
		name="Coordinates",
		description="Which volume's voxel coordinate system the exported points use",
		items=_COORDINATE_SPACE_ITEMS,
		default='SCENE',
		update=_coordinate_space_updated,
	)
	voxel_size: bpy.props.FloatProperty(
		name="Voxel Size", default=9.362, min=1e-6, soft_max=100.0, precision=3,
	)
	overwrite: bpy.props.BoolProperty(
		name="Replace Existing File", default=False,
		description="Allow replacing an existing umbilicus JSON file",
	)

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.type == 'MESH'

	def draw(self, context):
		layout = self.layout
		layout.prop(self, "coordinate_space")
		layout.prop(self, "voxel_size")
		layout.prop(self, "overwrite")

	def invoke(self, context, event):
		obj = context.active_object
		self.coordinate_space = obj.get("velend_umbilicus_coordinate_space", 'SCENE')
		self.voxel_size = float(obj.get(
			"velend_umbilicus_voxel_size",
			_coordinate_voxel_size(context, self.coordinate_space),
		))
		source = obj.get("velend_umbilicus_path")
		self.filepath = (
			os.path.splitext(str(source))[0] + ".json" if source else
			os.path.join(os.getcwd(), bpy.path.clean_name(obj.name) + "_umbilicus.json")
		)
		context.window_manager.fileselect_add(self)
		return {'RUNNING_MODAL'}

	def execute(self, context):
		obj = context.active_object
		if not self.filepath.strip():
			self.report({'ERROR'}, "No export file selected")
			return {'CANCELLED'}
		path = os.path.expanduser(bpy.path.abspath(self.filepath.strip()))
		if os.path.splitext(path)[1].lower() != ".json":
			self.report({'ERROR'}, "Umbilicus export must be a .json file")
			return {'CANCELLED'}
		if os.path.exists(path) and not self.overwrite:
			self.report({'ERROR'}, "Target exists; enable replacement to export")
			return {'CANCELLED'}
		if obj.modifiers:
			self.report({'ERROR'}, "Apply or remove modifiers before exporting umbilicus")
			return {'CANCELLED'}

		try:
			if obj.mode == 'EDIT':
				obj.update_from_editmode()
			mesh = obj.data
			if mesh.polygons:
				raise ValueError("the mesh must have edges only, without faces")
			points = np.empty((len(mesh.vertices), 3), dtype=np.float64)
			edges = np.empty((len(mesh.edges), 2), dtype=np.int32)
			mesh.vertices.foreach_get("co", points.ravel())
			mesh.edges.foreach_get("vertices", edges.ravel())
			scores = None
			attribute = mesh.attributes.get("score")
			if attribute is not None:
				if attribute.domain != 'POINT' or attribute.data_type != 'FLOAT':
					raise ValueError("the score attribute must be a FLOAT point attribute")
				scores = np.empty(len(mesh.vertices), dtype=np.float32)
				attribute.data.foreach_get("value", scores)
			world = np.asarray(obj.matrix_world, dtype=np.float64)
			points = points @ world[:3, :3].T + world[:3, 3]
			stored = obj.get("velend_umbilicus_placement")
			original_space = obj.get("velend_umbilicus_coordinate_space")
			original_size = obj.get("velend_umbilicus_voxel_size")
			if (stored is not None and self.coordinate_space == original_space
					and original_size is not None
					and math.isclose(self.voxel_size, float(original_size), rel_tol=1e-6)):
				placement = np.asarray(stored, dtype=np.float64).reshape(4, 4)
			else:
				placement = _placement(context, self.voxel_size, self.coordinate_space).matrix
			inverse = np.linalg.inv(placement)
			points = points @ inverse[:3, :3].T + inverse[:3, 3]
			metadata = json.loads(obj.get("velend_umbilicus_metadata", "{}"))
			settings = context.scene.velend
			source_volume = volume_metadata.source_volume_for(
				settings.scene_volume_id, settings.source_url, settings.volume_path
			)
			metadata = umbilicus.export_metadata(metadata, source_volume)
			umbilicus.write_umbilicus(
				path, points, edges, self.voxel_size, metadata, scores=scores
			)
		except Exception as error:
			self.report({'ERROR'}, "Could not export umbilicus: %s" % error)
			return {'CANCELLED'}
		self.report({'INFO'}, "Exported %d umbilicus points" % len(points))
		return {'FINISHED'}


def _view_menu(self, context):
	self.layout.operator(velend_OT_load_hires.bl_idname)


def _object_menu(self, context):
	if any(obj.type == 'MESH' for obj in context.selected_objects):
		self.layout.operator(velend_OT_setup_tifxyz_uv_aspect.bl_idname)


def _mesh_menu(self, context):
	self.layout.operator(velend_OT_fill_tifxyz_holes.bl_idname)


def _export_menu(self, context):
	self.layout.operator(
		velend_OT_export_tifxyz.bl_idname,
		text="Volume Cartographer Surface (tifxyz)",
	)
	self.layout.operator(
		velend_OT_export_umbilicus.bl_idname,
		text="Scroll Umbilicus (umbilicus.json)",
	)


def _import_menu(self, context):
	self.layout.operator(
		velend_OT_import_tifxyz.bl_idname,
		text="Volume Cartographer Surface (tifxyz)",
	)
	self.layout.operator(
		velend_OT_import_umbilicus.bl_idname,
		text="Scroll Umbilicus (umbilicus.json)",
	)


def register():
	bpy.utils.register_class(velend_OT_load_hires)
	bpy.utils.register_class(velend_OT_reload_volume)
	bpy.utils.register_class(velend_OT_choose_volpkg)
	bpy.utils.register_class(velend_OT_set_volpkg_volume)
	bpy.utils.register_class(velend_OT_setup_scene)
	bpy.utils.register_class(velend_OT_import_tifxyz)
	bpy.utils.register_class(velend_OT_setup_tifxyz_uv_aspect)
	bpy.utils.register_class(velend_OT_fill_tifxyz_holes)
	bpy.utils.register_class(velend_OT_export_tifxyz)
	bpy.utils.register_class(velend_OT_import_umbilicus)
	bpy.utils.register_class(velend_OT_export_umbilicus)
	bpy.types.VIEW3D_MT_view.append(_view_menu)
	bpy.types.VIEW3D_MT_object.append(_object_menu)
	bpy.types.VIEW3D_MT_edit_mesh.append(_mesh_menu)
	bpy.types.TOPBAR_MT_file_import.append(_import_menu)
	bpy.types.TOPBAR_MT_file_export.append(_export_menu)


def unregister():
	bpy.types.TOPBAR_MT_file_export.remove(_export_menu)
	bpy.types.TOPBAR_MT_file_import.remove(_import_menu)
	bpy.types.VIEW3D_MT_object.remove(_object_menu)
	bpy.types.VIEW3D_MT_edit_mesh.remove(_mesh_menu)
	bpy.types.VIEW3D_MT_view.remove(_view_menu)
	bpy.utils.unregister_class(velend_OT_export_umbilicus)
	bpy.utils.unregister_class(velend_OT_import_umbilicus)
	bpy.utils.unregister_class(velend_OT_export_tifxyz)
	bpy.utils.unregister_class(velend_OT_fill_tifxyz_holes)
	bpy.utils.unregister_class(velend_OT_setup_tifxyz_uv_aspect)
	bpy.utils.unregister_class(velend_OT_import_tifxyz)
	bpy.utils.unregister_class(velend_OT_setup_scene)
	bpy.utils.unregister_class(velend_OT_set_volpkg_volume)
	bpy.utils.unregister_class(velend_OT_choose_volpkg)
	bpy.utils.unregister_class(velend_OT_reload_volume)
	bpy.utils.unregister_class(velend_OT_load_hires)
