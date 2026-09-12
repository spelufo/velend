"""Scene settings: which volume to render, and how big its voxels are.

Blender saves these with the .blend, so a file set up against one scan reopens
against that same scan. "File > Defaults > Save Startup File" makes the current
pair the default for new files.
"""

import json
import os
import re
import shlex
import subprocess

import bpy

from . import metadata
from . import state
from .renderer import VolumeSamplerRenderEngine


# Volume directories are named like
# "20250820131727-9.362um-1.2m-113keV-masked.zarr-adf63bbdf658dd8f". The
# OME-Zarr metadata only records the relative scale of each pyramid level, so
# that µm figure is the only place the voxel size is written down.
_RESOLUTION_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)um")

# The catalogue selectors are useful for inspecting metadata, but are not part
# of the current workflow. Keep their implementation ready for when they are.
_SHOW_METADATA_DROPDOWNS = False


def resolution_from_path(path):
	"""The voxel size named by a volume directory, or None if it names none."""
	match = _RESOLUTION_RE.search(os.path.basename(os.path.normpath(path)))
	return float(match.group(1)) if match else None


# The scene, by name, whose volume is waiting to be asked the voxel size of, or
# None when nothing is waiting. By name, since a timer outlives whatever it was
# handed.
_pending_ask = None


def _volume_resolution(settings, volume_id):
	"""The voxel size of the volume the fields name, in micrometers, or None
	when nothing states it.

	The catalogue first, for a volume it lists, then the locations the id
	itself is read off, in the order `metadata.volume_id_for` reads them: a
	directory or URL naming one volume and a voxel size names them both.
	"""
	volume = metadata.VOLUMES.get(volume_id)
	if volume is not None and volume.pixel_size_um:
		return volume.pixel_size_um
	for location in (settings.source_url, settings.volume_path):
		resolution = resolution_from_path(location)
		if resolution is not None:
			return resolution
	return None


def _reopen(settings):
	"""Take the volume the fields now name as the one to render."""
	state.close_volume()
	VolumeSamplerRenderEngine.reset()


def _anchored(settings):
	"""Whether the volume being rendered is the one the scene's coordinates are
	in, in which case the two voxel sizes are the one number."""
	return settings.scene_volume_id == metadata.volume_id_for(
		settings.source_url, settings.volume_path
	)


def set_resolution(settings, resolution):
	"""State how wide a voxel of the volume being rendered is.

	The scene's own frame follows it while that volume is the one the scene is
	in: the two are the same voxel then, and nothing else states its size.
	Anything waiting to be asked has its answer in this.
	"""
	global _pending_ask
	_pending_ask = None
	# Assigning fires `_resolution_updated`, which reapplies both to the grid.
	settings.resolution = resolution
	if _anchored(settings):
		settings.scene_resolution = resolution


def _ask_resolution(settings):
	"""Ask how wide the volume's voxels are, on the next tick.

	A timer, so that it runs once whoever is writing the fields is done with
	them: an operator that sets both sets them one at a time, and the update
	callback sees each on its own, with the other still naming the volume being
	left -- which is a volume nothing is being asked about.
	"""
	global _pending_ask
	if _pending_ask is not None:
		return
	_pending_ask = settings.id_data.name
	bpy.app.timers.register(_check_resolution, first_interval=0.0)


def _check_resolution():
	global _pending_ask
	scene_name, _pending_ask = _pending_ask, None
	if scene_name is None:
		# Answered while this was waiting, by whoever called `set_resolution`.
		return
	scene = bpy.data.scenes.get(scene_name)
	if scene is None:
		return
	settings = scene.velend
	volume_id = metadata.volume_id_for(settings.source_url, settings.volume_path)
	resolution = _volume_resolution(settings, volume_id)
	if resolution is not None:
		# The fields settled on a volume that states its voxel size after all,
		# or the catalogue naming it finished downloading in the meantime.
		set_resolution(settings, resolution)
		return
	if bpy.app.background:
		# No one to ask. Say what the volume is being placed by instead.
		print("velend: nothing states the voxel size of %s, so it is placed at "
			"%g um, the size last set" % (
				volume_id or "the volume", settings.resolution))
		return
	bpy.ops.velend.set_resolution('INVOKE_DEFAULT')


def _volume_path_updated(self, context):
	volume_id = metadata.volume_id_for(self.source_url, self.volume_path)
	if not self.scene_resolution:
		# A file saved before the scene's frame stated a voxel size of its own,
		# where Voxel Size was the frame's rather than the rendered volume's. A
		# new file lands here too, with the property's default, which the volume
		# being loaded is about to become the frame and overwrite.
		self.scene_resolution = self.resolution
	if not self.scene_volume_id:
		# The scene is in no volume's frame yet, so the one being loaded becomes
		# it. Once it has one it keeps it: a volume the metadata cannot relate to
		# that one is placed by its own voxel size instead, which leaves what the
		# scene already holds where it is and comes out right if the pair is
		# registered later on.
		self.scene_volume_id = volume_id
	resolution = _volume_resolution(self, volume_id)
	if resolution is not None:
		set_resolution(self, resolution)
	else:
		# Nothing states how wide this volume's voxels are, and the scene cannot
		# place it without that.
		_ask_resolution(self)
	_reopen(self)


def _resolution_updated(self, context):
	VolumeSamplerRenderEngine.rescale()


def _frustum_culling_updated(self, context):
	# Not a shader define, unlike the toggle below: it only changes which bricks
	# the next working set asks for.
	VolumeSamplerRenderEngine.retarget_now()


def _debug_level_colors_updated(self, context):
	# The flag is a compile time define in the fragment shader, so the shaders
	# have to be built again for the toggle to show.
	VolumeSamplerRenderEngine.reload_shaders()


def _sampling_updated(self, context):
	# The sample count controls a shader loop and the physical distances are
	# converted to voxel-space constants when the shader is compiled.
	VolumeSamplerRenderEngine.reload_shaders()


# Blender does not copy the strings an enum callback returns, so anything they
# hand out has to outlive the call. Keeping the last list per property is the
# usual way around it; without it the menus fill with garbage.
_enum_items = {}


def _items(key, items):
	_enum_items[key] = items
	return items


def _sample_items(self, context):
	samples = sorted(metadata.SAMPLES.values(), key=lambda sample: sample.id)
	return _items("samples", [
		(sample.id, sample.id, (sample.type or "sample").capitalize())
		for sample in samples
	])


def _volume_items(self, context):
	sample = metadata.SAMPLES.get(self.sample_id)
	if sample is None:
		return _items("volumes", [])
	volumes = sorted(
		sample.volumes.values(),
		key=lambda volume: (volume.pixel_size_um or 0.0, volume.id),
	)
	return _items("volumes", [
		(volume.id, volume.long_id, _volume_description(volume))
		for volume in volumes
	])


def _volume_description(volume):
	shape = " x ".join(str(size) for size in reversed(volume.shape or ()))
	return "%s voxels of %s um, %s keV" % (
		shape or "?",
		volume.pixel_size_um or "?",
		volume.energy_keV or "?",
	)


def _segment_items(self, context):
	sample = metadata.SAMPLES.get(self.sample_id)
	if sample is None:
		return _items("segments", [])
	segments = sorted(
		sample.segments.values(), key=lambda segment: segment.id, reverse=True
	)
	return _items("segments", [
		(segment.id, segment.name or segment.long_id, _segment_description(segment))
		for segment in segments
	])


def _segment_description(segment):
	traced_in = segment.volume.long_id if segment.volume else "?"
	return "%s x %s, traced in %s" % (
		segment.width or "?", segment.height or "?", traced_in
	)


def _sample_updated(self, context):
	# The enums below hold an index into whatever their callback last returned,
	# so a new sample would otherwise leave them pointing at whichever of its
	# volumes and segments happens to sit at the old one's place.
	volumes = _volume_items(self, context)
	if volumes:
		self.volume_id = volumes[0][0]
	segments = _segment_items(self, context)
	if segments:
		self.segment_id = segments[0][0]


def _volume_updated(self, context):
	volume = metadata.VOLUMES.get(self.volume_id)
	if volume is None or not volume.zarr_url:
		return
	# Assigning fires `_volume_path_updated`, which reopens the volume and
	# takes its voxel size from the catalogue, this being a volume it lists.
	self.source_url = ""
	self.volume_path = volume.zarr_url


def _overlay_path_updated(self, context):
	# Cheap to redo: the manifest itself is cached, and the server answers the
	# freshness check with a 304 rather than 15MB of JSON.
	metadata.load_in_background()


class VelendPreferences(bpy.types.AddonPreferences):
	# Preferences are looked up by the add-on's package name, which for an
	# extension is its full "bl_ext.<repository>.velend" module path.
	bl_idname = __package__

	overlay_path: bpy.props.StringProperty(
		name="Extra Metadata",
		description=(
			"JSON file shaped like the open data metadata.json, merged over "
			"it. Anything it names wins, so it can carry volume transforms, "
			"volumes or segments that the published metadata does not have yet"
		),
		subtype='FILE_PATH',
		update=_overlay_path_updated,
	)

	def draw(self, context):
		layout = self.layout
		layout.use_property_split = True
		layout.prop(self, "overlay_path")


def _overlay_file():
	"""The extra metadata file to edit, named in the preferences and made up
	if they name none yet.

	A file of one's own rather than the cached manifest: that one is rewritten
	from the bucket whenever it changes.
	"""
	path = metadata.overlay_path()
	if path:
		return path
	directory = bpy.utils.extension_path_user(__package__, path="", create=True)
	path = os.path.join(directory, "metadata_overrides.json")
	addon = bpy.context.preferences.addons.get(__package__)
	if addon is not None:
		# Assigning reloads the catalogue through the file, so this is also
		# what makes anything written into it from now on count.
		addon.preferences.overlay_path = path
	return path


def _write_overlay_scaffold(path, from_id, to_id, scale):
	"""Start the file off with the transform it is being opened to write.

	The shape of it, and a matrix that would be the right one if the two
	volumes shared an origin and their axes and differed by voxel size alone:
	the only thing about the pair that is known here, and a placeholder to
	correct rather than an answer. The sample's published transforms come
	along because `metadata.merge` replaces a list whole, so an overlay naming
	only the new pair would take the rest of them away with it.
	"""
	volume = metadata.VOLUMES.get(from_id) or metadata.VOLUMES.get(to_id)
	sample = volume.sample if volume is not None else None
	transforms = {
		key: list(value)
		for key, value in (sample.volume_transforms.items() if sample else ())
	}
	transforms[from_id] = transforms.get(from_id, []) + [{
		"to_volume_id": to_id,
		"matrix": [
			[scale, 0.0, 0.0, 0.0],
			[0.0, scale, 0.0, 0.0],
			[0.0, 0.0, scale, 0.0],
		],
	}]
	sample_id = sample.id if sample is not None else "SAMPLE_ID"
	manifest = {"samples": {sample_id: {
		# `metadata.parse` reads all four of these off every sample. Merging
		# over one the manifest already has leaves its own in place; a sample
		# it does not know needs them spelled out, empty, to parse at all.
		"sample": {"id": sample_id, "properties": {"volume_transforms": [
			{"from_volume_id": key, "transforms": value}
			for key, value in transforms.items()
		]}},
		"scans": {},
		"volumes": {},
		"segments": {},
	}}}
	with open(path, "w") as file:
		json.dump(manifest, file, indent="\t")
		file.write("\n")


def _spawn_editor(path):
	"""$EDITOR on the file, and what was spawned, for the report.

	Blender started from a launcher rather than a shell often has neither
	$VISUAL nor $EDITOR, so the desktop's own handler for the file stands in.
	"""
	editor = (os.environ.get("VISUAL") or os.environ.get("EDITOR") or "").strip()
	if not editor:
		bpy.ops.wm.path_open(filepath=path)
		return "the system editor"
	command = shlex.split(editor)
	subprocess.Popen(command + [path])
	return os.path.basename(command[0])


class velend_OT_edit_metadata_overrides(bpy.types.Operator):
	bl_idname = "velend.edit_metadata_overrides"
	bl_label = "Edit Extra Metadata"
	bl_description = (
		"Open the extra metadata file in $EDITOR, to write the transform "
		"between the two volumes by hand. Reload Extra Metadata reads it back "
		"in once it is written, without waiting for the editor to close"
	)
	bl_options = {'INTERNAL'}

	from_volume_id: bpy.props.StringProperty(options={'SKIP_SAVE', 'HIDDEN'})
	to_volume_id: bpy.props.StringProperty(options={'SKIP_SAVE', 'HIDDEN'})
	# What the scaffold's placeholder matrix scales by: the voxels of the
	# volume the scene is in, over the voxels of the one it would switch to.
	scale: bpy.props.FloatProperty(default=1.0, options={'SKIP_SAVE', 'HIDDEN'})

	def execute(self, context):
		path = _overlay_file()
		started = not os.path.exists(path)
		try:
			if started:
				_write_overlay_scaffold(
					path, self.from_volume_id, self.to_volume_id, self.scale
				)
			editor = _spawn_editor(path)
		except (OSError, RuntimeError) as error:
			self.report({'ERROR'}, "Could not edit %s: %s" % (path, error))
			return {'CANCELLED'}
		self.report(
			{'INFO'},
			"%s %s in %s" % ("Started" if started else "Editing", path, editor),
		)
		return {'FINISHED'}


class velend_OT_reload_metadata(bpy.types.Operator):
	bl_idname = "velend.reload_metadata"
	bl_label = "Reload Extra Metadata"
	bl_description = (
		"Read the extra metadata file again, so that a transform just written "
		"into it counts. Nothing is downloaded: the catalogue it merges over "
		"is the one already cached"
	)
	bl_options = {'INTERNAL'}

	# The pair to say something about afterwards, when there is one in hand.
	from_volume_id: bpy.props.StringProperty(options={'SKIP_SAVE', 'HIDDEN'})
	to_volume_id: bpy.props.StringProperty(options={'SKIP_SAVE', 'HIDDEN'})

	def execute(self, context):
		overlay = metadata.overlay_path()
		if overlay and os.path.exists(overlay):
			# `metadata.load` merges what it can read and prints about what it
			# cannot, which is no way to hear that the file being edited is
			# half written.
			try:
				with open(overlay) as file:
					json.load(file)
			except (OSError, ValueError) as error:
				self.report({'ERROR'}, "%s: %s" % (os.path.basename(overlay), error))
				return {'CANCELLED'}
		# On this thread, unlike at startup: the answer is wanted now, by the
		# dialog the button was pressed in.
		try:
			loaded = metadata.load(overlay=overlay, refresh=False)
		except (OSError, ValueError) as error:
			self.report({'ERROR'}, "Could not read the catalogue: %s" % error)
			return {'CANCELLED'}
		if not loaded:
			self.report({'ERROR'}, "No catalogue cached for it to merge over")
			return {'CANCELLED'}
		if not (self.from_volume_id and self.to_volume_id):
			self.report({'INFO'}, "Metadata reloaded")
		elif metadata.volume_transform(self.from_volume_id, self.to_volume_id) is None:
			self.report({'WARNING'}, "Still nothing registers %s against %s" % (
				self.to_volume_id, self.from_volume_id))
		else:
			self.report({'INFO'}, "%s is registered against %s now" % (
				self.to_volume_id, self.from_volume_id))
		if context.region is not None:
			# The dialog this was pressed in says where the pair stands, and
			# that has just changed under it.
			context.region.tag_redraw()
		return {'FINISHED'}


class velend_OT_set_resolution(bpy.types.Operator):
	bl_idname = "velend.set_resolution"
	bl_label = "Set Voxel Size"
	bl_description = (
		"State how wide a full resolution voxel of the volume being rendered is. "
		"The scene asks when neither the metadata nor the volume's directory "
		"name states it, and this is how to correct what either of them says"
	)
	bl_options = {'INTERNAL'}

	resolution: bpy.props.FloatProperty(
		name="Voxel Size",
		description="Width of a full resolution voxel, in micrometers",
		default=9.362,
		min=1e-6,
		soft_max=100.0,
		precision=3,
		options={'SKIP_SAVE'},
	)

	def invoke(self, context, event):
		# What the scene is placing the volume by now, which is the last size
		# set: the one to correct, and the one to keep by pressing Return.
		self.resolution = context.scene.velend.resolution
		return context.window_manager.invoke_props_dialog(
			self, width=440, title="Voxel Size", confirm_text="Use This Size"
		)

	def draw(self, context):
		settings = context.scene.velend
		column = self.layout.column()
		volume_id = metadata.volume_id_for(settings.source_url, settings.volume_path)
		# One label per line: the dialog does not wrap what it is given.
		column.label(
			text="Nothing states how wide the voxels of %s are." % (
				volume_id or "this volume"),
			icon='ERROR',
		)
		column.label(text=(
			"A directory named like 20250820131727-9.362um-1.2m-113keV.zarr "
			"states it,"
		))
		column.label(text="and so does the open data metadata for a volume it lists.")
		column.separator()
		column.prop(self, "resolution")

	def execute(self, context):
		set_resolution(context.scene.velend, self.resolution)
		return {'FINISHED'}


class VelendSceneSettings(bpy.types.PropertyGroup):
	sample_id: bpy.props.EnumProperty(
		name="Sample",
		description="The scroll or fragment to browse",
		items=_sample_items,
		update=_sample_updated,
		options=set(),
	)
	volume_id: bpy.props.EnumProperty(
		name="Volume",
		description="Which of the sample's volumes to render",
		items=_volume_items,
		update=_volume_updated,
		options=set(),
	)
	segment_id: bpy.props.EnumProperty(
		name="Segment",
		description="One of the sample's traced sheets of papyrus",
		items=_segment_items,
		options=set(),
	)
	volume_path: bpy.props.StringProperty(
		name="Volume",
		description="OME-Zarr directory holding the multiresolution volume",
		subtype='DIR_PATH',
		update=_volume_path_updated,
	)
	scene_volume_id: bpy.props.StringProperty(
		name="Scene Volume",
		description=(
			"The volume whose frame the scene's own coordinates are in. Another "
			"volume of the same sample renders through the transform the "
			"metadata registers for the pair, so that meshes placed against one "
			"stay put in the other"
		),
		options=set(),
	)
	source_url: bpy.props.StringProperty(
		name="Source URL",
		description="Public HTTP(S) Zarr root for missing local mirror objects",
		default="",
		update=_volume_path_updated,
	)
	resolution: bpy.props.FloatProperty(
		name="Voxel Size",
		description=(
			"Width of a full resolution voxel of the volume being rendered, in "
			"micrometers. Taken from the open data metadata for a volume it "
			"lists, or from the volume's directory name when that names it, and "
			"asked for when neither states it. It says where in the scan the "
			"scene's coordinates fall, so it is read rather than set"
		),
		default=9.362,
		min=1e-6,
		soft_max=100.0,
		precision=3,
		update=_resolution_updated,
	)
	scene_resolution: bpy.props.FloatProperty(
		# How wide a voxel of Scene Volume is, which with it is what a
		# coordinate in this scene means. Left behind by a switch to a volume of
		# another voxel size, which is what keeps everything already placed
		# where it is. Zero in a file saved before the scene's frame stated a
		# size of its own; `_volume_path_updated` fills it in.
		name="Scene Voxel Size",
		default=0.0,
		min=0.0,
		precision=3,
		options={'HIDDEN'},
		update=_resolution_updated,
	)
	frustum_culling: bpy.props.BoolProperty(
		name="Frustum Culling",
		description=(
			"Only stream the bricks that fall inside a 3D viewport's view "
			"frustum, so the atlases and the download bandwidth go to what is "
			"on screen. Turn it off to keep the whole mesh loaded no matter "
			"where you are looking"
		),
		default=True,
		options=set(),
		update=_frustum_culling_updated,
	)
	debug_level_colors: bpy.props.BoolProperty(
		name="Level Colors",
		description=(
			"Tint each fragment by the resolution level its samples came from "
			"instead of shading it, to show what the streamer has loaded"
		),
		default=False,
		options=set(),
		update=_debug_level_colors_updated,
	)
	render_depth: bpy.props.FloatProperty(
		name="Render Depth",
		description="Total inward sampling depth, in micrometers",
		default=50.0,
		min=0.0,
		soft_max=1000.0,
		precision=3,
		update=_sampling_updated,
	)
	num_samples: bpy.props.IntProperty(
		name="Samples",
		description="Number of volume samples averaged for each surface point",
		default=5,
		min=1,
		soft_max=64,
		update=_sampling_updated,
	)
	render_depth_offset: bpy.props.FloatProperty(
		name="Depth Offset",
		description=(
			"Offset of the first sample from the surface, in micrometers; positive "
			"values move into the surface and negative values move out"
		),
		default=0.0,
		precision=3,
		update=_sampling_updated,
	)


class SCENE_PT_velend(bpy.types.Panel):
	bl_label = "Velend"
	bl_space_type = 'PROPERTIES'
	bl_region_type = 'WINDOW'
	bl_context = "scene"

	def draw(self, context):
		layout = self.layout
		settings = context.scene.velend

		# Only the fields are split into label and value columns; the buttons
		# below would be indented into the value column with them. Nothing
		# here is worth keyframing, so no animate decorators either.
		column = layout.column()
		column.use_property_split = True
		column.use_property_decorate = False
		if _SHOW_METADATA_DROPDOWNS:
			if metadata.SAMPLES:
				column.prop(settings, "sample_id")
				if settings.sample_id:
					column.prop(settings, "volume_id")
					column.prop(settings, "segment_id")
			else:
				# The catalogue downloads and parses on a thread at startup, and
				# stays empty when that failed with nothing cached to fall back on.
				column.label(text="Loading metadata...", icon='INFO')
			column.separator()
		column.prop(settings, "volume_path")
		column.prop(settings, "source_url")
		# What the volume being rendered states about itself rather than
		# something to set, so it is shown and not edited. The button beside it
		# is for a volume that states nothing, and for correcting a bad guess.
		row = column.row(align=True)
		field = row.row()
		field.enabled = False
		field.prop(settings, "resolution")
		row.operator("velend.set_resolution", text="", icon='GREASEPENCIL')

		# Fills the three fields above in from a VC3D project, so it sits with
		# them rather than with the buttons that act on what they name.
		layout.operator("velend.choose_volpkg", icon='FILE_FOLDER')

		layout.operator("velend.setup_scene", icon='SCENE_DATA')
		row = layout.row(align=True)
		row.operator("velend.import_tifxyz", text="Import Surface", icon='IMPORT')
		row.operator("velend.import_umbilicus", text="Import Umbilicus", icon='IMPORT')

		status, icon = VolumeSamplerRenderEngine.status()
		layout.label(text=status, icon=icon)
		volume_id = metadata.volume_id_for(settings.source_url, settings.volume_path)
		if settings.scene_volume_id and settings.scene_volume_id != volume_id:
			# Short lines: the panel is narrow at its default width, and a label
			# that does not fit is truncated rather than wrapped.
			box = layout.box()
			box.label(
				text="Scene is in %s" % settings.scene_volume_id,
				icon='ORIENTATION_LOCAL',
			)
			box.label(text="at %g um voxels" % settings.scene_resolution)
			if metadata.volume_transform(settings.scene_volume_id, volume_id) is None:
				# Rendering it anyway, on the only assumption there is to make
				# about two scans of one object. Saying so is the point: it is a
				# guess, and the buttons are how to replace it with a transform.
				box.separator()
				box.label(text="Nothing registers the two.", icon='ERROR')
				box.label(text="Placed as if they shared an")
				box.label(text="origin and their axes.")
				buttons = box.row(align=True)
				edit = buttons.operator("velend.edit_metadata_overrides", icon='TEXT')
				edit.from_volume_id = settings.scene_volume_id
				edit.to_volume_id = volume_id
				# The scale the pair would differ by if that were all they
				# differ by, the metadata stating transforms in voxels.
				edit.scale = (
					settings.scene_resolution / settings.resolution
					if settings.resolution else 1.0
				)
				reload = buttons.operator("velend.reload_metadata", icon='FILE_REFRESH')
				reload.from_volume_id = settings.scene_volume_id
				reload.to_volume_id = volume_id

		# A view option rather than something the volume is loaded through, so
		# it sits with the streaming controls and not with the fields above.
		debug = layout.column()
		debug.use_property_split = True
		debug.use_property_decorate = False
		debug.prop(settings, "frustum_culling")
		debug.prop(settings, "debug_level_colors")

		row = layout.row(align=True)
		row.operator("velend.load_hires")
		row.operator("velend.reload_volume", text="", icon='FILE_REFRESH')


class RENDER_PT_velend_sampling(bpy.types.Panel):
	bl_label = "Volume Sampling"
	bl_space_type = 'PROPERTIES'
	bl_region_type = 'WINDOW'
	bl_context = "render"
	COMPAT_ENGINES = {VolumeSamplerRenderEngine.bl_idname}

	@classmethod
	def poll(cls, context):
		return context.scene.render.engine in cls.COMPAT_ENGINES

	def draw(self, context):
		settings = context.scene.velend
		column = self.layout.column()
		column.use_property_split = True
		column.use_property_decorate = False
		column.prop(settings, "render_depth")
		column.prop(settings, "num_samples")
		column.prop(settings, "render_depth_offset")
		column.label(
			text="Sample Distance: %g um" % (
				settings.render_depth / max(settings.num_samples, 1)
			)
		)


@bpy.app.handlers.persistent
def _load_post(_file_path):
	"""Bring a file saved before the scene's frame stated a voxel size of its
	own up to date.

	Opening a file writes the fields without going through their update
	callbacks, so this is the only thing that notices. Voxel Size meant the
	frame's voxel size in such a file, which is what it still means here; what
	it means now, the voxel size of the volume being rendered, is read off that
	volume the way loading it would.
	"""
	for scene in bpy.data.scenes:
		settings = scene.velend
		if settings.scene_resolution:
			continue
		if not (settings.volume_path or settings.source_url):
			continue
		settings.scene_resolution = settings.resolution
		resolution = _volume_resolution(settings, metadata.volume_id_for(
			settings.source_url, settings.volume_path))
		if resolution is not None:
			settings.resolution = resolution


def register():
	bpy.utils.register_class(VelendPreferences)
	bpy.utils.register_class(velend_OT_edit_metadata_overrides)
	bpy.utils.register_class(velend_OT_reload_metadata)
	bpy.utils.register_class(velend_OT_set_resolution)
	bpy.utils.register_class(VelendSceneSettings)
	bpy.utils.register_class(SCENE_PT_velend)
	bpy.utils.register_class(RENDER_PT_velend_sampling)
	bpy.types.Scene.velend = bpy.props.PointerProperty(type=VelendSceneSettings)
	if _load_post not in bpy.app.handlers.load_post:
		bpy.app.handlers.load_post.append(_load_post)


def unregister():
	if _load_post in bpy.app.handlers.load_post:
		bpy.app.handlers.load_post.remove(_load_post)
	del bpy.types.Scene.velend
	bpy.utils.unregister_class(RENDER_PT_velend_sampling)
	bpy.utils.unregister_class(SCENE_PT_velend)
	bpy.utils.unregister_class(VelendSceneSettings)
	bpy.utils.unregister_class(velend_OT_set_resolution)
	bpy.utils.unregister_class(velend_OT_reload_metadata)
	bpy.utils.unregister_class(velend_OT_edit_metadata_overrides)
	bpy.utils.unregister_class(VelendPreferences)
