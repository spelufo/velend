"""Scene settings: which volume to render, and how big its voxels are.

Blender saves these with the .blend, so a file set up against one scan reopens
against that same scan. "File > Defaults > Save Startup File" makes the current
pair the default for new files.
"""

import json
import math
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


# The state a switch is put back to when it turns out to be one to ask about:
# the fields and what the scene's coordinates mean, from before the change
# being applied now. `None` when no check is waiting to run.
_pending = None

# Set while the fields are written by the check below rather than by whoever
# is switching volumes, so that their update callback knows it is not one.
_updating = False


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


def _remember(settings):
	"""Keep what the fields now name, so that a switch made from here can be
	put back. Blender hands an update callback the new value only."""
	settings.last_volume_path = settings.volume_path
	settings.last_source_url = settings.source_url


def _reopen(settings):
	"""Take the volume the fields now name as the one to render."""
	_remember(settings)
	state.close_volume()
	VolumeSamplerRenderEngine.reset()


def _set_volume(settings, volume_path, source_url, scene_volume_id, resolution):
	"""Put the fields, and what the scene's coordinates mean, at once: the way
	they were before a switch, or the way a switch that was asked about and
	confirmed leaves them."""
	global _updating
	_updating = True
	try:
		settings.volume_path = volume_path
		settings.source_url = source_url
		settings.scene_volume_id = scene_volume_id
		settings.resolution = resolution
	finally:
		_updating = False
	_reopen(settings)


def _watch_for_rescale(settings):
	"""Keep what the scene means by a coordinate, and look on the next tick at
	where this change has taken it.

	The fields are judged once they have settled rather than here: an operator
	that sets both sets them one at a time, and the update callback sees each
	on its own, with the other still naming the volume being left. What is
	kept is the state from before the change being applied now -- the `last_`
	fields only catch up once it is through.
	"""
	global _pending
	if _pending is not None:
		return
	if settings.scene_volume_id and not (
		settings.last_volume_path or settings.last_source_url
	):
		# A file saved before the two `last_` fields existed, so there is
		# nothing to put a switch back to. `_reopen` fills them in on the way
		# out of this change, so only the first switch in such a file goes
		# through unasked.
		return
	# By name, since a timer outlives whatever it was handed.
	_pending = (settings.id_data.name, (
		settings.last_volume_path,
		settings.last_source_url,
		settings.scene_volume_id,
		settings.resolution,
	))
	bpy.app.timers.register(_check_rescale, first_interval=0.0)


def _rescaling_switch(settings, previous):
	"""The switch the fields now name, when making it has changed the size of
	the scene's voxels with nothing to carry what is in the scene across.
	None when it is nothing to ask about.

	Everything in a scene is placed in the voxels of the volume the scene is
	in. A volume registered against that one renders through the matrix for
	the pair and leaves them alone; one that is not becomes the scene's own
	frame, and only its voxels being another size makes that a move.
	"""
	volume_path, source_url, scene_volume_id, resolution = previous
	if not scene_volume_id:
		# The scene was in no volume's voxels, so nothing in it was placed
		# against any.
		return None
	volume_id = metadata.volume_id_for(settings.source_url, settings.volume_path)
	if volume_id == scene_volume_id:
		return None
	if metadata.volume_transform(scene_volume_id, volume_id) is not None:
		return None
	if math.isclose(settings.resolution, resolution, rel_tol=1e-6):
		# The new volume's voxels are the size the scene was already in, or
		# nothing states their size and the scene kept the one it had.
		return None
	return {
		"volume_path": settings.volume_path,
		"source_url": settings.source_url,
		"volume_id": volume_id,
		"resolution": settings.resolution,
	}


def _check_rescale():
	"""Put such a switch back and ask about it instead. A timer, so that it
	runs once whoever was writing the fields is done with them."""
	global _pending
	if _pending is None:
		return
	(scene_name, previous), _pending = _pending, None
	scene = bpy.data.scenes.get(scene_name)
	if scene is None:
		return
	settings = scene.velend
	target = _rescaling_switch(settings, previous)
	if target is None:
		return
	if bpy.app.background:
		# No one to ask, and a script that set the fields meant to. Say what
		# it did rather than standing in its way.
		print("velend: nothing registers %s against %s, so the scene is now "
			"in %g um voxels, from %g um" % (
				target["volume_id"] or "the volume",
				previous[2],  # the volume the scene was in
				target["resolution"],
				previous[3],  # and the size of its voxels
			))
		return
	_set_volume(settings, *previous)
	bpy.ops.velend.confirm_volume_switch('INVOKE_DEFAULT', **target)


def _volume_path_updated(self, context):
	if _updating:
		return
	_watch_for_rescale(self)
	volume_id = metadata.volume_id_for(self.source_url, self.volume_path)
	if metadata.volume_transform(self.scene_volume_id, volume_id) is None:
		# Nothing in the manifest relates the two, so nothing says how the new
		# volume sits against the scene's coordinates. Its own voxels become
		# them, the way the first volume loaded did.
		self.scene_volume_id = volume_id
	if self.scene_volume_id == volume_id:
		resolution = _volume_resolution(self, volume_id)
		if resolution is not None:
			# Assigning fires `_resolution_updated` too, whose work the reset
			# below redoes; harmless, as nothing is loaded until the next draw
			# either way.
			self.resolution = resolution
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
	# reads the voxel size out of the URL. The manifest states it outright,
	# so take it from there instead -- unless that update left the scene in
	# another volume's voxels, which are the ones Voxel Size then states.
	self.source_url = ""
	self.volume_path = volume.zarr_url
	if volume.pixel_size_um and self.scene_volume_id == volume.id:
		self.resolution = volume.pixel_size_um


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


class velend_OT_confirm_volume_switch(bpy.types.Operator):
	bl_idname = "velend.confirm_volume_switch"
	bl_label = "Switch Volume"
	bl_description = (
		"Take the scene into the voxels of a volume nothing registers against "
		"the one the scene is in"
	)
	bl_options = {'INTERNAL'}

	# The switch `_check_rescale` put back, waiting on the answer.
	volume_path: bpy.props.StringProperty(options={'SKIP_SAVE', 'HIDDEN'})
	source_url: bpy.props.StringProperty(options={'SKIP_SAVE', 'HIDDEN'})
	volume_id: bpy.props.StringProperty(options={'SKIP_SAVE', 'HIDDEN'})
	resolution: bpy.props.FloatProperty(options={'SKIP_SAVE', 'HIDDEN'})

	def invoke(self, context, event):
		return context.window_manager.invoke_props_dialog(
			self,
			width=460,
			title="Volumes Not Registered",
			confirm_text="Switch Anyway",
			# Keeping the scene as it is is the safe answer, so it is the one
			# Return takes.
			cancel_default=True,
		)

	def draw(self, context):
		settings = context.scene.velend
		layout = self.layout
		column = layout.column()
		# Read every redraw rather than once: a transform written into the
		# extra metadata and reloaded from here changes the answer, and what
		# the buttons below then do.
		registered = metadata.volume_transform(
			settings.scene_volume_id, self.volume_id
		) is not None
		# One label per line: the dialog does not wrap what it is given.
		if registered:
			column.label(
				text="%s is registered against %s now." % (
					self.volume_id or "The volume", settings.scene_volume_id
				),
				icon='CHECKMARK',
			)
			column.label(text=(
				"Switching renders it through that transform, so the scene "
				"stays in"
			))
			column.label(text=(
				"%g um voxels and nothing already placed moves."
				% settings.resolution
			))
			column.separator()
		else:
			column.label(
				text="Nothing states how %s sits against %s." % (
					self.volume_id or "the volume", settings.scene_volume_id
				),
				icon='ERROR',
			)
			column.label(text=(
				"Switching puts the scene in %g um voxels, from %g um, so "
				"meshes," % (self.resolution, settings.resolution)
			))
			column.label(text=(
				"the cursor and everything else already placed would move and "
				"change size."
			))
			column.separator()
			column.label(text=(
				"A transform for the pair, written into the extra metadata, "
				"would carry"
			))
			column.label(text="them across instead.")
		# Both buttons stay put whichever of the two it is: they are what the
		# dialog is left open to use, and one that moves out from under the
		# pointer between redraws is one pressed by accident.
		edit = column.operator("velend.edit_metadata_overrides", icon='TEXT')
		edit.from_volume_id = settings.scene_volume_id
		edit.to_volume_id = self.volume_id
		# The scale the pair would differ by if that were all they differ by.
		edit.scale = settings.resolution / self.resolution if self.resolution else 1.0
		reload = column.operator("velend.reload_metadata", icon='FILE_REFRESH')
		reload.from_volume_id = settings.scene_volume_id
		reload.to_volume_id = self.volume_id

	def execute(self, context):
		settings = context.scene.velend
		if metadata.volume_transform(settings.scene_volume_id, self.volume_id) is not None:
			# A transform for the pair turned up while the dialog stood open.
			# There is nothing left to confirm: the scene stays where it is and
			# the new volume renders through the matrix, as it would have had
			# the transform been there all along.
			_set_volume(
				settings,
				self.volume_path,
				self.source_url,
				settings.scene_volume_id,
				settings.resolution,
			)
			return {'FINISHED'}
		_set_volume(
			settings,
			self.volume_path,
			self.source_url,
			# The scene takes the new volume's voxels for its own, there being
			# nothing that relates them to the ones it was in.
			self.volume_id,
			self.resolution or settings.resolution,
		)
		return {'FINISHED'}

	def cancel(self, context):
		# The fields are already back where they were, so turning the switch
		# down takes nothing. The extra metadata is read again anyway, in case
		# the dialog was dismissed after editing it rather than reloading it
		# from here: cheap, and it lets a second try find the transform.
		metadata.load_in_background()


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
			"The volume whose voxels the scene's own coordinates are in, which "
			"is also what Voxel Size states. Another volume of the same sample "
			"renders through the transform the metadata registers for the pair, "
			"so that meshes placed against one stay put in the other"
		),
		options=set(),
	)
	source_url: bpy.props.StringProperty(
		name="Source URL",
		description="Public HTTP(S) Zarr root for missing local mirror objects",
		default="",
		update=_volume_path_updated,
	)
	last_volume_path: bpy.props.StringProperty(options={'HIDDEN'})
	last_source_url: bpy.props.StringProperty(
		# What the two fields above last named and we let through, so a switch
		# the user is asked about and turns down can be put back. Kept with the
		# .blend rather than in the module, which a script reload empties.
		options={'HIDDEN'},
	)
	resolution: bpy.props.FloatProperty(
		name="Voxel Size",
		description=(
			"Width of a full resolution voxel of the volume the scene's "
			"coordinates are in, in micrometers. Filled in from that volume's "
			"directory name when it names it"
		),
		default=9.362,
		min=1e-6,
		soft_max=100.0,
		precision=3,
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
		column.prop(settings, "resolution")

		# Fills the three fields above in from a VC3D project, so it sits with
		# them rather than with the buttons that act on what they name.
		layout.operator("velend.choose_volpkg", icon='FILE_FOLDER')

		layout.operator("velend.setup_scene", icon='SCENE_DATA')
		row = layout.row(align=True)
		row.operator("velend.import_tifxyz", text="Import Surface", icon='IMPORT')
		row.operator("velend.import_umbilicus", text="Import Umbilicus", icon='IMPORT')

		status, icon = VolumeSamplerRenderEngine.status()
		layout.label(text=status, icon=icon)
		if settings.scene_volume_id and settings.scene_volume_id != metadata.volume_id_for(
			settings.source_url, settings.volume_path
		):
			layout.label(
				text="Scene is in %s voxels" % settings.scene_volume_id,
				icon='ORIENTATION_LOCAL',
			)

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


def register():
	bpy.utils.register_class(VelendPreferences)
	bpy.utils.register_class(velend_OT_edit_metadata_overrides)
	bpy.utils.register_class(velend_OT_reload_metadata)
	bpy.utils.register_class(velend_OT_confirm_volume_switch)
	bpy.utils.register_class(VelendSceneSettings)
	bpy.utils.register_class(SCENE_PT_velend)
	bpy.types.Scene.velend = bpy.props.PointerProperty(type=VelendSceneSettings)


def unregister():
	del bpy.types.Scene.velend
	bpy.utils.unregister_class(SCENE_PT_velend)
	bpy.utils.unregister_class(VelendSceneSettings)
	bpy.utils.unregister_class(velend_OT_confirm_volume_switch)
	bpy.utils.unregister_class(velend_OT_reload_metadata)
	bpy.utils.unregister_class(velend_OT_edit_metadata_overrides)
	bpy.utils.unregister_class(VelendPreferences)
