"""Scene settings: which volume to render, and how big its voxels are.

Blender saves these with the .blend, so a file set up against one scan reopens
against that same scan. "File > Defaults > Save Startup File" makes the current
pair the default for new files.
"""

import os
import re

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


def _volume_path_updated(self, context):
	resolution = resolution_from_path(self.volume_path)
	if resolution is not None:
		# Assigning fires `_resolution_updated` too, whose work the reset below
		# redoes; harmless, as nothing is loaded until the next draw either way.
		self.resolution = resolution
	state.close_volume()
	VolumeSamplerRenderEngine.reset()


def _resolution_updated(self, context):
	VolumeSamplerRenderEngine.rescale()


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
	# so take it from there instead.
	self.source_url = ""
	self.volume_path = volume.zarr_url
	if volume.pixel_size_um:
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
	source_url: bpy.props.StringProperty(
		name="Source URL",
		description="Public HTTP(S) Zarr root for missing local mirror objects",
		default="",
		update=_volume_path_updated,
	)
	resolution: bpy.props.FloatProperty(
		name="Voxel Size",
		description=(
			"Width of a full resolution voxel, in micrometers. Filled in from "
			"the volume's directory name when that names it"
		),
		default=9.362,
		min=1e-6,
		soft_max=100.0,
		precision=3,
		update=_resolution_updated,
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

		row = layout.row(align=True)
		row.operator("velend.load_hires")
		row.operator("velend.reload_volume", text="", icon='FILE_REFRESH')


def register():
	bpy.utils.register_class(VelendPreferences)
	bpy.utils.register_class(VelendSceneSettings)
	bpy.utils.register_class(SCENE_PT_velend)
	bpy.types.Scene.velend = bpy.props.PointerProperty(type=VelendSceneSettings)


def unregister():
	del bpy.types.Scene.velend
	bpy.utils.unregister_class(SCENE_PT_velend)
	bpy.utils.unregister_class(VelendSceneSettings)
	bpy.utils.unregister_class(VelendPreferences)
