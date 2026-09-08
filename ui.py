"""Scene settings: which volume to render, and how big its voxels are.

Blender saves these with the .blend, so a file set up against one scan reopens
against that same scan. "File > Defaults > Save Startup File" makes the current
pair the default for new files.
"""

import os
import re

import bpy

from . import state
from .renderer import VolumeSamplerRenderEngine


# Volume directories are named like
# "20250820131727-9.362um-1.2m-113keV-masked.zarr-adf63bbdf658dd8f". The
# OME-Zarr metadata only records the relative scale of each pyramid level, so
# that µm figure is the only place the voxel size is written down.
_RESOLUTION_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)um")


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


class VelendSceneSettings(bpy.types.PropertyGroup):
	volume_path: bpy.props.StringProperty(
		name="Volume",
		description="OME-Zarr directory holding the multiresolution volume",
		subtype='DIR_PATH',
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
		# below would be indented into the value column with them.
		column = layout.column()
		column.use_property_split = True
		column.prop(settings, "volume_path")
		column.prop(settings, "resolution")

		layout.operator("velend.setup_scene", icon='SCENE_DATA')

		status, icon = VolumeSamplerRenderEngine.status()
		layout.label(text=status, icon=icon)

		row = layout.row(align=True)
		row.operator("velend.load_hires")
		row.operator("velend.reload_volume", text="", icon='FILE_REFRESH')


def register():
	bpy.utils.register_class(VelendSceneSettings)
	bpy.utils.register_class(SCENE_PT_velend)
	bpy.types.Scene.velend = bpy.props.PointerProperty(type=VelendSceneSettings)


def unregister():
	del bpy.types.Scene.velend
	bpy.utils.unregister_class(SCENE_PT_velend)
	bpy.utils.unregister_class(VelendSceneSettings)
