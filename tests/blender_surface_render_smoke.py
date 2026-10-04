"""Foreground Blender GPU smoke test for the tiled tifxyz PNG exporter."""

import os
import struct
import sys
import tempfile
import time
import traceback
from pathlib import Path

if os.environ.get('VELEND_TEST_DEPS'):
	sys.path.insert(0, os.environ['VELEND_TEST_DEPS'])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bpy
import numpy as np
import zarr

import velend
from velend import bricks
from velend import surface_images
from velend.renderer import VolumeSamplerRenderEngine as Engine
from velend.surface_output import PNG_SIGNATURE, read_velend_metadata


for module in velend._modules:
	if hasattr(module, 'register'):
		module.register()

bricks.SLOTS_PER_AXIS = {level: 1 for level in bricks.LEVELS}
bricks.SLOT_COUNTS = {level: 1 for level in bricks.LEVELS}
bricks.ATLAS_DIMS = {level: bricks.BRICK_SIZE for level in bricks.LEVELS}

temporary = tempfile.TemporaryDirectory()
root = Path(temporary.name)
volume_path = root / 'volume-9.362um.zarr'
output_path = root / 'surface.png'
texture_output_path = root / 'surface-texture.png'
overlay_output_path = root / 'surface-overlay.png'
group = zarr.open_group(volume_path, mode='w', zarr_format=2)
group.attrs['multiscales'] = [{'datasets': [{'path': str(level)} for level in range(6)]}]
for level in range(6):
	array = group.create_array(
		str(level), shape=(64, 64, 64), chunks=(32, 32, 32), dtype='u1'
	)
	array[:] = 80

bpy.context.scene.velend.volume_path = str(volume_path)
Engine.get_volume()
phase = 0
started = time.monotonic()


def png_size(path):
	with open(path, 'rb') as file:
		assert file.read(8) == PNG_SIGNATURE
		length = struct.unpack('>I', file.read(4))[0]
		assert file.read(4) == b'IHDR' and length == 13
		return struct.unpack('>II', file.read(8))


def tick():
	global phase
	try:
		if time.monotonic() - started > 30:
			raise RuntimeError('surface render smoke test timed out')
		if phase == 0:
			if Engine.get_volume() is None:
				return 0.1
			assert bpy.ops.velend.setup_scene() == {'FINISHED'}
			phase = 1
			return 1.0
		if phase == 1:
			obj = bpy.data.objects['Cut Z']
			bpy.context.view_layer.objects.active = obj
			obj.select_set(True)
			uvs = obj.data.uv_layers.new(name='UVMap')
			for loop, uv in zip(uvs.data, ((0, 0), (1, 0), (1, 1), (0, 1))):
				loop.uv = uv
			obj['velend_tifxyz_shape'] = (8, 12)
			obj['velend_tifxyz_scale'] = (1.0, 1.0)
			obj['velend_tifxyz_voxel_size'] = 9.362
			result = bpy.ops.velend.render_tifxyz(
				'EXEC_DEFAULT', filepath=str(output_path), pixel_size_um=9.362
			)
			assert result == {'RUNNING_MODAL'}, result
			phase = 2
			return 0.1
		if phase == 2 and not output_path.exists():
			if not Engine.exporting:
				raise RuntimeError('surface render operator stopped without a PNG')
			return 0.1
		if phase == 2:
			assert png_size(output_path) == (64, 64)
			metadata = read_velend_metadata(output_path)
			assert metadata['schema'] == 'velend.surface_render.v1'
			assert abs(metadata['image']['pixel_size_um'] - 9.362) < 1e-5
			assert metadata['image']['volume_level'] == 0
			assert metadata['image']['sizing'] == 'uv_physical_rms_stretch'
			image = bpy.data.images.load(str(output_path), check_existing=False)
			pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(64, 64, 4)
			interior = pixels[1:-1, 1:-1]
			assert interior[..., 3].min() > 0.99
			assert np.ptp(interior[..., :3]) < 0.03
			assert interior[..., :3].mean() > 0.01
			material = bpy.context.active_object.active_material
			node = surface_images.selected_node(material)
			assert node is not None and node.image.filepath == str(output_path)
			surface_images.connect_for_render(material, node)
			material.velend_render_mode = 'TEXTURE'
			bpy.context.scene.velend.volume_path = ""
			result = bpy.ops.velend.render_tifxyz(
				'EXEC_DEFAULT', filepath=str(texture_output_path), pixel_size_um=9.362
			)
			assert result == {'RUNNING_MODAL'}, result
			phase = 3
			return 0.1
		if phase == 3 and not texture_output_path.exists():
			if not Engine.exporting:
				raise RuntimeError('texture surface render stopped without a PNG')
			return 0.1
		if phase == 3:
			texture_metadata = read_velend_metadata(texture_output_path)
			assert texture_metadata['material']['mode'] == 'texture'
			assert texture_metadata['material']['image']['name']
			assert png_size(texture_output_path) == (64, 64)
			material = bpy.context.active_object.active_material
			material.velend_render_mode = 'OVERLAY'
			material.velend_texture_opacity = 0.5
			bpy.context.scene.velend.volume_path = str(volume_path)
			result = bpy.ops.velend.render_tifxyz(
				'EXEC_DEFAULT', filepath=str(overlay_output_path), pixel_size_um=9.362
			)
			assert result == {'RUNNING_MODAL'}, result
			phase = 4
			return 0.1
		if not overlay_output_path.exists():
			if not Engine.exporting:
				raise RuntimeError('overlay surface render stopped without a PNG')
			return 0.1
		overlay_metadata = read_velend_metadata(overlay_output_path)
		assert overlay_metadata['material']['mode'] == 'overlay'
		assert overlay_metadata['material']['opacity'] == 0.5
		print('VELEND SURFACE PNG PASSED', flush=True)
		bpy.ops.wm.quit_blender()
	except Exception:
		traceback.print_exc()
		sys.stdout.flush()
		sys.stderr.flush()
		os._exit(1)
	return None


bpy.app.timers.register(tick, first_interval=0.1)
