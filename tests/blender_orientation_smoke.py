"""Run with Blender --background --factory-startup --python-exit-code 1 --python.

Real-world scene orientation and editable 3D-cursor volume coordinates.
Temporary data only; VELEND_TEST_DEPS may point at unpacked dependency wheels.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

if os.environ.get('VELEND_TEST_DEPS'):
    sys.path.insert(0, os.environ['VELEND_TEST_DEPS'])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bpy
import numpy as np
import zarr

import velend
from velend import metadata, tifxyz
from velend.renderer import VolumeSamplerRenderEngine as E

for mod in velend._modules:
    if hasattr(mod, 'register'):
        mod.register()

A, B = '20260101000000', '20260102000000'
MATRIX = [
    [1.0, 0.0, 0.0, 10.0],
    [0.0, 1.0, 0.0, 20.0],
    [0.0, 0.0, 1.0, 30.0],
]


def manifest():
    orientation = {
        'pixel_size_um': 1000.0,
        'shape': [6, 4, 2],
        'left_handed_coordinates': False,
        'z_direction_is_top_to_bottom': True,
    }
    return {'samples': {'PHercOrientation': {
        'sample': {'id': 'PHercOrientation', 'properties': {
            'type': 'scroll',
            'volume_transforms': [{'from_volume_id': A, 'transforms': [
                {'to_volume_id': B, 'matrix': MATRIX},
            ]}],
        }},
        'scans': {},
        'volumes': {
            A: {'id': A, 'long_id': A + '-1000um.zarr', 'properties': orientation},
            B: {'id': B, 'long_id': B + '-1000um.zarr', 'properties': orientation},
        },
        'segments': {},
    }}}


def write_volume(path):
    group = zarr.open_group(path, mode='w', zarr_format=2)
    group.attrs['multiscales'] = [{'datasets': [{'path': str(i)} for i in range(6)]}]
    for level in range(6):
        array = group.create_array(
            str(level), shape=(6, 4, 2), chunks=(2, 2, 2), dtype='u1')
        array[:] = 80
    return path


def load(path):
    bpy.context.scene.velend.volume_path = str(path)
    deadline = time.monotonic() + 10
    while E.get_volume() is None:
        assert time.monotonic() < deadline, E.status()
        time.sleep(0.01)
    E.apply_reset()
    E.ensure_grid()


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    manifest_path = root / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest()))
    assert metadata.load(str(manifest_path), refresh=False)
    volume_a = write_volume(root / (A + '-1000um.zarr'))
    volume_b = write_volume(root / (B + '-1000um.zarr'))

    settings = bpy.context.scene.velend
    assert settings.left_handed_coordinates is False
    assert settings.z_direction_is_top_to_bottom is True
    load(volume_a)
    assert settings.scene_volume_id == A
    assert settings.left_handed_coordinates is False
    assert settings.z_direction_is_top_to_bottom is True
    assert bpy.ops.velend.setup_scene() == {'FINISHED'}
    assert tuple(settings.scene_shape_xyz) == (2, 4, 6)

    expected_orientation = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 4.0],
        [0.0, 0.0, -1.0, 6.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    np.testing.assert_allclose(E.scene_from_voxels(), expected_orientation)
    low, high = E.world_bounds()
    np.testing.assert_allclose(low, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(high, [2.0, 4.0, 6.0])

    # Both fields are editable views of Blender's one world-space cursor.
    settings.scene_cursor_voxels = (0.25, 1.5, 2.0)
    np.testing.assert_allclose(bpy.context.scene.cursor.location, [0.25, 2.5, 4.0])
    np.testing.assert_allclose(settings.scene_cursor_voxels, [0.25, 1.5, 2.0])

    # A surface in raw A voxels receives the same orientation and round-trips.
    surface_dir = root / 'surface'
    points = np.array([
        [[0.25, 1.0, 2.0], [0.75, 1.0, 2.0]],
        [[0.25, 2.0, 2.0], [0.75, 2.0, 2.0]],
    ], dtype=np.float32)
    tifxyz.write_surface(
        surface_dir, points, np.ones((2, 2), dtype=np.float32), {}, {},
        'oriented-surface', (1.0, 1.0),
    )
    assert bpy.ops.velend.import_tifxyz(
        directory=str(surface_dir), voxel_size=1000.0) == {'FINISHED'}
    surface = bpy.context.active_object
    np.testing.assert_allclose(surface.data.vertices[0].co, [0.25, 3.0, 4.0])
    exported_dir = root / 'exported'
    assert bpy.ops.velend.export_tifxyz(filepath=str(exported_dir)) == {'FINISHED'}
    exported = np.stack([
        tifxyz.read_page(exported_dir / (axis + '.tif')) for axis in 'xyz'
    ], axis=-1)
    np.testing.assert_allclose(exported, points, atol=1e-6)

    # The additional current-volume vector uses the pair's registration.
    load(volume_b)
    settings.scene_cursor_voxels = (1.0, 2.0, 3.0)
    np.testing.assert_allclose(settings.current_cursor_voxels, [11.0, 22.0, 33.0])
    settings.current_cursor_voxels = (12.0, 23.0, 34.0)
    np.testing.assert_allclose(settings.scene_cursor_voxels, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(bpy.context.scene.cursor.location, [2.0, 1.0, 2.0])

    # Manual changes persist and replace the corresponding matrix immediately.
    settings.z_direction_is_top_to_bottom = False
    assert settings.orientation_user_set
    np.testing.assert_allclose(E.scene_from_voxels()[:3, :3], np.eye(3))

    saved = root / 'oriented.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(saved))
    bpy.ops.wm.open_mainfile(filepath=str(saved))
    settings = bpy.context.scene.velend
    assert settings.left_handed_coordinates is False
    assert settings.z_direction_is_top_to_bottom is False
    assert settings.orientation_user_set
    assert tuple(settings.scene_shape_xyz) == (2, 4, 6)

for mod in reversed(velend._modules):
    if hasattr(mod, 'unregister'):
        mod.unregister()
print('VELEND ORIENTATION SMOKE PASSED')
