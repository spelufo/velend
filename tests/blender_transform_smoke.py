"""Run with Blender --background --factory-startup --python-exit-code 1 --python.

Switching between two volumes of one sample, and the affine that keeps the
scene's coordinates meaning the same place in both. Temporary data only;
VELEND_TEST_DEPS may point at unpacked dependency wheels.
"""
import sys
from pathlib import Path
import os
if os.environ.get('VELEND_TEST_DEPS'):
    sys.path.insert(0, os.environ['VELEND_TEST_DEPS'])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bpy, json, numpy as np, tempfile, time, zarr
import velend
from velend import metadata
from velend.renderer import VolumeSamplerRenderEngine as E
# Register modules directly to avoid downloading the catalogue.
for mod in velend._modules:
    if hasattr(mod, 'register'):
        mod.register()

A, B = '20231027191953', '20231117143551'
MATRIX = [[0.5, 0.0, 0.0, 10.0], [0.0, 0.25, 0.0, 20.0], [0.0, 0.0, 0.125, 30.0]]
M = np.eye(4)
M[:3] = MATRIX


def manifest():
    return {'samples': {'PHercTest': {
        'sample': {'id': 'PHercTest', 'properties': {'type': 'scroll',
            'volume_transforms': [{'from_volume_id': A, 'transforms': [
                {'to_volume_id': B, 'matrix': MATRIX}]}]}},
        'scans': {},
        'volumes': {
            A: {'id': A, 'long_id': A + '-3.240um.zarr'},
            B: {'id': B, 'long_id': B + '-7.910um.zarr'},
        },
        'segments': {},
    }}}


def write_volume(root, size):
    group = zarr.open_group(root, mode='w', zarr_format=2)
    group.attrs['multiscales'] = [{'datasets': [{'path': str(i)} for i in range(6)]}]
    for i in range(6):
        array = group.create_array(str(i), shape=(size,) * 3, chunks=(2, 2, 2), dtype='u1')
        array[:] = 80
    return root


def load(path):
    """Point the scene at a volume and wait for it to open."""
    bpy.context.scene.velend.volume_path = str(path)
    deadline = time.monotonic() + 10
    while E.get_volume() is None:
        assert time.monotonic() < deadline, E.status()
        time.sleep(.01)
    E.apply_reset()
    E.ensure_grid()


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / 'manifest.json').write_text(json.dumps(manifest()))
    assert metadata.load(str(root / 'manifest.json'), refresh=False)
    volume_a = write_volume(root / (A + '-3.240um.zarr'), 4)
    volume_b = write_volume(root / (B + '-7.910um.zarr'), 8)
    plain = write_volume(root / 'someone-elses.zarr', 4)
    settings = bpy.context.scene.velend

    # The first volume loads as itself: the scene's coordinates are its voxels.
    load(volume_a)
    assert settings.scene_volume_id == A, settings.scene_volume_id
    assert abs(settings.resolution - 3.240) < 1e-6, settings.resolution
    assert np.allclose(E.volume_transform, np.eye(4)), E.volume_transform
    assert bpy.ops.velend.setup_scene() == {'FINISHED'}
    cursor = tuple(bpy.context.scene.cursor.location)

    # The second renders through the matrix registered for the pair, and the
    # scene stays in the first one's voxels, voxel size included.
    load(volume_b)
    assert settings.scene_volume_id == A, settings.scene_volume_id
    assert abs(settings.resolution - 3.240) < 1e-6, settings.resolution
    assert np.allclose(E.volume_transform, M), E.volume_transform
    assert np.allclose(E.to_volume_voxels(np.array([2., 4., 6.])), [11., 21., 30.75])
    assert np.allclose(
        E.to_volume_voxels(np.array([[0., 0., 0.], [2., 4., 6.]])),
        [[10., 20., 30.], [11., 21., 30.75]])
    # Volume B spans 8 voxels of its own per axis, which the matrix scales up
    # by 2, 4 and 8 in the scene's.
    low, high = E.scene_voxel_bounds()
    assert np.allclose(low, [-20., -80., -240.]), low
    assert np.allclose(high, [-4., -48., -176.]), high
    # Nothing moved the geometry: it means the same place in both volumes.
    assert tuple(bpy.context.scene.cursor.location) == cursor

    # And back, through the inverse of the only direction the sample states.
    load(volume_a)
    assert settings.scene_volume_id == A
    assert np.allclose(E.volume_transform, np.eye(4)), E.volume_transform

    # A volume nothing relates to the scene's becomes the scene's own frame.
    load(plain)
    assert settings.scene_volume_id == '', settings.scene_volume_id
    assert np.allclose(E.volume_transform, np.eye(4)), E.volume_transform
    load(volume_b)
    assert settings.scene_volume_id == B, settings.scene_volume_id
    assert abs(settings.resolution - 7.910) < 1e-6, settings.resolution
    assert np.allclose(E.volume_transform, np.eye(4)), E.volume_transform

print('VELEND TRANSFORM SMOKE PASSED')
