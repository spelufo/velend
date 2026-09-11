"""Run with Blender --background --factory-startup --python-exit-code 1 --python.

A volume nothing registers against the one the scene is in: it is placed by its
own voxel size, on the assumption that the two share an origin and their axes,
and the panel offers writing the missing transform into the extra metadata and
reading it back in. Temporary data only; VELEND_TEST_DEPS may point at unpacked
dependency wheels.
"""
import sys
from pathlib import Path
import os
if os.environ.get('VELEND_TEST_DEPS'):
    sys.path.insert(0, os.environ['VELEND_TEST_DEPS'])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bpy, json, numpy as np, tempfile, time, zarr
import velend
from velend import metadata, ui
from velend.renderer import VolumeSamplerRenderEngine as E
# Register modules directly to avoid downloading the catalogue.
for mod in velend._modules:
    if hasattr(mod, 'register'):
        mod.register()

# Two volumes of one sample registered to each other, and a third the sample
# states nothing about, whose voxels are another size.
A, B, C = '20231027191953', '20231117143551', '20240101000000'
MATRIX = [[0.5, 0.0, 0.0, 10.0], [0.0, 0.5, 0.0, 20.0], [0.0, 0.0, 0.5, 30.0]]


def manifest():
    return {'samples': {'PHercTest': {
        'sample': {'id': 'PHercTest', 'properties': {'type': 'scroll',
            'volume_transforms': [{'from_volume_id': A, 'transforms': [
                {'to_volume_id': B, 'matrix': MATRIX}]}]}},
        'scans': {},
        'volumes': {
            A: {'id': A, 'long_id': A + '-3.240um.zarr'},
            B: {'id': B, 'long_id': B + '-7.910um.zarr'},
            C: {'id': C, 'long_id': C + '-2.000um.zarr'},
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


def wait():
    """Until the volume the fields name is open and set up to render."""
    deadline = time.monotonic() + 10
    while E.get_volume() is None:
        assert time.monotonic() < deadline, E.status()
        time.sleep(.01)
    E.apply_reset()
    E.ensure_grid()


def load(path):
    bpy.context.scene.velend.volume_path = str(path)
    wait()


def scale(um):
    """What `world_to_voxels` is when nothing but the voxel size separates the
    scene's coordinates from the volume's. What a Blender unit is worth in
    micrometers comes from the scene, which "Setup Scene for Volume" changes.
    """
    return np.diag([E.um_per_unit() / um] * 3 + [1.0])


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / 'manifest.json').write_text(json.dumps(manifest()))
    assert metadata.load(str(root / 'manifest.json'), refresh=False)
    volume_a = write_volume(root / (A + '-3.240um.zarr'), 4)
    volume_b = write_volume(root / (B + '-7.910um.zarr'), 8)
    volume_c = write_volume(root / (C + '-2.000um.zarr'), 4)
    settings = bpy.context.scene.velend

    # The scene is in A's frame, and nothing relates C to it.
    load(volume_a)
    assert settings.scene_volume_id == A, settings.scene_volume_id
    assert abs(settings.scene_resolution - 3.240) < 1e-6, settings.scene_resolution

    # So C is placed by its own voxel size, and the scene stays in A's frame:
    # nothing in it moves, and Voxel Size states the volume being rendered
    # while Scene Voxel Size still states what a coordinate means.
    load(volume_c)
    assert settings.scene_volume_id == A, settings.scene_volume_id
    assert abs(settings.resolution - 2.0) < 1e-6, settings.resolution
    assert abs(settings.scene_resolution - 3.240) < 1e-6, settings.scene_resolution
    assert np.allclose(E.physical_transform(), np.eye(4)), E.physical_transform()
    assert np.allclose(E.world_to_voxels, scale(2.0)), E.world_to_voxels

    # The scaffold the panel's first button starts the extra metadata off with
    # states the missing pair without dropping the pair the catalogue had.
    overlay = root / 'overrides.json'
    ui._write_overlay_scaffold(str(overlay), A, C, 3.240 / 2.0)
    assert metadata.load(
        str(root / 'manifest.json'), overlay=str(overlay), refresh=False)
    assert np.allclose(
        metadata.volume_transform(A, C), np.diag([1.62, 1.62, 1.62, 1.0])
    ), metadata.volume_transform(A, C)
    assert metadata.volume_transform(A, B) is not None

    # And it is the transform the pair was being placed by all along: the
    # voxels differ by their sizes alone, which in micrometers is no transform.
    E.rescale()
    assert np.allclose(E.physical_transform(), np.eye(4)), E.physical_transform()
    assert np.allclose(E.world_to_voxels, scale(2.0)), E.world_to_voxels

    # The button itself writes the file and hands it to $EDITOR.
    written = root / 'from-the-button.json'
    ui._overlay_file = lambda: str(written)
    os.environ['EDITOR'] = '/usr/bin/true'
    assert bpy.ops.velend.edit_metadata_overrides(
        from_volume_id=A, to_volume_id=C, scale=1.62) == {'FINISHED'}
    assert json.loads(written.read_text())['samples']['PHercTest']

    # The whole round trip the two buttons are there for: the scene is in A's
    # frame, C is not registered against it, a transform is written into the
    # extra metadata by hand, and the second button reads it back in without
    # waiting for the editor.
    metadata.overlay_path = lambda: str(written)
    metadata.cache_path = lambda: str(root / 'manifest.json')
    assert metadata.load(str(root / 'manifest.json'), refresh=False)
    assert metadata.volume_transform(A, C) is None
    hand_written = json.loads(written.read_text())
    hand_written['samples']['PHercTest']['sample']['properties'][
        'volume_transforms'] = [{'from_volume_id': A, 'transforms': [
            {'to_volume_id': C, 'matrix': [
                [1.62, 0.0, 0.0, 4.0], [0.0, 1.62, 0.0, 0.0],
                [0.0, 0.0, 1.62, 0.0]]}]}]
    written.write_text(json.dumps(hand_written))
    assert bpy.ops.velend.reload_metadata(
        from_volume_id=A, to_volume_id=C) == {'FINISHED'}
    assert metadata.volume_transform(A, C) is not None

    # From then on C renders through it: four of its voxels along x, which is
    # what the matrix translates by, on top of the placement it had.
    E.rescale()
    physical = E.physical_transform()
    assert np.allclose(physical[:3, :3], np.eye(3)), physical
    assert np.allclose(physical[:3, 3], [8.0, 0.0, 0.0]), physical
    expected = scale(2.0)
    expected[:3, 3] = [4.0, 0.0, 0.0]
    assert np.allclose(E.world_to_voxels, expected), E.world_to_voxels
    # And the scene is still in A's frame, at A's voxel size.
    assert settings.scene_volume_id == A, settings.scene_volume_id
    assert abs(settings.scene_resolution - 3.240) < 1e-6, settings.scene_resolution

    # A file half written is worth hearing about, rather than merging what can
    # be read of it and carrying on.
    written.write_text('{"samples": {')
    try:
        bpy.ops.velend.reload_metadata(from_volume_id=A, to_volume_id=C)
        assert False, 'a broken overlay went unreported'
    except RuntimeError as error:
        assert written.name in str(error), error

print('VELEND VOLUME SWITCH SMOKE PASSED')
