"""Run with Blender --background --factory-startup --python-exit-code 1 --python.

Switching to a volume nothing registers against the one the scene is in: which
of those switches is one to ask about, since it changes the size of the voxels
everything in the scene is placed in, and what putting one back and confirming
one do. Temporary data only; VELEND_TEST_DEPS may point at unpacked dependency
wheels.
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


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / 'manifest.json').write_text(json.dumps(manifest()))
    assert metadata.load(str(root / 'manifest.json'), refresh=False)
    volume_a = write_volume(root / (A + '-3.240um.zarr'), 4)
    volume_b = write_volume(root / (B + '-7.910um.zarr'), 8)
    volume_c = write_volume(root / (C + '-2.000um.zarr'), 4)
    # Unregistered like C, and named as the size the scene is already in.
    same = write_volume(root / 'someone-elses-3.240um.zarr', 4)
    settings = bpy.context.scene.velend

    # The scene is in A's voxels, and nothing relates C to them.
    load(volume_a)
    assert settings.scene_volume_id == A, settings.scene_volume_id
    assert abs(settings.resolution - 3.240) < 1e-6, settings.resolution
    before = (settings.last_volume_path, settings.last_source_url,
              settings.scene_volume_id, settings.resolution)

    # So switching to C is a switch to ask about: it took the scene from A's
    # 3.240 um voxels into C's 2 um ones with nothing to carry the scene's own
    # contents across.
    load(volume_c)
    assert settings.scene_volume_id == C, settings.scene_volume_id
    assert abs(settings.resolution - 2.0) < 1e-6, settings.resolution
    target = ui._rescaling_switch(settings, before)
    assert target == {'volume_path': str(volume_c), 'source_url': '',
                      'volume_id': C, 'resolution': settings.resolution}, target

    # Putting it back is what the dialog's Cancel leaves behind.
    ui._set_volume(settings, *before)
    wait()
    assert settings.volume_path == str(volume_a), settings.volume_path
    assert settings.scene_volume_id == A, settings.scene_volume_id
    assert abs(settings.resolution - 3.240) < 1e-6, settings.resolution
    assert np.allclose(E.volume_transform, np.eye(4)), E.volume_transform

    # A volume of the same voxel size is nothing to ask about: unregistered or
    # not, the scene keeps meaning what it meant.
    load(same)
    assert ui._rescaling_switch(settings, before) is None
    assert settings.scene_volume_id == '', settings.scene_volume_id
    assert abs(settings.resolution - 3.240) < 1e-6, settings.resolution

    # Nor is one the sample registers against the volume the scene is in: the
    # scene stays where it is and the matrix does the moving.
    load(volume_a)
    before = (settings.last_volume_path, settings.last_source_url,
              settings.scene_volume_id, settings.resolution)
    load(volume_b)
    assert settings.scene_volume_id == A, settings.scene_volume_id
    assert ui._rescaling_switch(settings, before) is None

    # Nor is a scene in no volume's voxels: they become the first one loaded.
    ui._set_volume(settings, str(volume_a), '', '', 3.240)
    load(volume_c)
    assert ui._rescaling_switch(
        settings, (str(volume_a), '', '', 3.240)) is None

    # Answering the dialog takes the scene into C's voxels, as the fields
    # having named it would have.
    ui._set_volume(settings, str(volume_a), '', A, 3.240)
    assert bpy.ops.velend.confirm_volume_switch(
        'EXEC_DEFAULT', volume_path=str(volume_c), source_url='',
        volume_id=C, resolution=2.0,
    ) == {'FINISHED'}
    wait()
    assert settings.volume_path == str(volume_c), settings.volume_path
    assert settings.last_volume_path == str(volume_c), settings.last_volume_path
    assert settings.scene_volume_id == C, settings.scene_volume_id
    assert abs(settings.resolution - 2.0) < 1e-6, settings.resolution
    assert np.allclose(E.volume_transform, np.eye(4)), E.volume_transform

    # The scaffold the dialog's third button starts the file off with states
    # the missing pair without dropping the pair the catalogue already had.
    overlay = root / 'overrides.json'
    ui._write_overlay_scaffold(str(overlay), A, C, 3.240 / 2.0)
    assert metadata.load(
        str(root / 'manifest.json'), overlay=str(overlay), refresh=False)
    assert np.allclose(
        metadata.volume_transform(A, C), np.diag([1.62, 1.62, 1.62, 1.0])
    ), metadata.volume_transform(A, C)
    assert metadata.volume_transform(A, B) is not None

    # And the button itself writes it and hands it to $EDITOR.
    written = root / 'from-the-button.json'
    ui._overlay_file = lambda: str(written)
    os.environ['EDITOR'] = '/usr/bin/true'
    assert bpy.ops.velend.edit_metadata_overrides(
        from_volume_id=A, to_volume_id=C, scale=1.62) == {'FINISHED'}
    assert json.loads(written.read_text())['samples']['PHercTest']

    # The whole round trip the dialog is there for: the scene is in A's
    # voxels, C is not registered against them, the transform is written into
    # the extra metadata, and the button reads it back in without waiting for
    # the editor.
    metadata.overlay_path = lambda: str(written)
    metadata.cache_path = lambda: str(root / 'manifest.json')
    assert metadata.load(str(root / 'manifest.json'), refresh=False)
    assert metadata.volume_transform(A, C) is None
    ui._set_volume(settings, str(volume_a), '', A, 3.240)
    assert bpy.ops.velend.reload_metadata(
        from_volume_id=A, to_volume_id=C) == {'FINISHED'}
    assert metadata.volume_transform(A, C) is not None

    # Confirming then makes the ordinary switch, through the transform: the
    # scene stays in A's voxels and the volume renders through the matrix.
    assert bpy.ops.velend.confirm_volume_switch(
        'EXEC_DEFAULT', volume_path=str(volume_c), source_url='',
        volume_id=C, resolution=2.0,
    ) == {'FINISHED'}
    wait()
    assert settings.volume_path == str(volume_c), settings.volume_path
    assert settings.scene_volume_id == A, settings.scene_volume_id
    assert abs(settings.resolution - 3.240) < 1e-6, settings.resolution
    assert np.allclose(
        E.volume_transform, np.diag([1.62, 1.62, 1.62, 1.0])), E.volume_transform

    # A file half written is worth hearing about, rather than merging what can
    # be read of it and carrying on.
    written.write_text('{"samples": {')
    try:
        bpy.ops.velend.reload_metadata(from_volume_id=A, to_volume_id=C)
        assert False, 'a broken overlay went unreported'
    except RuntimeError as error:
        assert written.name in str(error), error

print('VELEND VOLUME SWITCH SMOKE PASSED')
