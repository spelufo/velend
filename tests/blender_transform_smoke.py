"""Run with Blender --background --factory-startup --python-exit-code 1 --python.

Switching between volumes of one sample: the affine that keeps the scene's
coordinates meaning the same place in all of them, and the voxel size each is
placed by. Temporary data only; VELEND_TEST_DEPS may point at unpacked
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
from velend import metadata, tifxyz, ui
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


def wait():
    """Until the volume the fields name is open and set up to render."""
    deadline = time.monotonic() + 10
    while E.get_volume() is None:
        assert time.monotonic() < deadline, E.status()
        time.sleep(.01)
    E.apply_reset()
    E.ensure_grid()


def load(path):
    """Point the scene at a volume and wait for it to open."""
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
    plain = write_volume(root / 'someone-elses-2.000um.zarr', 4)
    unnamed = write_volume(root / 'someone-elses.zarr', 4)
    settings = bpy.context.scene.velend

    # The first volume loads as itself: the scene's coordinates are its frame,
    # and a Blender unit is worth what its voxel size says.
    load(volume_a)
    assert settings.scene_volume_id == A, settings.scene_volume_id
    assert abs(settings.resolution - 3.240) < 1e-6, settings.resolution
    assert abs(settings.scene_resolution - 3.240) < 1e-6, settings.scene_resolution
    assert np.allclose(E.physical_transform(), np.eye(4)), E.physical_transform()
    assert np.allclose(E.world_to_voxels, scale(3.240)), E.world_to_voxels
    assert bpy.ops.velend.setup_scene() == {'FINISHED'}
    cursor = tuple(bpy.context.scene.cursor.location)

    # The second renders through the matrix registered for the pair, and the
    # scene stays in the first one's frame: Voxel Size states the volume being
    # rendered, and the scene's own stays what it was.
    load(volume_b)
    assert settings.scene_volume_id == A, settings.scene_volume_id
    assert abs(settings.resolution - 7.910) < 1e-6, settings.resolution
    assert abs(settings.scene_resolution - 3.240) < 1e-6, settings.scene_resolution
    # The metadata states the pair in each other's voxels; the registration the
    # scene is placed through is that matrix in micrometers.
    physical = E.physical_transform()
    assert np.allclose(physical[:3, :3], M[:3, :3] * (7.910 / 3.240)), physical
    assert np.allclose(physical[:3, 3], M[:3, 3] * 7.910), physical
    # One voxel of A, in Blender units: what the scene's coordinates are in.
    unit = 3.240 / E.um_per_unit()
    assert np.allclose(E.to_voxels(np.array([2., 4., 6.]) * unit), [11., 21., 30.75])
    assert np.allclose(
        E.to_voxels(np.array([[0., 0., 0.], [2., 4., 6.]]) * unit),
        [[10., 20., 30.], [11., 21., 30.75]])
    # Volume B spans 8 voxels of its own per axis, which the matrix scales up
    # by 2, 4 and 8 in the scene's.
    low, high = E.world_bounds()
    assert np.allclose(low, np.array([-20., -80., -240.]) * unit), low
    assert np.allclose(high, np.array([-4., -48., -176.]) * unit), high
    # Nothing moved the geometry: it means the same place in both volumes.
    assert tuple(bpy.context.scene.cursor.location) == cursor

    # Imports default to the scene volume even while another volume is being
    # rendered. A tifxyz made in A therefore stays in A's coordinates, and the
    # renderer carries it through the registration when sampling B.
    surface_dir = root / 'surface'
    surface_points = np.array([
        [[2., 4., 6.], [3., 4., 6.]],
        [[2., 5., 6.], [3., 5., 6.]],
    ], dtype=np.float32)
    tifxyz.write_surface(
        surface_dir,
        surface_points,
        np.ones((2, 2), dtype=np.float32),
        {},
        {},
        'scene-surface',
        (1.0, 1.0),
    )
    assert bpy.ops.velend.import_tifxyz(
        directory=str(surface_dir), voxel_size=3.240) == {'FINISHED'}
    surface = bpy.context.active_object
    surface_world = np.array(surface.data.vertices[0].co)
    assert np.allclose(surface_world, surface_points[0, 0] * unit), surface_world
    assert np.allclose(E.to_voxels(surface_world), [11., 21., 30.75])
    stored = np.asarray(surface['velend_tifxyz_placement']).reshape(4, 4)
    assert np.allclose(stored, np.diag([unit, unit, unit, 1.])), stored
    scene_export = root / 'scene-export'
    assert bpy.ops.velend.export_tifxyz(filepath=str(scene_export)) == {'FINISHED'}
    exported = np.stack([
        tifxyz.read_page(scene_export / (axis + '.tif')) for axis in 'xyz'
    ], axis=-1)
    assert np.allclose(exported, surface_points), exported

    # Export can instead express the same unchanged geometry in the volume
    # currently being rendered, bypassing the placement retained at import.
    current_export = root / 'current-export'
    assert bpy.ops.velend.export_tifxyz(
        filepath=str(current_export), use_current_volume_coordinates=True
    ) == {'FINISHED'}
    exported = np.stack([
        tifxyz.read_page(current_export / (axis + '.tif')) for axis in 'xyz'
    ], axis=-1)
    expected = surface_points @ M[:3, :3].T + M[:3, 3]
    assert np.allclose(exported, expected), exported
    bpy.data.objects.remove(surface, do_unlink=True)

    # The explicit alternative retains the old behavior: points are B voxels
    # and are transformed back into the frame of A for storage in the scene.
    assert bpy.ops.velend.import_tifxyz(
        directory=str(surface_dir), voxel_size=7.910,
        coordinate_space='RENDERED') == {'FINISHED'}
    surface = bpy.context.active_object
    surface_world = np.array(surface.data.vertices[0].co)
    assert np.allclose(E.to_voxels(surface_world), surface_points[0, 0]), surface_world
    rendered_export = root / 'rendered-export'
    assert bpy.ops.velend.export_tifxyz(filepath=str(rendered_export)) == {'FINISHED'}
    exported = np.stack([
        tifxyz.read_page(rendered_export / (axis + '.tif')) for axis in 'xyz'
    ], axis=-1)
    assert np.allclose(exported, surface_points), exported
    bpy.data.objects.remove(surface, do_unlink=True)

    # Umbilici use the same choice. A size declared in the JSON wins over the
    # operator's value, while the default still says that its frame is A.
    umbilicus_path = root / 'umbilicus.json'
    umbilicus_path.write_text(json.dumps({
        'control_points': [{'x': 2., 'y': 4., 'z': 6.},
                           {'x': 3., 'y': 5., 'z': 7.}],
        'metadata': {'voxelsize_um': 3.240},
    }))
    assert bpy.ops.velend.import_umbilicus(
        filepath=str(umbilicus_path), voxel_size=99.) == {'FINISHED'}
    axis = bpy.context.active_object
    axis_world = np.array(axis.data.vertices[0].co)
    assert np.allclose(axis_world, np.array([2., 4., 6.]) * unit), axis_world
    assert abs(axis['velend_umbilicus_voxel_size'] - 3.240) < 1e-6
    bpy.data.objects.remove(axis, do_unlink=True)

    rendered_umbilicus = root / 'rendered-umbilicus.json'
    rendered_umbilicus.write_text(json.dumps({
        'control_points': [{'x': 2., 'y': 4., 'z': 6.},
                           {'x': 3., 'y': 5., 'z': 7.}],
    }))
    assert bpy.ops.velend.import_umbilicus(
        filepath=str(rendered_umbilicus), voxel_size=7.910,
        coordinate_space='RENDERED') == {'FINISHED'}
    axis = bpy.context.active_object
    axis_world = np.array(axis.data.vertices[0].co)
    assert np.allclose(E.to_voxels(axis_world), [2., 4., 6.]), axis_world
    bpy.data.objects.remove(axis, do_unlink=True)

    # And back, through the inverse of the only direction the sample states.
    load(volume_a)
    assert settings.scene_volume_id == A
    assert np.allclose(E.physical_transform(), np.eye(4)), E.physical_transform()

    # A volume nothing relates to the scene's leaves the scene in the frame it
    # is in and is placed by its own voxel size, the two taken to share an
    # origin and their axes. A Blender unit is worth the same micrometers it
    # was, so nothing already placed moves or changes size.
    load(plain)
    assert settings.scene_volume_id == A, settings.scene_volume_id
    assert abs(settings.resolution - 2.0) < 1e-6, settings.resolution
    assert abs(settings.scene_resolution - 3.240) < 1e-6, settings.scene_resolution
    assert np.allclose(E.physical_transform(), np.eye(4)), E.physical_transform()
    assert np.allclose(E.world_to_voxels, scale(2.0)), E.world_to_voxels
    assert tuple(bpy.context.scene.cursor.location) == cursor

    # So going back to a volume of the pair picks the registration up again,
    # rather than having re-anchored the scene on the way through.
    load(volume_b)
    assert settings.scene_volume_id == A, settings.scene_volume_id
    assert np.allclose(E.to_voxels(np.array([2., 4., 6.]) * unit), [11., 21., 30.75])

    # A volume neither the metadata nor its own name states the voxel size of
    # is one to ask about. The timer that asks is the app's, which the
    # background Blender running this script never reaches, so this is what it
    # would have done: say so, and leave the volume at the size last set.
    load(unnamed)
    assert ui._pending_ask == bpy.context.scene.name, ui._pending_ask
    ui._check_resolution()
    assert abs(settings.resolution - 7.910) < 1e-6, settings.resolution

    # Answering states it, and the volume is placed by the answer. The scene is
    # in A's frame, so what a coordinate in it means is untouched.
    assert bpy.ops.velend.set_resolution('EXEC_DEFAULT', resolution=1.5) == {'FINISHED'}
    assert abs(settings.resolution - 1.5) < 1e-6, settings.resolution
    assert abs(settings.scene_resolution - 3.240) < 1e-6, settings.scene_resolution
    assert np.allclose(E.world_to_voxels, scale(1.5)), E.world_to_voxels

    # A file saved before the scene's frame stated a voxel size of its own:
    # Voxel Size was the frame's then, and opening the file is the only chance
    # to say so -- the fields are written without their update callbacks.
    settings.volume_path = str(volume_b)
    settings.scene_volume_id = A
    settings.scene_resolution = 0.0
    settings.resolution = 3.240
    saved = root / 'legacy.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(saved))
    bpy.ops.wm.open_mainfile(filepath=str(saved))
    settings = bpy.context.scene.velend
    assert settings.scene_volume_id == A, settings.scene_volume_id
    assert abs(settings.scene_resolution - 3.240) < 1e-6, settings.scene_resolution
    assert abs(settings.resolution - 7.910) < 1e-6, settings.resolution
    # Which leaves the scene placing B exactly where the file always had it.
    wait()
    assert np.allclose(
        E.to_voxels(np.array([2., 4., 6.]) * 3.240 / E.um_per_unit()),
        [11., 21., 30.75])

print('VELEND TRANSFORM SMOKE PASSED')
