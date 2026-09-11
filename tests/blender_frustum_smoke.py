"""Run with Blender --background --factory-startup --python-exit-code 1 --python.

View frustum culling of the brick working set: the planes that come out of a
matrix Blender itself built, and the chunks `retarget` drops because of them.
Temporary data only; VELEND_TEST_DEPS may point at unpacked dependency wheels.
"""
import sys
from pathlib import Path
import os
if os.environ.get('VELEND_TEST_DEPS'):
    sys.path.insert(0, os.environ['VELEND_TEST_DEPS'])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bpy, numpy as np, tempfile, time, zarr
import velend
from velend import bricks
from velend import renderer
from velend.renderer import VolumeSamplerRenderEngine as E
for mod in velend._modules:
    if hasattr(mod, 'register'):
        mod.register()

CORE = bricks.BRICK_CORE
# Sixteen level 0 chunks per axis, comfortably more than the frustum margin
# reaches, and a proper pyramid above them so every level in LEVELS has an
# array of its own.
CHUNKS = 16
SIZE = CHUNKS * CORE
SHEET_Z = 8 * CORE
CLIP_START, CLIP_END = 0.5, 100.0


class FakeViewport:
    """Stands in for an engine instance in `live_instances`.

    `view_frusta` reads the matrix off it and `request_redraw` pokes it, which
    is the whole of what the engine's own instances are used for here.
    """

    def __init__(self, matrix):
        self.frustum_matrix = matrix

    def tag_redraw(self):
        pass


def write_volume(root):
    """A pyramid of the right shape, left unwritten.

    `retarget` reads the shapes and nothing else, and the brick reads it
    dispatches land after the assertions, so there is no point filling
    gigabytes of voxels that only the loader would ever look at.
    """
    group = zarr.open_group(root, mode='w', zarr_format=2)
    group.attrs['multiscales'] = [{'datasets': [{'path': str(i)} for i in range(6)]}]
    for i in range(6):
        size = max(SIZE >> i, 1)
        group.create_array(str(i), shape=(size,) * 3, chunks=(64,) * 3, dtype='u1')
    return root


def camera_matrix(location, clip_start=CLIP_START, clip_end=CLIP_END):
    """A perspective matrix built the way `RegionView3D` builds its own.

    Taken off a real camera rather than assembled here, so that it carries
    whatever depth range Blender's own projection code uses -- which is the
    half of `frustum_planes` this file exists to pin down.
    """
    camera = bpy.data.objects['Camera']
    camera.location = location
    # Factory startup aims the camera at the default cube; straight down -Z is
    # what makes the assertions below readable.
    camera.rotation_euler = (0.0, 0.0, 0.0)
    camera.data.clip_start = clip_start
    camera.data.clip_end = clip_end
    camera.data.angle = 0.5
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    window = camera.evaluated_get(depsgraph).calc_matrix_camera(depsgraph, x=960, y=960)
    perspective = window @ camera.matrix_world.inverted()
    return tuple(tuple(row) for row in perspective)


def sheet(z, lo, hi):
    """A flat quad at height `z`, spanning `[lo, hi]` in x and y."""
    mesh = bpy.data.meshes.new('sheet')
    mesh.from_pydata(
        [(lo, lo, z), (hi, lo, z), (hi, hi, z), (lo, hi, z)], [], [(0, 1, 2, 3)])
    mesh.update()
    obj = bpy.data.objects.new('sheet', mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def wanted_total():
    E.retarget(bpy.context.evaluated_depsgraph_get(), bpy.context.scene.cursor.location)
    return sum(len(E.wanted[level]) for level in bricks.LEVELS)


# The camera sits at z = 100 looking down -Z, so "in front" is decreasing z.
matrix = camera_matrix((0.0, 0.0, 100.0))
planes = bricks.frustum_planes(matrix)
assert planes is not None


def visible(*points):
    return list(bricks.boxes_visible([planes], np.asarray(points, dtype=float), 0.0))


# The near and far planes land where the camera's own clipping says, which is
# what says the extraction reads Blender's depth range the right way round. A
# point short of clip_start is in front of the camera and still outside.
assert visible((0.0, 0.0, 100.0 - CLIP_START * 0.8)) == [False]
assert visible((0.0, 0.0, 100.0 - CLIP_START * 1.2)) == [True]
assert visible((0.0, 0.0, 100.0 - CLIP_END * 0.9)) == [True]
assert visible((0.0, 0.0, 100.0 - CLIP_END * 1.1)) == [False]
# Behind the camera, and off to the side of a half-radian cone.
assert visible((0.0, 0.0, 110.0)) == [False]
assert visible((60.0, 0.0, 50.0)) == [False]
assert visible((0.0, 0.0, 50.0)) == [True]

with tempfile.TemporaryDirectory() as tmp:
    settings = bpy.context.scene.velend
    settings.volume_path = str(write_volume(Path(tmp) / 'volume-1000000um.zarr'))
    deadline = time.monotonic() + 30
    while E.get_volume() is None:
        assert time.monotonic() < deadline, E.status()
        time.sleep(.01)
    E.apply_reset()
    E.ensure_grid()
    # One Blender unit per level 0 voxel, so the assertions can be read in
    # chunk coordinates directly.
    settings.resolution = 1000000.0
    E.world_to_voxels = E.compute_world_to_voxels()
    assert abs(E.world_to_voxels[0, 0] - 1.0) < 1e-9, E.world_to_voxels
    assert E.shape_xyz == (SIZE, SIZE, SIZE), E.shape_xyz

    # A sheet across the whole volume, seen from close up by a narrow camera
    # aimed square at one chunk of it, so most of it falls outside the frustum.
    sheet(float(SHEET_Z), 0.0, float(SIZE))
    viewport = FakeViewport(camera_matrix(
        (2.5 * CORE, 2.5 * CORE, SHEET_Z + 100.0), clip_end=1000.0))
    E.live_instances.add(viewport)

    settings.frustum_culling = False
    unculled = wanted_total()
    settings.frustum_culling = True
    culled = wanted_total()
    assert unculled > 0, unculled
    assert culled > 0, culled
    assert culled < unculled, (culled, unculled)

    # The chunk under the camera is kept; the far corner of the sheet is not.
    residency = E.residencies[0]
    keys = residency.keys_of(np.array(
        [[2, 2, SHEET_Z // CORE], [CHUNKS - 1, CHUNKS - 1, SHEET_Z // CORE]],
        dtype=np.int64))
    assert int(keys[0]) in E.wanted[0], sorted(E.wanted[0])
    assert int(keys[1]) not in E.wanted[0], sorted(E.wanted[0])

    # Turning it off puts the far corner back, and an unknown view culls nothing.
    settings.frustum_culling = False
    assert wanted_total() == unculled
    assert int(keys[1]) in E.wanted[0]
    settings.frustum_culling = True
    viewport.frustum_matrix = None
    assert wanted_total() == unculled

    # The debounce: the tick that first sees a new view only records it, and
    # the one after it, with the view unchanged, is what retargets. Orbiting
    # therefore costs nothing until it stops.
    viewport.frustum_matrix = camera_matrix(
        (2.5 * CORE, 2.5 * CORE, SHEET_Z + 100.0), clip_end=1000.0)
    E.last_view_key, E.view_settled = None, False
    renderer._watch_view()
    assert sum(len(E.wanted[level]) for level in bricks.LEVELS) == unculled
    renderer._watch_view()
    assert sum(len(E.wanted[level]) for level in bricks.LEVELS) == culled
    # And a settled view is not retargeted for over and over.
    E.wanted = {level: frozenset() for level in bricks.LEVELS}
    renderer._watch_view()
    assert sum(len(E.wanted[level]) for level in bricks.LEVELS) == 0

    E.loader.shutdown()

print('culled %d of %d wanted chunks' % (unculled - culled, unculled))

print('VELEND FRUSTUM SMOKE PASSED')
