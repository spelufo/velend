"""Run in foreground Blender with --factory-startup --python; quits after testing.

Draws the volume shader into an offscreen buffer with and without a shift in
`world_to_voxels`, and checks the drawing moved by exactly what the matrix
says. That is also what pins the uniform block's layout down: the matrix has to
still land where the shader reads it.

Uses a temporary synthetic volume, without saving preferences.
VELEND_TEST_DEPS optionally points at unpacked wheels.
"""
import sys
from pathlib import Path
import os
if os.environ.get('VELEND_TEST_DEPS'):
    sys.path.insert(0, os.environ['VELEND_TEST_DEPS'])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bpy, gpu, numpy as np, tempfile, time, traceback, zarr
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix
import velend
from velend import bricks
from velend.renderer import VolumeSamplerRenderEngine as E
for mod in velend._modules:
    if hasattr(mod, 'register'): mod.register()

# The volume is one level 0 chunk across, so streaming its single brick in is
# all it takes to have the whole quad sample through the level 0 atlas.
SIZE = 64
# One Blender unit per level 0 voxel, so that the numbers below are both. The
# directory names it too, so the scene has nothing to ask about.
VOXEL_SIZE_UM = 1000000.0
# The quad, in world units: a square across the x/y gradient at a fixed depth.
SPAN = float(SIZE)
DEPTH = 32.0
SHIFT = 16.0
WIDTH = HEIGHT = 200
# Where the same voxel ends up on screen once the transform has shifted it.
SHIFT_PIXELS = SHIFT / SPAN * WIDTH
# Columns to check the ramp in. The shader interpolates between voxel centres,
# so these are picked to land near one rather than between two.
COLUMNS = (0, 10, 45, 95, 145)

tmp = tempfile.TemporaryDirectory()
root = Path(tmp.name) / 'volume-1000000um.zarr'
group = zarr.open_group(root, mode='w', zarr_format=2)
group.attrs['multiscales'] = [{'datasets': [{'path': str(i)} for i in range(6)]}]
# A ramp along x, bright enough everywhere that no fragment is discarded.
ramp = (51 + 3 * np.arange(SIZE)).astype('u1')
for i in range(6):
    array = group.create_array(str(i), shape=(SIZE,) * 3, chunks=(32, 32, 32), dtype='u1')
    array[:] = np.broadcast_to(ramp, (SIZE, SIZE, SIZE))
bpy.context.scene.unit_settings.scale_length = 1.0
bpy.context.scene.velend.resolution = VOXEL_SIZE_UM
bpy.context.scene.velend.volume_path = str(root)
E.get_volume()

# The quad's own square, its place in the world, and the projection that puts
# that place back over the whole viewport.
MODEL = Matrix.Translation((0.0, 0.0, DEPTH)) @ Matrix.Diagonal((SPAN, SPAN, 1.0, 1.0))
VIEW_PROJECTION = (
    Matrix.Translation((-1.0, -1.0, 0.0))
    @ Matrix.Diagonal((2.0, 2.0, 1.0, 1.0))
    @ MODEL.inverted()
)
CORNERS = [(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)]


def draw(offscreen, transform):
    """The shader's output over the quad, as a (height, width) array."""
    E.world_to_voxels = transform
    batch = batch_for_shader(E.shader, 'TRIS', {"position": CORNERS}, indices=[(0, 1, 2), (0, 2, 3)])
    with offscreen.bind():
        framebuffer = gpu.state.active_framebuffer_get()
        framebuffer.clear(color=(0.0, 0.0, 0.0, 1.0))
        E.shader.bind()
        for level in bricks.LEVELS:
            E.shader.uniform_sampler("l%dAtlas" % level, E.atlases[level].texture)
            E.shader.uniform_sampler("l%dPageTable" % level, E.atlases[level].page_texture)
        E.update_uniform_buffer(VIEW_PROJECTION, MODEL)
        E.shader.uniform_block("volumeUniforms", E.uniform_buffer)
        batch.draw(E.shader)
        pixels = np.asarray(framebuffer.read_color(0, 0, WIDTH, HEIGHT, 4, 0, 'FLOAT').to_list())
    return pixels[:, :, 0]


def expected(column):
    """What the shader should draw in one column, from the ramp it samples."""
    voxel = SPAN * (column + 0.5) / WIDTH
    lo = int(voxel)
    fraction = voxel - lo
    value = ramp[lo] * (1.0 - fraction) + ramp[min(lo + 1, SIZE - 1)] * fraction
    return (value / 255.0) ** 2


started = time.monotonic()
def tick():
    try:
        if time.monotonic() - started > 20: raise RuntimeError('GPU smoke timed out')
        if E.get_volume() is None: return .1
        E.ensure_gpu_resources()
        assert E.shader is not None, E.status()
        assert np.allclose(E.compute_world_to_voxels(), np.eye(4)), \
            E.compute_world_to_voxels()

        # The quad covers the volume's only level 0 chunk. Ask for it directly
        # rather than through `retarget`, which would need scene geometry.
        residency = E.residencies[0]
        if not residency.slot_of:
            E.wanted[0] = frozenset({0})
            E.loader.request(0, 0, (0, 0, 0))
            E.pump_uploads()
            return .05

        offscreen = gpu.types.GPUOffScreen(WIDTH, HEIGHT, format='RGBA32F')
        try:
            plain = draw(offscreen, np.eye(4))
            shifted = draw(offscreen, np.array([
                [1., 0., 0., SHIFT], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]))
        finally:
            offscreen.free()

        # The ramp reaches the framebuffer as the shader's own arithmetic says,
        # which is only true while every uniform is read where it was written.
        for column in COLUMNS:
            assert abs(plain[HEIGHT // 2, column] - expected(column)) < 0.01, (
                column, plain[HEIGHT // 2, column], expected(column))
        # And the transform moves it by exactly the voxels it translates by.
        overlap = int(WIDTH - SHIFT_PIXELS) - 1
        offset = int(round(SHIFT_PIXELS))
        difference = np.abs(shifted[:, :overlap] - plain[:, offset:offset + overlap]).max()
        assert difference < 0.02, difference
        assert shifted[:, :overlap].min() > plain[:, :overlap].min(), 'the ramp did not move'
        print('VELEND TRANSFORM GPU PASSED', flush=True)
        bpy.ops.wm.quit_blender()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    return None
bpy.app.timers.register(tick, first_interval=.1)
