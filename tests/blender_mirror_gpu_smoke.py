"""Run in foreground Blender with --factory-startup --python; quits after testing.

Uses a temporary synthetic volume and small atlases, without saving preferences.
VELEND_TEST_DEPS optionally points at unpacked wheels.
"""
import sys
from pathlib import Path
import os
if os.environ.get('VELEND_TEST_DEPS'):
    sys.path.insert(0, os.environ['VELEND_TEST_DEPS'])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bpy, tempfile, time, zarr, traceback
from pathlib import Path
import velend
from velend.renderer import VolumeSamplerRenderEngine as E
from velend import bricks, uv_renderer
for mod in velend._modules:
    if hasattr(mod, 'register'): mod.register()
bricks.SLOTS_PER_AXIS = {level: 1 for level in bricks.LEVELS}
bricks.SLOT_COUNTS = {level: 1 for level in bricks.LEVELS}
bricks.ATLAS_DIMS = {level: bricks.BRICK_SIZE for level in bricks.LEVELS}
tmp = tempfile.TemporaryDirectory()
root = Path(tmp.name)/'volume-9.362um.zarr'
group = zarr.open_group(root, mode='w', zarr_format=2)
group.attrs['multiscales'] = [{'datasets':[{'path':str(i)} for i in range(6)]}]
for i in range(6):
    a = group.create_array(str(i), shape=(64,64,64), chunks=(32,32,32), dtype='u1')
    a[:] = 80
bpy.context.scene.velend.volume_path = str(root)
E.get_volume()
phase = 0
started = time.monotonic()
area = next(a for a in bpy.context.screen.areas if a.type == 'VIEW_3D')
def tick():
    global phase, started
    try:
        if time.monotonic() - started > 20: raise RuntimeError('GPU smoke timed out')
        if phase == 0:
            if E.get_volume() is None: return .1
            assert bpy.ops.velend.setup_scene() == {'FINISHED'}
            phase = 1
            return 2
        if phase == 1:
            assert E.atlases and E.shader is not None, E.status()
            print('VELEND 3D GPU PASSED', flush=True)
            obj = bpy.data.objects['Cut Z']
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            obj.data.uv_layers.new(name='UVMap')
            for loop, uv in zip(obj.data.uv_layers.active.data, [(0,0),(1,0),(1,1),(0,1)]): loop.uv = uv
            area.type = 'IMAGE_EDITOR'
            area.spaces.active.mode = 'UV'
            phase = 2
            return 2
        assert uv_renderer._shader is not None, 'UV shader was not drawn'
        print('VELEND UV GPU PASSED', flush=True)
        bpy.ops.wm.quit_blender()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    return None
bpy.app.timers.register(tick, first_interval=.1)
