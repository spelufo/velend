"""Run with Blender --background --factory-startup --python-exit-code 1 --python.

VELEND_TEST_DEPS may point at unpacked dependency wheels. Uses temporary data only.
"""
import sys
from pathlib import Path
import os
if os.environ.get('VELEND_TEST_DEPS'):
    sys.path.insert(0, os.environ['VELEND_TEST_DEPS'])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bpy, tempfile, time, zarr
from pathlib import Path
import velend
from velend.renderer import VolumeSamplerRenderEngine as E
# Register modules directly to avoid downloading the catalogue.
for mod in velend._modules:
    if hasattr(mod, 'register'):
        mod.register()
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)/'volume'
    group = zarr.open_group(root, mode='w', zarr_format=2)
    group.attrs['multiscales'] = [{'datasets':[{'path':str(i)} for i in range(6)]}]
    for i in range(6):
        a = group.create_array(str(i), shape=(4,4,4), chunks=(2,2,2), dtype='u1')
        a[:] = 80
    bpy.context.scene.velend.volume_path = str(root)
    assert E.get_volume() is None
    deadline = time.monotonic()+10
    while E.get_volume() is None:
        assert time.monotonic() < deadline, E.status()
        time.sleep(.01)
    assert len(E.pyramid) == 6
    assert bpy.ops.velend.setup_scene() == {'FINISHED'}
    assert bpy.data.objects.get('Cut X')
    assert tuple(bpy.context.scene.cursor.location) != (0,0,0)
    bpy.context.scene.velend.source_url = 'http://127.0.0.1:1'
    assert E.load_key is None and E.pending_reset
    bpy.ops.velend.reload_volume()
    assert E.load_key is None
    bpy.context.scene.velend.source_url = ''
    E.get_volume()
    old = E.load_future
    bpy.context.scene.velend.volume_path = str(root/'missing')
    E.get_volume()
    assert E.load_future is not old
    if not old.cancelled():
        old.result(timeout=10)
    assert E.pyramid is None
for mod in reversed(velend._modules):
    if hasattr(mod, 'unregister'):
        mod.unregister()
print('VELEND SMOKE PASSED')
