import bpy
import importlib
import sys

# NOTE: On my machine that's where blender's python puts packages if I
# `blender_python -m pip install` them, so that's how I installed vesuvius, and how I accepted the
# terms. Extensions don't have it in the path (surely) because they want you to make and distribute
# wheels. TODO: Do that instead, and figure out where the terms file ends up, or it fails silently.
sys.path.append("/Users/spelufo/.local/lib/python3.13/site-packages")
sys.path.append("/Users/spelufo/pro/vesuvius/villa/vesuvius/src")


from . import renderer
from . import commands

_modules = [
	renderer,
	commands,
]

_reload = "_loaded" in locals()
_loaded = True

def register():
	print()
	for mod in _modules:
		if _reload:
			print("Reloading: ", mod)
			importlib.reload(mod)
		if hasattr(mod, "register"):
			getattr(mod, "register")()

def unregister():
	for mod in reversed(_modules):
		if hasattr(mod, "unregister"):
			getattr(mod, "unregister")()
