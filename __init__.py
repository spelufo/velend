import bpy
import importlib

from . import renderer

_modules = [
	renderer
]

_reload = "_loaded" in locals()
_loaded = True

def register():
	for mod in _modules:
		if _reload:
			importlib.reload(mod)
		if hasattr(mod, "register"):
			getattr(mod, "register")()

def unregister():
	for mod in reversed(_modules):
		if hasattr(mod, "unregister"):
			getattr(mod, "unregister")()
