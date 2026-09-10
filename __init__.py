import bpy
import importlib

# zarr and its dependencies come from the wheels in `wheels/`, which Blender
# unpacks and puts on the path itself, in a development checkout as much as in
# an installed extension. `scripts/develop.py` and `scripts/build.py` fill that
# directory and list its contents in the manifest.

from . import metadata
from . import bricks
from . import atlas
from . import renderer
from . import uv_renderer
from . import tifxyz
from . import umbilicus
from . import volpkg
from . import commands
from . import ui

_modules = [
	bricks,
	atlas,
	renderer,
	uv_renderer,
	tifxyz,
	umbilicus,
	volpkg,
	commands,
	ui,
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
	# Not one of the modules above: like `state`, it holds what a reload should
	# keep, so it is only fetched and parsed the first time we register. It
	# goes last because it reads the add-on preferences `ui` registers.
	if not metadata.SAMPLES:
		metadata.load_in_background()

def unregister():
	for mod in reversed(_modules):
		if hasattr(mod, "unregister"):
			getattr(mod, "unregister")()
