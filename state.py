# Unlike the other modules, this one isn't reloaded when running "Reload Scripts".

from vesuvius import Volume


volpath = "/Users/spelufo/.VC3D/remote_cache/open_data/volumes/PHerc1203/20250820131727-9.362um-1.2m-113keV-masked.zarr-adf63bbdf658dd8f"
# volpath = "/Users/spelufo/.VC3D/remote_cache/open_data/volumes/PHerc1203/20250820131727-surface-20260413222639-surface-m7-L0-th0.2.zarr-bda84c333af9ce4a"
resolution = 9.362
volume = None

def new_volume():
	# Brick loader threads take one of these each, since `Volume` makes no
	# threadsafety promise and sharing the singleton would rely on one.
	volume = Volume(type="zarr", path=volpath, return_as_type='float32')
	volume.resolution = resolution
	return volume

def get_volume():
	global volume
	if volume is None:
		volume = new_volume()
	return volume
