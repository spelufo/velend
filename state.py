# Unlike the other modules, this one isn't reloaded when running "Reload Scripts".

import zarr


volpath = "/Users/spelufo/.VC3D/remote_cache/open_data/volumes/PHerc1203/20250820131727-9.362um-1.2m-113keV-masked.zarr-adf63bbdf658dd8f"
# volpath = "/Users/spelufo/.VC3D/remote_cache/open_data/volumes/PHerc1203/20250820131727-surface-20260413222639-surface-m7-L0-th0.2.zarr-bda84c333af9ce4a"
resolution = 9.362
volume = None

def open_volume():
	"""Open the OME-Zarr pyramid as a list of (Z, Y, X) arrays, finest first.

	The arrays are only ever read, which zarr does without touching any shared
	mutable state, so the brick loader threads all share these handles.
	"""
	group = zarr.open_group(volpath, mode="r")
	datasets = group.attrs["multiscales"][0]["datasets"]
	return [group[dataset["path"]] for dataset in datasets]

def get_volume():
	global volume
	if volume is None:
		volume = open_volume()
	return volume
