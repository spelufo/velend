# Unlike the other modules, this one isn't reloaded when running "Reload Scripts".

import zarr


# The pyramid currently open, and the path it was opened from. Module level so
# that reloading the add-on's other modules doesn't drop the handles.
volume = None
volume_path = None


def open_volume(path):
	"""Open the OME-Zarr pyramid as a list of (Z, Y, X) arrays, finest first.

	The arrays are only ever read, which zarr does without touching any shared
	mutable state, so the brick loader threads all share these handles.
	"""
	group = zarr.open_group(path, mode="r")
	datasets = group.attrs["multiscales"][0]["datasets"]
	return [group[dataset["path"]] for dataset in datasets]


def get_volume(path):
	"""The pyramid at `path`, reopening it when the path changed."""
	global volume, volume_path
	if volume is None or path != volume_path:
		# Only rebound once the open succeeded, so a bad path leaves whatever
		# was already open alone.
		volume = open_volume(path)
		volume_path = path
	return volume


def close_volume():
	global volume, volume_path
	volume = None
	volume_path = None
